import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { gunzip } from "node:zlib";
import { promisify } from "node:util";
import { pathToFileURL } from "node:url";

const gunzipAsync = promisify(gunzip);

class ProductionAdapterError extends Error {
  constructor(code, message, details = {}) {
    super(`${code}: ${message}`);
    this.name = "ProductionAdapterError";
    this.code = code;
    this.details = details;
  }
}

function parseBooleanArgument(value, name) {
  const token = String(value).trim().toLowerCase();
  if (token === "true") return true;
  if (token === "false") return false;
  throw new ProductionAdapterError("INVALID_ARGUMENT", `--${name} must be true or false`);
}

function parseArgs(argv) {
  const values = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith("--")) {
      throw new ProductionAdapterError("INVALID_ARGUMENT", `Unexpected argument: ${token}`);
    }
    const value = argv[index + 1];
    if (!value || value.startsWith("--")) {
      throw new ProductionAdapterError("INVALID_ARGUMENT", `Missing value for ${token}`);
    }
    values[token.slice(2)] = value;
    index += 1;
  }
  for (const required of ["postrun-root", "config", "output-dir"]) {
    if (!values[required]) {
      throw new ProductionAdapterError("INVALID_ARGUMENT", `--${required} is required`);
    }
  }
  return {
    postrunRoot: path.resolve(values["postrun-root"]),
    configPath: path.resolve(values.config),
    outputDir: path.resolve(values["output-dir"]),
    overwrite: values.overwrite === undefined
      ? false
      : parseBooleanArgument(values.overwrite, "overwrite"),
  };
}

function isInside(base, target) {
  const relative = path.relative(base, target);
  return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
}

function resolveSource(root, relativePath) {
  const target = path.resolve(root, relativePath);
  if (!isInside(root, target)) {
    throw new ProductionAdapterError(
      "SOURCE_OUTSIDE_POSTRUN",
      `Configured source resolves outside the postrun root: ${relativePath}`,
    );
  }
  return target;
}

function scanCsv(text, onRow) {
  const input = String(text).replace(/^\uFEFF/, "");
  let field = "";
  let row = [];
  let quoted = false;
  let rowNumber = 0;
  for (let index = 0; index < input.length; index += 1) {
    const character = input[index];
    if (quoted) {
      if (character === '"') {
        if (input[index + 1] === '"') {
          field += '"';
          index += 1;
        } else {
          quoted = false;
        }
      } else {
        field += character;
      }
      continue;
    }
    if (character === '"' && field.length === 0) {
      quoted = true;
    } else if (character === ",") {
      row.push(field);
      field = "";
    } else if (character === "\r" || character === "\n") {
      row.push(field);
      field = "";
      rowNumber += 1;
      onRow(row, rowNumber);
      row = [];
      if (character === "\r" && input[index + 1] === "\n") index += 1;
    } else {
      field += character;
    }
  }
  if (quoted) {
    throw new ProductionAdapterError("INVALID_CSV", "CSV ended inside a quoted field");
  }
  if (field.length || row.length) {
    row.push(field);
    onRow(row, rowNumber + 1);
  }
}

function inspectCsvText(text, label, expected = {}, rowVisitor = null) {
  let headers = null;
  let rows = 0;
  scanCsv(text, (row, rowNumber) => {
    if (rowNumber === 1) {
      headers = row.map((value) => value.trim());
      if (headers.length === 0 || headers.some((header) => !header)) {
        throw new ProductionAdapterError("INVALID_HEADERS", `${label} has blank headers`);
      }
      if (new Set(headers).size !== headers.length) {
        throw new ProductionAdapterError("INVALID_HEADERS", `${label} has duplicate headers`);
      }
      return;
    }
    if (!headers) throw new ProductionAdapterError("EMPTY_TABLE", `${label} is empty`);
    if (row.length !== headers.length) {
      throw new ProductionAdapterError(
        "RAGGED_CSV",
        `${label} row ${rowNumber} has ${row.length} fields; expected ${headers.length}`,
      );
    }
    rows += 1;
    if (rowVisitor) rowVisitor(row, headers, rows);
  });
  if (!headers) throw new ProductionAdapterError("EMPTY_TABLE", `${label} is empty`);
  if (expected.rows !== undefined && rows !== expected.rows) {
    throw new ProductionAdapterError(
      "ROW_COUNT_MISMATCH",
      `${label} has ${rows} rows; expected ${expected.rows}`,
    );
  }
  if (expected.columns !== undefined && headers.length !== expected.columns) {
    throw new ProductionAdapterError(
      "COLUMN_COUNT_MISMATCH",
      `${label} has ${headers.length} columns; expected ${expected.columns}`,
    );
  }
  return { headers, rows, columns: headers.length };
}

function objectsFromCsv(text, label, expected = {}) {
  let headers = null;
  const rows = [];
  const stats = inspectCsvText(text, label, expected, (row, rowHeaders) => {
    headers = rowHeaders;
    rows.push(Object.fromEntries(headers.map((header, index) => [header, row[index]])));
  });
  return { ...stats, rows };
}

function requireColumns(headers, required, label) {
  const missing = required.filter((column) => !headers.includes(column));
  if (missing.length) {
    throw new ProductionAdapterError(
      "MISSING_REQUIRED_COLUMNS",
      `${label} is missing required columns: ${missing.join(", ")}`,
    );
  }
}

function csvEscape(value) {
  if (value === null || value === undefined) return "";
  const text = String(value);
  if (/[",\r\n]/.test(text)) return `"${text.replaceAll('"', '""')}"`;
  return text;
}

function serializeCsv(headers, rows) {
  const lines = [headers.map(csvEscape).join(",")];
  for (const row of rows) {
    const values = Array.isArray(row)
      ? row
      : headers.map((header) => row[header] ?? "");
    lines.push(values.map(csvEscape).join(","));
  }
  return `${lines.join("\r\n")}\r\n`;
}

function pivotEndpointLongToWide(rows, options) {
  const keyColumns = options.keyColumns ?? [];
  const metadataColumns = options.endpointMetadataColumns
    ?? (options.finiteCountColumn ? [options.finiteCountColumn] : []);
  if (!keyColumns.length) {
    throw new ProductionAdapterError("INVALID_PIVOT_CONFIG", "Pivot keyColumns cannot be empty");
  }
  const endpointColumn = options.endpointColumn ?? "endpoint";
  const valueColumn = options.valueColumn ?? "value";
  const endpoints = [...new Set(rows.map((row) => row[endpointColumn]))].sort();
  if (!endpoints.length || endpoints.some((endpoint) => !endpoint)) {
    throw new ProductionAdapterError("INVALID_PIVOT_VALUE", "Pivot contains a blank endpoint");
  }
  const grouped = new Map();
  for (const sourceRow of rows) {
    for (const column of [...keyColumns, endpointColumn, valueColumn, ...metadataColumns]) {
      if (!Object.hasOwn(sourceRow, column)) {
        throw new ProductionAdapterError(
          "MISSING_REQUIRED_COLUMNS",
          `Pivot input is missing required column ${column}`,
        );
      }
    }
    const key = JSON.stringify(keyColumns.map((column) => sourceRow[column]));
    let group = grouped.get(key);
    if (!group) {
      group = {
        keyValues: keyColumns.map((column) => sourceRow[column]),
        endpoints: new Map(),
      };
      grouped.set(key, group);
    }
    const endpoint = sourceRow[endpointColumn];
    if (group.endpoints.has(endpoint)) {
      throw new ProductionAdapterError(
        "DUPLICATE_PIVOT_VALUE",
        `Duplicate endpoint ${endpoint} for pivot key ${key}`,
      );
    }
    group.endpoints.set(endpoint, {
      value: sourceRow[valueColumn],
      metadata: Object.fromEntries(
        metadataColumns.map((column) => [column, sourceRow[column]]),
      ),
    });
  }
  const metadataHeaders = metadataColumns.flatMap((column) =>
    endpoints.map((endpoint) => `${column}__${endpoint}`));
  const headers = [...keyColumns, ...metadataHeaders, ...endpoints];
  const outputRows = [...grouped.values()].map((group) => [
    ...group.keyValues,
    ...metadataColumns.flatMap((column) =>
      endpoints.map((endpoint) => group.endpoints.get(endpoint)?.metadata[column] ?? "")),
    ...endpoints.map((endpoint) => group.endpoints.get(endpoint)?.value ?? ""),
  ]);
  return { headers, rows: outputRows, endpoints };
}

function truthy(value) {
  return ["true", "1", "yes", "y"].includes(String(value ?? "").trim().toLowerCase());
}

function aggregateControlSpots(spotText, expected) {
  const aggregate = new Map();
  let indices = null;
  const stats = inspectCsvText(spotText, "corrected spots", expected, (row, headers) => {
    if (!indices) {
      requireColumns(headers, [
        "image_key", "is_control", "eligible_for_sampling", "sampled_in_analysis",
        "null_usable", "qki_threshold_positive_q90", "qki_threshold_positive_q95",
        "qki_threshold_positive_q99",
      ], "corrected spots");
      indices = Object.fromEntries(headers.map((header, index) => [header, index]));
    }
    if (!truthy(row[indices.is_control])) return;
    const imageKey = row[indices.image_key];
    const result = aggregate.get(imageKey) ?? {
      spot_rows: 0,
      eligible_spots: 0,
      sampled_spots: 0,
      null_usable: 0,
      q90_positive: 0,
      q95_positive: 0,
      q99_positive: 0,
    };
    result.spot_rows += 1;
    result.eligible_spots += Number(truthy(row[indices.eligible_for_sampling]));
    result.sampled_spots += Number(truthy(row[indices.sampled_in_analysis]));
    result.null_usable += Number(truthy(row[indices.null_usable]));
    result.q90_positive += Number(truthy(row[indices.qki_threshold_positive_q90]));
    result.q95_positive += Number(truthy(row[indices.qki_threshold_positive_q95]));
    result.q99_positive += Number(truthy(row[indices.qki_threshold_positive_q99]));
    aggregate.set(imageKey, result);
  });
  return { aggregate, stats };
}

function deriveControlRows(ledgerRows, spotAggregate) {
  return ledgerRows.map((row) => {
    const calls = spotAggregate.get(row.image_key) ?? {
      spot_rows: 0,
      eligible_spots: 0,
      sampled_spots: 0,
      null_usable: 0,
      q90_positive: 0,
      q95_positive: 0,
      q99_positive: 0,
    };
    const signalStatus = String(row.control_signal_qc_status ?? "").toLowerCase();
    const designation = String(row.designation ?? "").toLowerCase();
    let status = "PASS";
    let notes = "Secondary-only control retained for descriptive QC; excluded from biological inference.";
    if (signalStatus === "expected_negative_zero_spot" || designation.includes("expected-negative zero")) {
      status = "PASS";
      notes = "Expected-negative structural zero: zero MIAT calls is the intended control outcome; excluded from biological inference.";
    } else if (signalStatus === "artifact_bearing_miat_calls" || designation.includes("artifact-bearing")) {
      status = "REVIEW";
      notes = "Artifact-bearing MIAT calls; retain as a diagnostic anomaly and exclude from biological inference.";
    } else if (signalStatus && signalStatus !== "recorded_signal_qc_pass") {
      status = "REVIEW";
      notes = `Control requires review from canonical signal call: ${row.control_signal_qc_status}.`;
    }
    return {
      image_key: row.image_key,
      image: row.image,
      slide: row.slide,
      selected_z_1based: row.selected_z_1based,
      voxel_xy_nm: row.voxel_xy_nm,
      miat_call_count: row.miat_call_count,
      qki_background_median_raw: row.qki_background_median_raw,
      qki_background_p95_raw: row.qki_background_p95_raw,
      plane_integrity_status: row.plane_integrity_status ?? "",
      recorded_image_qc_status: row.recorded_image_qc_status ?? "",
      recorded_qc_flags: row.recorded_qc_flags ?? "",
      control_signal_qc_status: row.control_signal_qc_status ?? "",
      designation: row.designation,
      corrected_spot_rows: calls.spot_rows,
      eligible_spots: calls.eligible_spots ?? 0,
      sampled_spots: calls.sampled_spots ?? 0,
      null_usable: calls.null_usable,
      q90_positive: calls.q90_positive,
      q95_positive: calls.q95_positive,
      q99_positive: calls.q99_positive,
      combined_primary_omission: row.combined_primary_omission,
      biological_inference_eligible: row.biological_inference_eligible,
      status,
      notes,
    };
  });
}

function representativeOutputRows(rows, figuresRoot) {
  return rows.map((row) => {
    const nucleusFolder = `nucleus_${row.nucleus_id}_${row.selection_role}`;
    const walkthroughRoot = path.join(figuresRoot, "fields", row.image_key, nucleusFolder);
    return {
      ...row,
      output_file_png: path.join(walkthroughRoot, "threshold_walkthrough.png"),
      output_file_pdf: path.join(walkthroughRoot, "threshold_walkthrough.pdf"),
      output_file_svg: path.join(walkthroughRoot, "threshold_walkthrough.svg"),
      composite_png: path.join(figuresRoot, "clean_micrograph_grid.png"),
      composite_pdf: path.join(figuresRoot, "clean_micrograph_grid.pdf"),
      composite_svg: path.join(figuresRoot, "clean_micrograph_grid.svg"),
    };
  });
}

function buildProductionSummaryRows(endpointRows, ratioRows, config) {
  const result = [];
  for (const selected of config.endpointRows ?? []) {
    const matches = endpointRows.filter((row) =>
      row.cohort === "sampled_primary"
      && !row.excluded_biological_set
      && row.endpoint === selected.endpoint);
    if (matches.length !== 1) {
      throw new ProductionAdapterError(
        "SUMMARY_SOURCE_MISMATCH",
        `Expected one sampled_primary full-cohort inference row for ${selected.endpoint}; found ${matches.length}`,
      );
    }
    const row = matches[0];
    result.push({
      section: selected.section ?? "Primary endpoint",
      result: selected.displayLabel ?? selected.endpoint,
      numerator_endpoint: row.endpoint,
      denominator_endpoint: "",
      mean_nt: row.mean_nt,
      mean_kd: row.mean_kd,
      kd_over_nt: row.ratio_kd_over_nt,
      percent_change_kd_vs_nt: row.percent_change_kd_vs_nt,
      primary_exact_permutation_p: row.permutation_p_exact_two_sided,
      numerator_kd_over_nt: "",
      denominator_kd_over_nt: "",
      ratio_of_ratios: "",
      ratio_of_ratios_ci95_low: "",
      ratio_of_ratios_ci95_high: "",
      ratio_of_ratios_p_two_sided: "",
      note: selected.note ?? "",
    });
  }
  for (const selected of config.ratioRows ?? []) {
    const matches = ratioRows.filter((row) =>
      row.cohort === "sampled_primary"
      && row.numerator_endpoint === selected.numeratorEndpoint
      && row.denominator_endpoint === selected.denominatorEndpoint);
    if (matches.length !== 1) {
      throw new ProductionAdapterError(
        "SUMMARY_SOURCE_MISMATCH",
        `Expected one sampled_primary ratio-of-ratios row for ${selected.numeratorEndpoint}/${selected.denominatorEndpoint}; found ${matches.length}`,
      );
    }
    const row = matches[0];
    result.push({
      section: selected.section ?? "q95 associated/global contrast",
      result: selected.displayLabel ?? `${selected.numeratorEndpoint} / ${selected.denominatorEndpoint}`,
      numerator_endpoint: row.numerator_endpoint,
      denominator_endpoint: row.denominator_endpoint,
      mean_nt: "",
      mean_kd: "",
      kd_over_nt: "",
      percent_change_kd_vs_nt: "",
      primary_exact_permutation_p: "",
      numerator_kd_over_nt: row.numerator_ratio_kd_over_nt,
      denominator_kd_over_nt: row.denominator_ratio_kd_over_nt,
      ratio_of_ratios: row.ratio_of_ratios,
      ratio_of_ratios_ci95_low: row.ratio_of_ratios_ci95_low,
      ratio_of_ratios_ci95_high: row.ratio_of_ratios_ci95_high,
      ratio_of_ratios_p_two_sided: row.p_two_sided,
      note: selected.note ?? "Ratio of the KD/NT associated-signal ratio to the KD/NT global-signal ratio.",
    });
  }
  return result;
}

async function sha256File(filePath) {
  const buffer = await fs.readFile(filePath);
  return crypto.createHash("sha256").update(buffer).digest("hex");
}

async function assertFile(filePath, label) {
  try {
    const stat = await fs.stat(filePath);
    if (!stat.isFile()) throw new Error("not a file");
  } catch {
    throw new ProductionAdapterError("MISSING_REQUIRED_SOURCE", `${label}: ${filePath}`);
  }
}

async function assertAdapterOutputSafe(outputDir, expectedNames, overwrite) {
  try {
    const stat = await fs.stat(outputDir);
    if (!stat.isDirectory()) {
      throw new ProductionAdapterError("OUTPUT_TARGET_TYPE", `Adapter output is not a directory: ${outputDir}`);
    }
    const entries = await fs.readdir(outputDir, { withFileTypes: true });
    if (entries.length && !overwrite) {
      throw new ProductionAdapterError(
        "OUTPUT_EXISTS",
        `Refusing to overwrite existing adapter outputs without --overwrite true: ${outputDir}`,
      );
    }
    const allowed = new Set(expectedNames);
    const unexpected = entries.filter((entry) => !entry.isFile() || !allowed.has(entry.name));
    if (unexpected.length) {
      throw new ProductionAdapterError(
        "UNSAFE_OUTPUT_DIRECTORY",
        `Adapter output contains files this operation does not own: ${unexpected.map((entry) => entry.name).join(", ")}`,
      );
    }
  } catch (error) {
    if (error?.code === "ENOENT") return;
    throw error;
  }
}

function samplingAuditRows(sampling, manifest, pixelRows) {
  const rows = [];
  for (const [item, value] of Object.entries(sampling)) {
    rows.push({ item, value, source: "sampling_correction_audit.json", note: "Authoritative sampling-correction provenance." });
  }
  for (const item of [
    "analysis", "hierarchy", "pairing_semantics", "quantitation_plane",
    "sampling_metadata_source", "source_run",
  ]) {
    rows.push({ item, value: manifest[item] ?? "", source: "postrun_manifest.json", note: "Postrun analysis provenance." });
  }
  rows.push({
    item: "footprint_pixel_table_rows",
    value: pixelRows,
    source: "canonical gz CSV",
    note: "Retained losslessly outside Excel because 707,446 pixel rows would impair workbook usability.",
  });
  rows.push({
    item: "footprint_pixel_table_path",
    value: manifest.source_tables?.pixels?.path ?? "",
    source: "postrun_manifest.json",
    note: "Canonical lossless pixel-level table; intentionally omitted from the main workbook.",
  });
  rows.push({
    item: "footprint_pixel_table_sha256",
    value: manifest.source_tables?.pixels?.sha256 ?? "",
    source: "postrun_manifest.json",
    note: "SHA-256 for the canonical gz pixel table.",
  });
  return rows;
}

async function prepareProductionInputs(options) {
  const config = JSON.parse(await fs.readFile(options.configPath, "utf8"));
  const expectedOutputNames = [
    ...Object.values(config.outputs),
    config.adapterManifest,
  ];
  await assertAdapterOutputSafe(options.outputDir, expectedOutputNames, options.overwrite);
  const sourcePaths = Object.fromEntries(
    Object.entries(config.sources).map(([key, relative]) => [
      key,
      resolveSource(options.postrunRoot, relative),
    ]),
  );
  for (const [key, sourcePath] of Object.entries(sourcePaths)) {
    await assertFile(sourcePath, key);
  }
  await fs.mkdir(options.outputDir, { recursive: true });
  const staged = [];

  const spotCompressed = await fs.readFile(sourcePaths.spots);
  const spotText = (await gunzipAsync(spotCompressed)).toString("utf8");
  const controlSpots = aggregateControlSpots(spotText, config.expected.spots);
  const spotsOutput = path.join(options.outputDir, config.outputs.spots);
  await fs.writeFile(spotsOutput, spotText, "utf8");
  staged.push({ key: "spots", source: sourcePaths.spots, output: spotsOutput, ...controlSpots.stats });

  const exactCopies = [
    "nucleus", "endpointInference", "cartesian", "ratioOfRatios",
    "fovCorrelations", "setCorrelations", "correlationInference",
    "imageQc", "nucleusQc", "thresholdSensitivity",
  ];
  const parsed = {};
  for (const key of exactCopies) {
    const text = await fs.readFile(sourcePaths[key], "utf8");
    const stats = inspectCsvText(text, key, config.expected[key] ?? {});
    const output = path.join(options.outputDir, config.outputs[key]);
    await fs.copyFile(sourcePaths[key], output);
    staged.push({ key, source: sourcePaths[key], output, ...stats });
    if (["endpointInference", "ratioOfRatios"].includes(key)) {
      parsed[key] = objectsFromCsv(text, key, config.expected[key] ?? {}).rows;
    }
  }

  for (const key of ["fov", "set"]) {
    const text = await fs.readFile(sourcePaths[key], "utf8");
    const source = objectsFromCsv(text, key, config.expected[key] ?? {});
    const wide = pivotEndpointLongToWide(source.rows, config.pivots[key]);
    const output = path.join(options.outputDir, config.outputs[key]);
    await fs.writeFile(output, serializeCsv(wide.headers, wide.rows), "utf8");
    staged.push({
      key,
      source: sourcePaths[key],
      output,
      rows: wide.rows.length,
      columns: wide.headers.length,
      sourceRows: source.rows.length,
      sourceColumns: source.columns,
      endpointCount: wide.endpoints.length,
    });
  }

  const representativesSource = objectsFromCsv(
    await fs.readFile(sourcePaths.representatives, "utf8"),
    "representatives",
    config.expected.representatives,
  );
  const representatives = representativeOutputRows(
    representativesSource.rows,
    path.dirname(sourcePaths.representatives),
  );
  for (const row of representatives) {
    for (const key of [
      "output_file_png", "output_file_pdf", "output_file_svg",
      "composite_png", "composite_pdf", "composite_svg",
    ]) await assertFile(row[key], `representative ${key}`);
  }
  const representativeHeaders = [...representativesSource.headers, ...[
    "output_file_png", "output_file_pdf", "output_file_svg",
    "composite_png", "composite_pdf", "composite_svg",
  ]];
  const representativesOutput = path.join(options.outputDir, config.outputs.representatives);
  await fs.writeFile(
    representativesOutput,
    serializeCsv(representativeHeaders, representatives),
    "utf8",
  );
  staged.push({
    key: "representatives",
    source: sourcePaths.representatives,
    output: representativesOutput,
    rows: representatives.length,
    columns: representativeHeaders.length,
  });

  const controlSource = objectsFromCsv(
    await fs.readFile(sourcePaths.controls, "utf8"),
    "controls",
    config.expected.controls,
  );
  requireColumns(controlSource.headers, [
    "image_key", "miat_call_count", "control_signal_qc_status", "designation",
  ], "CONTROL_DIAGNOSTICS_V2 ledger");
  const controls = deriveControlRows(controlSource.rows, controlSpots.aggregate);
  const requiredControlStatuses = new Map([
    ["miat_647_qki_565__kd_sec_only_42.vsi", "PASS"],
    ["miat_647_qki_565__nt_sec_only_21.vsi", "REVIEW"],
  ]);
  for (const [imageKey, expectedStatus] of requiredControlStatuses) {
    const match = controls.find((row) => row.image_key === imageKey);
    if (!match || match.status !== expectedStatus) {
      throw new ProductionAdapterError(
        "CONTROL_STATUS_MISMATCH",
        `${imageKey} must be ${expectedStatus}; observed ${match?.status ?? "missing"}`,
      );
    }
  }
  const controlHeaders = Object.keys(controls[0]);
  const controlsOutput = path.join(options.outputDir, config.outputs.controls);
  await fs.writeFile(controlsOutput, serializeCsv(controlHeaders, controls), "utf8");
  staged.push({
    key: "controls",
    source: sourcePaths.controls,
    output: controlsOutput,
    rows: controls.length,
    columns: controlHeaders.length,
  });

  const summary = buildProductionSummaryRows(
    parsed.endpointInference,
    parsed.ratioOfRatios,
    config.summary,
  );
  const summaryHeaders = Object.keys(summary[0]);
  const summaryOutput = path.join(options.outputDir, config.outputs.summary);
  await fs.writeFile(summaryOutput, serializeCsv(summaryHeaders, summary), "utf8");
  staged.push({
    key: "summary",
    source: `${sourcePaths.endpointInference}; ${sourcePaths.ratioOfRatios}`,
    output: summaryOutput,
    rows: summary.length,
    columns: summaryHeaders.length,
  });

  const sampling = JSON.parse(await fs.readFile(sourcePaths.samplingAudit, "utf8"));
  const postrunManifest = JSON.parse(await fs.readFile(sourcePaths.postrunManifest, "utf8"));
  const samplingRows = samplingAuditRows(
    sampling,
    postrunManifest,
    config.expected.pixelRows,
  );
  const samplingHeaders = ["item", "value", "source", "note"];
  const samplingOutput = path.join(options.outputDir, config.outputs.samplingAudit);
  await fs.writeFile(samplingOutput, serializeCsv(samplingHeaders, samplingRows), "utf8");
  staged.push({
    key: "samplingAudit",
    source: `${sourcePaths.samplingAudit}; ${sourcePaths.postrunManifest}`,
    output: samplingOutput,
    rows: samplingRows.length,
    columns: samplingHeaders.length,
  });

  for (const entry of staged) {
    entry.sourceSha256 = entry.source.includes("; ")
      ? "multiple_sources_see_source_field"
      : await sha256File(entry.source);
    entry.outputSha256 = await sha256File(entry.output);
  }
  const manifest = {
    status: "success",
    adapterVersion: 1,
    postrunRoot: options.postrunRoot,
    configPath: options.configPath,
    outputDir: options.outputDir,
    v2SourcesOnly: true,
    valuesPolicy: "Source tables are copied exactly or pivoted without changing cell values; derived sheets are explicitly labeled.",
    omittedFromWorkbook: {
      table: postrunManifest.source_tables?.pixels?.path ?? "",
      rows: config.expected.pixelRows,
      sha256: postrunManifest.source_tables?.pixels?.sha256 ?? "",
      reason: "The 707,446-row pixel table remains losslessly retained as canonical gz CSV because including it would impair workbook usability.",
    },
    tables: staged,
  };
  const adapterManifestPath = path.join(options.outputDir, config.adapterManifest);
  await fs.writeFile(adapterManifestPath, `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  return { ...manifest, adapterManifestPath };
}

async function main() {
  try {
    const report = await prepareProductionInputs(parseArgs(process.argv.slice(2)));
    process.stdout.write(`${JSON.stringify({
      status: report.status,
      outputDir: report.outputDir,
      adapterManifestPath: report.adapterManifestPath,
      tables: report.tables.map(({ key, rows, columns }) => ({ key, rows, columns })),
    })}\n`);
  } catch (error) {
    process.stderr.write(`${JSON.stringify({
      status: "error",
      code: error.code ?? "UNEXPECTED_ERROR",
      message: error.message,
      details: error.details ?? {},
    })}\n`);
    process.exitCode = 1;
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}

export {
  ProductionAdapterError,
  buildProductionSummaryRows,
  deriveControlRows,
  pivotEndpointLongToWide,
  prepareProductionInputs,
  representativeOutputRows,
};
