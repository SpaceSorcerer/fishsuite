import fs from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { pathToFileURL } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const COLORS = {
  navy: "#1F4E78",
  navyDark: "#17365D",
  blue: "#0072B2",
  bluePale: "#DCE6F1",
  vermillion: "#D55E00",
  vermillionPale: "#FCE8E6",
  amberPale: "#FFF4CC",
  green: "#2E7D32",
  greenPale: "#E2F0D9",
  border: "#D9E2F3",
  white: "#FFFFFF",
};

class WorkbookBuildError extends Error {
  constructor(code, message, details = {}) {
    super(`${code}: ${message}`);
    this.name = "WorkbookBuildError";
    this.code = code;
    this.details = details;
  }
}

function parseArgs(argv) {
  const values = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith("--")) {
      throw new WorkbookBuildError("INVALID_ARGUMENT", `Unexpected positional argument: ${token}`);
    }
    const name = token.slice(2);
    const value = argv[index + 1];
    if (!value || value.startsWith("--")) {
      throw new WorkbookBuildError("INVALID_ARGUMENT", `Missing value for --${name}`);
    }
    values[name] = value;
    index += 1;
  }
  for (const required of ["input-dir", "config", "output"]) {
    if (!values[required]) {
      throw new WorkbookBuildError("INVALID_ARGUMENT", `--${required} is required`);
    }
  }
  return {
    inputDir: path.resolve(values["input-dir"]),
    configPath: path.resolve(values.config),
    outputPath: path.resolve(values.output),
    previewDir: values["preview-dir"] ? path.resolve(values["preview-dir"]) : null,
    verificationPath: values["verification-json"]
      ? path.resolve(values["verification-json"])
      : null,
    overwrite: values.overwrite === undefined
      ? false
      : parseBooleanArgument(values.overwrite, "overwrite"),
  };
}

function parseBooleanArgument(value, name) {
  const token = String(value).trim().toLowerCase();
  if (token === "true") return true;
  if (token === "false") return false;
  throw new WorkbookBuildError(
    "INVALID_ARGUMENT",
    `--${name} must be true or false`,
  );
}

function assertNonemptyArray(value, label) {
  if (!Array.isArray(value) || value.length === 0) {
    throw new WorkbookBuildError("INVALID_CONFIG", `${label} must be a non-empty array`);
  }
}

function validateConfig(config) {
  if (!config || typeof config !== "object") {
    throw new WorkbookBuildError("INVALID_CONFIG", "Configuration must be a JSON object");
  }
  assertNonemptyArray(config.tables, "tables");
  const keys = new Set();
  const sheets = new Set(["README", "Data Dictionary"]);
  const tableNames = new Set(["DataDictionaryTable", "WorkbookInventoryTable"]);
  const invalidSheet = /[\\/:*?\[\]]/;
  config.tables.forEach((spec, index) => {
    for (const field of ["key", "file", "sheet", "tableName"]) {
      if (typeof spec[field] !== "string" || spec[field].trim() === "") {
        throw new WorkbookBuildError(
          "INVALID_CONFIG",
          `tables[${index}].${field} must be a non-empty string`,
        );
      }
    }
    if (keys.has(spec.key)) {
      throw new WorkbookBuildError("INVALID_CONFIG", `Duplicate table key: ${spec.key}`);
    }
    if (sheets.has(spec.sheet) || spec.sheet.length > 31 || invalidSheet.test(spec.sheet)) {
      throw new WorkbookBuildError("INVALID_CONFIG", `Invalid or duplicate sheet name: ${spec.sheet}`);
    }
    if (tableNames.has(spec.tableName) || !/^[A-Za-z_][A-Za-z0-9_]*$/.test(spec.tableName)) {
      throw new WorkbookBuildError("INVALID_CONFIG", `Invalid or duplicate table name: ${spec.tableName}`);
    }
    if (spec.requiredColumns !== undefined && !Array.isArray(spec.requiredColumns)) {
      throw new WorkbookBuildError(
        "INVALID_CONFIG",
        `requiredColumns for ${spec.key} must be an array`,
      );
    }
    keys.add(spec.key);
    sheets.add(spec.sheet);
    tableNames.add(spec.tableName);
  });
  (config.superplots ?? []).forEach((plot, index) => {
    for (const field of ["title", "sourceTable", "conditionColumn", "valueColumn", "labelColumn"]) {
      if (typeof plot[field] !== "string" || plot[field].trim() === "") {
        throw new WorkbookBuildError(
          "INVALID_CONFIG",
          `superplots[${index}].${field} must be a non-empty string`,
        );
      }
    }
    if (!keys.has(plot.sourceTable)) {
      throw new WorkbookBuildError(
        "INVALID_CONFIG",
        `Superplot ${plot.title} references unknown table key ${plot.sourceTable}`,
      );
    }
    assertNonemptyArray(plot.groups, `superplots[${index}].groups`);
  });
}

function isInside(base, target) {
  const relative = path.relative(base, target);
  return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
}

async function pathState(target) {
  if (!target) return { exists: false, stat: null };
  try {
    return { exists: true, stat: await fs.stat(target) };
  } catch (error) {
    if (error?.code === "ENOENT") return { exists: false, stat: null };
    throw error;
  }
}

function previewFilename(sheetName) {
  const safeName = sheetName
    .toLowerCase()
    .replaceAll(/[^a-z0-9]+/g, "_")
    .replaceAll(/^_|_$/g, "");
  return `${safeName}.png`;
}

async function assertSafeOutputTargets(options, expectedPreviewFiles = []) {
  const targets = [options.outputPath, options.verificationPath].filter(Boolean);
  if (new Set(targets.map((target) => path.resolve(target).toLowerCase())).size !== targets.length) {
    throw new WorkbookBuildError(
      "OUTPUT_TARGET_COLLISION",
      "Workbook and verification outputs must be distinct paths",
    );
  }
  for (const target of targets) {
    const state = await pathState(target);
    if (!state.exists) continue;
    if (!state.stat.isFile()) {
      throw new WorkbookBuildError(
        "OUTPUT_TARGET_TYPE",
        `Output target exists and is not a file: ${target}`,
      );
    }
    if (!options.overwrite) {
      throw new WorkbookBuildError(
        "OUTPUT_EXISTS",
        `Refusing to overwrite existing output without --overwrite true: ${target}`,
      );
    }
  }
  if (!options.previewDir) return;
  const previewState = await pathState(options.previewDir);
  if (!previewState.exists) return;
  if (!previewState.stat.isDirectory()) {
    throw new WorkbookBuildError(
      "OUTPUT_TARGET_TYPE",
      `Preview target exists and is not a directory: ${options.previewDir}`,
    );
  }
  const existing = await fs.readdir(options.previewDir, { withFileTypes: true });
  if (existing.length && !options.overwrite) {
    throw new WorkbookBuildError(
      "OUTPUT_EXISTS",
      `Refusing to overwrite existing preview files without --overwrite true: ${options.previewDir}`,
    );
  }
  const allowed = new Set(expectedPreviewFiles);
  const unexpected = existing.filter(
    (entry) => !entry.isFile() || !allowed.has(entry.name),
  );
  if (unexpected.length) {
    throw new WorkbookBuildError(
      "UNSAFE_PREVIEW_DIRECTORY",
      `Preview directory contains files that this build does not own: ${unexpected.map((entry) => entry.name).join(", ")}`,
    );
  }
}

async function preflightFiles(inputDir, tableSpecs) {
  const available = [];
  const missingRequired = [];
  const missingOptional = [];
  for (const spec of tableSpecs) {
    const filePath = path.resolve(inputDir, spec.file);
    if (!isInside(inputDir, filePath)) {
      throw new WorkbookBuildError(
        "INVALID_CONFIG",
        `Table ${spec.key} resolves outside --input-dir: ${spec.file}`,
      );
    }
    try {
      const stat = await fs.stat(filePath);
      if (!stat.isFile()) throw new Error("not a file");
      available.push({ spec, filePath });
    } catch {
      if (spec.required === false) missingOptional.push(spec.file);
      else missingRequired.push(spec.file);
    }
  }
  if (missingRequired.length) {
    throw new WorkbookBuildError(
      "MISSING_REQUIRED_TABLES",
      `Required CSV files were not found: ${missingRequired.join(", ")}`,
      { missingRequired },
    );
  }
  return { available, missingOptional };
}

function columnLetter(index) {
  let number = index + 1;
  let result = "";
  while (number > 0) {
    const remainder = (number - 1) % 26;
    result = String.fromCharCode(65 + remainder) + result;
    number = Math.floor((number - 1) / 26);
  }
  return result;
}

function quoteSheet(name) {
  return `'${name.replaceAll("'", "''")}'`;
}

function normalizeHeader(value) {
  return String(value ?? "").replace(/^\uFEFF/, "").trim();
}

function parseBoolean(value) {
  if (typeof value === "boolean") return value;
  const token = String(value).trim().toLowerCase();
  if (["true", "yes", "y", "1"].includes(token)) return true;
  if (["false", "no", "n", "0"].includes(token)) return false;
  return value;
}

function isIdentifierHeader(header) {
  return /(^|_)(id|uid|key|name|label|file|path|condition|status|type|role|scope|endpoint|contrast|population|cohort|scale|method|notes?)(_|$)/.test(
    header.toLowerCase(),
  );
}

function isPValueHeader(header) {
  return /(^p($|_)|_p($|_)|p_value|pvalue)/.test(header.toLowerCase());
}

function inferredType(header, values, configuredType) {
  if (configuredType) return configuredType;
  const name = header.toLowerCase();
  if (isIdentifierHeader(header)) return "text";
  if (isPValueHeader(header)) return "pvalue";
  if (/(^|_)percentile($|_)/.test(name)) return "integer";
  if (/percent_change|percentage_change/.test(name)) return "percent_points";
  if (/per_nucleus|_mean$|_median$|enrichment|rho|distance/.test(name)) return "number";
  if (
    /(^is_|^has_|sampled|eligible|estimable|(^|_)complete$|(^|_)selected$|_valid$|_call$|_flag$|passed$)/.test(
      name,
    )
  ) {
    return "boolean";
  }
  if (/fraction|proportion|(^|_)pct($|_)/.test(name)) return "percentage";
  if (/count|(^|_)n_|_n$|spots|nuclei|fov|draws|selected_z|slide$/.test(name)) return "integer";
  const nonblank = values.filter(
    (value) => value !== null && value !== undefined && String(value).trim() !== "",
  );
  if (
    nonblank.length > 0 &&
    nonblank.every((value) =>
      /^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$/.test(String(value).trim()),
    )
  ) {
    return "number";
  }
  return "text";
}

function convertValue(value, type) {
  if (value === null || value === undefined || String(value).trim() === "") return null;
  if (type === "boolean") return parseBoolean(value);
  if (["integer", "number", "percentage", "percent_points", "pvalue"].includes(type)) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : value;
  }
  return String(value);
}

function defaultNumberFormat(header, type) {
  const name = header.toLowerCase();
  if (type === "integer") return "#,##0";
  if (type === "percentage") return "0.00%";
  if (type === "percent_points") return '0.00"%"';
  if (type === "pvalue") return "0.0000";
  if (/rho|correlation|effect|estimate|enrichment|distance|z_score|fold/.test(name)) return "0.000";
  if (/mass|intensity|sum|mean|median|threshold|area/.test(name)) return "#,##0.00";
  if (type === "number") return "#,##0.000";
  return "General";
}

function safeMatrix(values, filename) {
  if (!Array.isArray(values) || values.length < 2 || !Array.isArray(values[0])) {
    throw new WorkbookBuildError(
      "EMPTY_TABLE",
      `${filename} must contain a header and at least one data row`,
    );
  }
  const width = values[0].length;
  if (width === 0) throw new WorkbookBuildError("EMPTY_TABLE", `${filename} has an empty header`);
  return values.map((row) => {
    const normalized = Array.isArray(row) ? [...row] : [];
    while (normalized.length < width) normalized.push(null);
    return normalized.slice(0, width);
  });
}

function validateAndCoerceTable(spec, rawMatrix) {
  const rectangular = safeMatrix(rawMatrix, spec.file);
  const headers = rectangular[0].map(normalizeHeader);
  const invalidHeaders = headers.filter(
    (header, index) => !header || headers.indexOf(header) !== index,
  );
  if (invalidHeaders.length) {
    throw new WorkbookBuildError(
      "INVALID_HEADERS",
      `${spec.file} contains blank or duplicate headers: ${[...new Set(invalidHeaders)].join(", ")}`,
    );
  }
  const missing = (spec.requiredColumns ?? []).filter((column) => !headers.includes(column));
  if (missing.length) {
    throw new WorkbookBuildError(
      "MISSING_REQUIRED_COLUMNS",
      `${spec.file} is missing required columns: ${missing.join(", ")}`,
      { table: spec.key, missing },
    );
  }
  const rows = rectangular.slice(1);
  const columnInfo = headers.map((header, column) => {
    const metadata = spec.columns?.[header] ?? {};
    const type = inferredType(
      header,
      rows.map((row) => row[column]),
      metadata.type,
    );
    return {
      header,
      type,
      definition: metadata.definition ?? "Source column preserved from the configured CSV.",
      unit: metadata.unit ?? "",
      role: metadata.role ?? "source",
      numberFormat: metadata.numberFormat ?? defaultNumberFormat(header, type),
      required: (spec.requiredColumns ?? []).includes(header),
    };
  });
  const matrix = [
    headers,
    ...rows.map((row) =>
      row.map((value, column) => convertValue(value, columnInfo[column].type)),
    ),
  ];
  return { matrix, headers, rows: matrix.slice(1), columnInfo };
}

function estimateColumnWidth(header, values, type) {
  const strings = [
    header,
    ...values.slice(0, 80).map((value) => (value === null || value === undefined ? "" : String(value))),
  ];
  const maxLength = Math.max(...strings.map((value) => value.length));
  return Math.max(type === "text" ? 12 : 10, Math.min(42, maxLength + 2));
}

function styleDataSheet(sheet, record, freezePanesEnabled = true) {
  const { matrix, columnInfo, spec } = record;
  const rowCount = matrix.length;
  const columnCount = matrix[0].length;
  const lastColumn = columnLetter(columnCount - 1);
  const usedRange = sheet.getRange(`A1:${lastColumn}${rowCount}`);
  const writeChunkRows = Math.max(1, Number(spec.writeChunkRows ?? 1000));
  for (let rowOffset = 0; rowOffset < matrix.length; rowOffset += writeChunkRows) {
    const chunk = matrix.slice(rowOffset, rowOffset + writeChunkRows);
    const startRow = rowOffset + 1;
    const endRow = rowOffset + chunk.length;
    sheet.getRange(`A${startRow}:${lastColumn}${endRow}`).values = chunk;
  }
  if (!spec.compactStyle) {
    usedRange.format.font = { name: "Arial", size: 10, color: "#222222" };
    usedRange.format.verticalAlignment = "center";
  }
  const header = sheet.getRange(`A1:${lastColumn}1`);
  header.format = {
    fill: COLORS.navy,
    font: { name: "Arial", size: 10, bold: true, color: COLORS.white },
    wrapText: true,
    verticalAlignment: "center",
    borders: { preset: "outside", style: "thin", color: COLORS.navyDark },
  };
  header.format.rowHeight = 32;
  sheet.showGridLines = false;
  if (freezePanesEnabled) {
    sheet.freezePanes.freezeRows(1);
    if (spec.freezeColumns) sheet.freezePanes.freezeColumns(spec.freezeColumns);
  }
  for (let column = 0; column < columnCount; column += 1) {
    const info = columnInfo[column];
    const letter = columnLetter(column);
    const columnRange = sheet.getRange(`${letter}2:${letter}${rowCount}`);
    sheet.getRange(`${letter}:${letter}`).format.columnWidth = estimateColumnWidth(
      info.header,
      matrix.slice(1).map((row) => row[column]),
      info.type,
    );
    if (!spec.compactStyle) {
      columnRange.format.numberFormat = info.numberFormat;
      if (info.type === "text") columnRange.format.horizontalAlignment = "left";
      else if (info.type === "boolean") columnRange.format.horizontalAlignment = "center";
      else columnRange.format.horizontalAlignment = "right";
      const name = info.header.toLowerCase();
      if (/interpretation|notes|description|definition|exclusion_reason/.test(name)) {
        columnRange.format.wrapText = true;
        columnRange.format.rowHeight = 30;
      }
      if (isPValueHeader(name)) {
        columnRange.conditionalFormats.add("cellIs", {
          operator: "lessThan",
          formula: 0.05,
          format: {
            fill: COLORS.vermillionPale,
            font: { bold: true, color: COLORS.vermillion },
          },
        });
      }
      if (/status|qc_result|validation_result/.test(name)) {
        for (const token of ["PASS", "OK", "COMPLETE"]) {
          columnRange.conditionalFormats.add("containsText", {
            text: token,
            format: { fill: COLORS.greenPale, font: { bold: true, color: COLORS.green } },
          });
        }
        for (const token of ["WARN", "REVIEW"]) {
          columnRange.conditionalFormats.add("containsText", {
            text: token,
            format: { fill: COLORS.amberPale, font: { bold: true, color: "#8A5A00" } },
          });
        }
        for (const token of ["FAIL", "ERROR"]) {
          columnRange.conditionalFormats.add("containsText", {
            text: token,
            format: {
              fill: COLORS.vermillionPale,
              font: { bold: true, color: COLORS.vermillion },
            },
          });
        }
      }
    }
  }
  const table = sheet.tables.add(`A1:${lastColumn}${rowCount}`, true, spec.tableName);
  table.style = spec.tableStyle ?? "TableStyleMedium2";
  table.showHeaders = true;
  table.showFilterButton = true;
  table.showBandedColumns = false;
}

function writeReadmeSheet(sheet, config, records, missingOptional, freezePanesEnabled = true) {
  const title = config.workbook?.title ?? "MIAT×QKI postrun analysis workbook";
  sheet.showGridLines = false;
  sheet.mergeCells("A1:H1");
  sheet.getRange("A1").values = [[title]];
  sheet.getRange("A1:H1").format = {
    fill: COLORS.navyDark,
    font: { name: "Arial", size: 18, bold: true, color: COLORS.white },
    verticalAlignment: "center",
  };
  sheet.getRange("A1:H1").format.rowHeight = 34;
  sheet.mergeCells("A2:H2");
  sheet.getRange("A2").values = [[
    config.workbook?.subtitle ?? "Review-ready processed tables and audit metadata",
  ]];
  sheet.getRange("A2:H2").format = {
    fill: COLORS.bluePale,
    font: { name: "Arial", size: 11, italic: true, color: COLORS.navyDark },
  };
  const notes = [
    ["Purpose", config.workbook?.purpose ?? "Consolidate postrun CSV outputs without altering source measurements."],
    ["Primary interpretation guard", config.workbook?.interpretationGuard ?? "Spatial association is not evidence of direct molecular binding."],
    ["Primary analysis convention", config.workbook?.analysisConvention ?? "Exact footprints on one reviewed z plane shared by all channels."],
    ["Workbook provenance", config.workbook?.provenance ?? "Values are imported from processed CSV files; chart helpers are formula-linked."],
  ];
  sheet.getRange("A4:B7").values = notes;
  sheet.getRange("A4:A7").format = {
    fill: COLORS.navy,
    font: { name: "Arial", size: 10, bold: true, color: COLORS.white },
  };
  sheet.getRange("B4:B7").format = {
    font: { name: "Arial", size: 10, color: "#222222" },
    wrapText: true,
  };
  sheet.getRange("A4:B7").format.borders = {
    preset: "outside",
    style: "thin",
    color: COLORS.border,
  };
  sheet.getRange("A9:F9").values = [[
    "Table key", "Worksheet", "Source CSV", "Rows", "Columns", "Description",
  ]];
  const inventory = records.map((record) => [
    record.spec.key,
    record.spec.sheet,
    record.spec.file,
    record.matrix.length - 1,
    record.headers.length,
    record.spec.description ?? "Processed analysis table.",
  ]);
  const inventoryEnd = 9 + inventory.length;
  sheet.getRange(`A10:F${inventoryEnd}`).values = inventory;
  sheet.getRange(`A9:F${inventoryEnd}`).format.font = {
    name: "Arial", size: 10, color: "#222222",
  };
  sheet.getRange("A9:F9").format = {
    fill: COLORS.navy,
    font: { name: "Arial", size: 10, bold: true, color: COLORS.white },
    wrapText: true,
  };
  sheet.getRange(`D10:E${inventoryEnd}`).format.numberFormat = "#,##0";
  sheet.getRange(`F10:F${inventoryEnd}`).format.wrapText = true;
  sheet.getRange(`A10:F${inventoryEnd}`).format.rowHeight = 28;
  const table = sheet.tables.add(`A9:F${inventoryEnd}`, true, "WorkbookInventoryTable");
  table.style = "TableStyleMedium2";
  table.showFilterButton = true;
  let nextRow = inventoryEnd + 2;
  if (missingOptional.length) {
    sheet.getRange(`A${nextRow}:B${nextRow}`).values = [[
      "Optional CSV files not supplied", missingOptional.join(", "),
    ]];
    sheet.getRange(`A${nextRow}`).format = {
      fill: COLORS.amberPale,
      font: { name: "Arial", bold: true, color: "#8A5A00" },
    };
    nextRow += 2;
  }
  const configNotes = config.workbook?.readmeNotes ?? [];
  if (configNotes.length) {
    sheet.getRange(`A${nextRow}:B${nextRow}`).values = [["Additional analysis notes", ""]];
    sheet.getRange(`A${nextRow}:B${nextRow}`).format = {
      fill: COLORS.bluePale,
      font: { name: "Arial", bold: true, color: COLORS.navyDark },
    };
    const noteRows = configNotes.map((note, index) => [index + 1, note]);
    sheet.getRange(`A${nextRow + 1}:B${nextRow + noteRows.length}`).values = noteRows;
    sheet.getRange(`B${nextRow + 1}:B${nextRow + noteRows.length}`).format.wrapText = true;
    nextRow += noteRows.length + 1;
  }
  sheet.getRange(`A1:H${Math.max(nextRow, inventoryEnd)}`).format.font.name = "Arial";
  sheet.getRange("A:A").format.columnWidth = 29;
  sheet.getRange("B:B").format.columnWidth = 64;
  sheet.getRange("C:C").format.columnWidth = 26;
  sheet.getRange("D:E").format.columnWidth = 12;
  sheet.getRange("F:F").format.columnWidth = 46;
  if (freezePanesEnabled) sheet.freezePanes.freezeRows(2);
  return Math.max(nextRow, inventoryEnd);
}

function writeDataDictionary(sheet, records, freezePanesEnabled = true) {
  const headers = [
    "Table key", "Worksheet", "Source CSV", "Column", "Required",
    "Type", "Number format", "Unit", "Role", "Definition",
  ];
  const rows = records.flatMap((record) =>
    record.columnInfo.map((column) => [
      record.spec.key,
      record.spec.sheet,
      record.spec.file,
      column.header,
      column.required,
      column.type,
      column.numberFormat,
      column.unit,
      column.role,
      column.definition,
    ]),
  );
  const matrix = [headers, ...rows];
  sheet.getRange(`A1:J${matrix.length}`).values = matrix;
  sheet.getRange(`A1:J${matrix.length}`).format.font = {
    name: "Arial", size: 10, color: "#222222",
  };
  sheet.getRange("A1:J1").format = {
    fill: COLORS.navy,
    font: { name: "Arial", size: 10, bold: true, color: COLORS.white },
    wrapText: true,
  };
  sheet.getRange("A1:J1").format.rowHeight = 32;
  sheet.getRange(`E2:E${matrix.length}`).format.horizontalAlignment = "center";
  sheet.getRange(`J2:J${matrix.length}`).format.wrapText = true;
  sheet.showGridLines = false;
  if (freezePanesEnabled) {
    sheet.freezePanes.freezeRows(1);
    sheet.freezePanes.freezeColumns(3);
  }
  [14, 18, 26, 35, 10, 14, 16, 14, 14, 58].forEach((width, index) => {
    const letter = columnLetter(index);
    sheet.getRange(`${letter}:${letter}`).format.columnWidth = width;
  });
  const table = sheet.tables.add(`A1:J${matrix.length}`, true, "DataDictionaryTable");
  table.style = "TableStyleMedium2";
  table.showFilterButton = true;
  return matrix.length;
}

function requireHeader(record, header, context) {
  const index = record.headers.indexOf(header);
  if (index < 0) {
    throw new WorkbookBuildError(
      "MISSING_SUPERPLOT_COLUMN",
      `${context} requires column ${header} in ${record.spec.file}`,
    );
  }
  return index;
}

function buildSuperplots(summarySheet, recordsByKey, plots, summaryTableRows) {
  let helperRow = Math.max(summaryTableRows + 4, 12);
  let chartTop = 2;
  let maxRow = summaryTableRows;
  for (const plot of plots) {
    const record = recordsByKey.get(plot.sourceTable);
    if (!record) {
      if (plot.required === false) continue;
      throw new WorkbookBuildError(
        "MISSING_SUPERPLOT_SOURCE",
        `Superplot ${plot.title} requires table ${plot.sourceTable}`,
      );
    }
    const conditionIndex = requireHeader(record, plot.conditionColumn, plot.title);
    const valueIndex = requireHeader(record, plot.valueColumn, plot.title);
    const labelIndex = requireHeader(record, plot.labelColumn, plot.title);
    const filterColumns = Object.entries(plot.filters ?? {}).map(([column, expected]) => ({
      index: requireHeader(record, column, plot.title),
      expected,
    }));
    const valueInfo = record.columnInfo[valueIndex];
    summarySheet.getRange(`A${helperRow}:D${helperRow}`).merge();
    summarySheet.getRange(`A${helperRow}`).values = [[plot.title]];
    summarySheet.getRange(`A${helperRow}:D${helperRow}`).format = {
      fill: COLORS.bluePale,
      font: { name: "Arial", size: 11, bold: true, color: COLORS.navyDark },
    };
    helperRow += 1;
    summarySheet.getRange(`A${helperRow}:D${helperRow}`).values = [[
      "Condition", "Set", "X position", plot.valueColumn,
    ]];
    summarySheet.getRange(`A${helperRow}:D${helperRow}`).format = {
      fill: COLORS.navy,
      font: { name: "Arial", size: 10, bold: true, color: COLORS.white },
    };
    const helperHeaderRow = helperRow;
    helperRow += 1;
    const seriesRanges = [];
    for (const group of plot.groups) {
      const matching = [];
      record.rows.forEach((row, rowIndex) => {
        if (
          String(row[conditionIndex]) === String(group.value)
          && filterColumns.every(({ index, expected }) => String(row[index]) === String(expected))
        ) matching.push(rowIndex);
      });
      if (!matching.length) {
        if (plot.required === false) continue;
        throw new WorkbookBuildError(
          "EMPTY_SUPERPLOT_GROUP",
          `Superplot ${plot.title} has no rows for group ${group.value}`,
        );
      }
      const start = helperRow;
      const jitterStep = matching.length > 1 ? 0.24 / (matching.length - 1) : 0;
      summarySheet.getRange(`A${start}:D${start + matching.length - 1}`).values =
        matching.map((_, index) => [
          group.value,
          null,
          Number(group.x) - 0.12 + jitterStep * index,
          null,
        ]);
      summarySheet.getRange(`B${start}:B${start + matching.length - 1}`).formulas =
        matching.map((sourceRow) => [
          `=${quoteSheet(record.spec.sheet)}!$${columnLetter(labelIndex)}$${sourceRow + 2}`,
        ]);
      summarySheet.getRange(`D${start}:D${start + matching.length - 1}`).formulas =
        matching.map((sourceRow) => [
          `=${quoteSheet(record.spec.sheet)}!$${columnLetter(valueIndex)}$${sourceRow + 2}`,
        ]);
      summarySheet.getRange(`C${start}:C${start + matching.length - 1}`).format.numberFormat =
        "0.00";
      summarySheet.getRange(`D${start}:D${start + matching.length - 1}`).format.numberFormat =
        valueInfo.numberFormat;
      seriesRanges.push({ group, start, end: start + matching.length - 1 });
      helperRow += matching.length;
    }
    summarySheet.getRange(`A${helperHeaderRow}:D${helperRow - 1}`).format.font = {
      name: "Arial", size: 10, color: "#222222",
    };
    summarySheet.getRange(`A${helperHeaderRow}:D${helperHeaderRow}`).format.font = {
      name: "Arial", size: 10, bold: true, color: COLORS.white,
    };
    const chart = summarySheet.charts.add("scatter", {
      chartType: "scatter", title: plot.title, hasLegend: true,
    });
    for (const entry of seriesRanges) {
      const series = chart.series.add(String(entry.group.label ?? entry.group.value));
      series.categoryFormula =
        `${quoteSheet(summarySheet.name)}!$C$${entry.start}:$C$${entry.end}`;
      series.formula =
        `${quoteSheet(summarySheet.name)}!$D$${entry.start}:$D$${entry.end}`;
      series.fill =
        entry.group.color ?? (entry.group.value === "KD" ? COLORS.vermillion : COLORS.blue);
    }
    chart.title = plot.title;
    chart.titleTextStyle.fontSize = 12;
    chart.hasLegend = true;
    chart.xAxis = {
      axisType: "valueAxis",
      min: Math.min(...plot.groups.map((group) => Number(group.x))) - 0.5,
      max: Math.max(...plot.groups.map((group) => Number(group.x))) + 0.5,
      majorUnit: 1,
      numberFormatCode: "0",
    };
    chart.yAxis = { numberFormatCode: valueInfo.numberFormat };
    chart.xAxis.title.text = plot.xAxisTitle ?? "Condition (NT=1, KD=2)";
    chart.yAxis.title.text = plot.yAxisTitle ?? plot.valueColumn;
    chart.setPosition(
      `${plot.chartStartColumn ?? "G"}${chartTop}`,
      `${plot.chartEndColumn ?? "O"}${chartTop + 16}`,
    );
    chartTop += 19;
    helperRow += 2;
    maxRow = Math.max(maxRow, helperRow, chartTop + 16);
  }
  summarySheet.getRange("A:A").format.columnWidth = 38;
  summarySheet.getRange("B:B").format.columnWidth = 18;
  summarySheet.getRange("C:C").format.columnWidth = 12;
  summarySheet.getRange("D:D").format.columnWidth = 30;
  return maxRow;
}

function previewRangeFor(record) {
  const rows = Math.min(Math.max(record.matrix.length, 8), 36);
  const columns = Math.min(Math.max(record.headers.length, 6), 18);
  return `A1:${columnLetter(columns - 1)}${rows}`;
}

function assertNoFormulaErrors(ndjson) {
  const raw = String(ndjson ?? "").trim();
  const records = raw
    ? raw.split(/\r?\n/).filter(Boolean).map((line) => {
        try {
          return JSON.parse(line);
        } catch {
          throw new WorkbookBuildError(
            "FORMULA_ERROR_SCAN_UNCERTAIN",
            "Formula-error inspection returned invalid NDJSON",
            { line },
          );
        }
      })
    : [];
  const errorPattern = /#REF!|#DIV\/0!|#VALUE!|#NAME\?|#N\/A/;
  const matches = records.filter((record) => errorPattern.test(JSON.stringify(record)));
  if (matches.length) {
    throw new WorkbookBuildError(
      "FORMULA_ERRORS_FOUND",
      `Formula-error inspection found ${matches.length} matching record(s)`,
      { matches },
    );
  }
  const confirmedZero = records.some(
    (record) => record.kind === "notice" && /matched 0 entries/i.test(String(record.message ?? "")),
  );
  if (!confirmedZero) {
    throw new WorkbookBuildError(
      "FORMULA_ERROR_SCAN_UNCERTAIN",
      "Formula-error inspection did not explicitly confirm zero matches",
      { records },
    );
  }
}

async function exportXlsxWithBoundedInspection(
  workbook,
  boundedInspectionNdjson,
  exporter = (target) => SpreadsheetFile.exportXlsx(target),
) {
  const originalInspect = workbook.inspect;
  workbook.inspect = async function boundedExportInspect(options) {
    const kind = String(options?.kind ?? "");
    if (
      options?.maxChars === 0
      && kind.includes("workbook")
      && kind.includes("computedStyle")
      && kind.includes("drawing")
    ) {
      return { ndjson: String(boundedInspectionNdjson ?? "") };
    }
    return originalInspect.call(this, options);
  };
  try {
    return await exporter(workbook);
  } finally {
    workbook.inspect = originalInspect;
  }
}

async function verifyAndRender(
  workbook,
  records,
  previewDir,
  readmeRows,
  dictionaryRows,
  summaryMaxRow,
  summaryPreviewLastColumn = "O",
) {
  const sheetInspect = await workbook.inspect({
    kind: "sheet",
    include: "id,name",
    maxChars: 8000,
  });
  const summaryInspect = await workbook.inspect({
    kind: "table,formula,drawing",
    sheetId: "Summary",
    range: `A1:${summaryPreviewLastColumn}${Math.min(Math.max(summaryMaxRow, 20), 90)}`,
    maxChars: 8000,
    tableMaxRows: 20,
    tableMaxCols: 15,
    options: { maxResults: 150 },
  });
  const formulaErrors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 300 },
    summary: "final formula error scan",
    maxChars: 8000,
  });
  assertNoFormulaErrors(formulaErrors.ndjson);
  const previewFiles = [];
  if (previewDir) {
    await fs.mkdir(previewDir, { recursive: true });
    const ranges = new Map([
      ["README", `A1:H${Math.min(Math.max(readmeRows, 12), 40)}`],
      ["Data Dictionary", `A1:J${Math.min(Math.max(dictionaryRows, 12), 36)}`],
    ]);
    for (const record of records) ranges.set(record.spec.sheet, previewRangeFor(record));
    ranges.set(
      "Summary",
      `A1:${summaryPreviewLastColumn}${Math.min(Math.max(summaryMaxRow, 22), 90)}`,
    );
    for (const [sheetName, range] of ranges) {
      const blob = await workbook.render({ sheetName, range, scale: 1, format: "png" });
      const previewPath = path.join(previewDir, previewFilename(sheetName));
      await fs.writeFile(previewPath, new Uint8Array(await blob.arrayBuffer()));
      previewFiles.push(previewPath);
    }
  }
  return {
    sheetInspect: sheetInspect.ndjson,
    summaryInspect: summaryInspect.ndjson,
    formulaErrorScan: formulaErrors.ndjson,
    previewFiles,
  };
}

async function buildWorkbook(options) {
  const config = JSON.parse(await fs.readFile(options.configPath, "utf8"));
  validateConfig(config);
  const expectedPreviewFiles = [
    "README",
    "Data Dictionary",
    ...config.tables.map((spec) => spec.sheet),
  ].map(previewFilename);
  await assertSafeOutputTargets(options, expectedPreviewFiles);
  const { available, missingOptional } = await preflightFiles(options.inputDir, config.tables);
  const workbook = Workbook.create();
  const readmeSheet = workbook.worksheets.add("README");
  const dictionarySheet = workbook.worksheets.add("Data Dictionary");
  const records = [];
  for (const { spec, filePath } of available) {
    const csvText = (await fs.readFile(filePath, "utf8")).replace(/^\uFEFF/, "");
    const parsedWorkbook = await Workbook.fromCSV(csvText, { sheetName: spec.sheet });
    const parsedSheet = parsedWorkbook.worksheets.getItem(spec.sheet);
    const used = parsedSheet.getUsedRange(true) ?? parsedSheet.getUsedRange();
    const record = { spec, filePath, ...validateAndCoerceTable(spec, used.values) };
    const sheet = workbook.worksheets.add(spec.sheet);
    styleDataSheet(sheet, record, config.workbook?.freezePanes !== false);
    records.push(record);
  }
  const recordsByKey = new Map(records.map((record) => [record.spec.key, record]));
  const summaryKey = config.summaryTableKey ?? "summary";
  const summaryRecord = recordsByKey.get(summaryKey);
  if (!summaryRecord) {
    throw new WorkbookBuildError(
      "MISSING_SUMMARY_TABLE",
      `No imported table matches summaryTableKey ${summaryKey}`,
    );
  }
  const readmeRows = writeReadmeSheet(
    readmeSheet,
    config,
    records,
    missingOptional,
    config.workbook?.freezePanes !== false,
  );
  const dictionaryRows = writeDataDictionary(
    dictionarySheet,
    records,
    config.workbook?.freezePanes !== false,
  );
  const summarySheet = workbook.worksheets.getItem(summaryRecord.spec.sheet);
  const summaryMaxRow = buildSuperplots(
    summarySheet,
    recordsByKey,
    config.superplots ?? [],
    summaryRecord.matrix.length,
  );
  const verification = await verifyAndRender(
    workbook,
    records,
    options.previewDir,
    readmeRows,
    dictionaryRows,
    summaryMaxRow,
    config.workbook?.summaryPreviewLastColumn ?? "O",
  );
  await fs.mkdir(path.dirname(options.outputPath), { recursive: true });
  const boundedExportInspection = [
    verification.sheetInspect,
    verification.summaryInspect,
    verification.formulaErrorScan,
  ].filter(Boolean).join("\n");
  const output = await exportXlsxWithBoundedInspection(
    workbook,
    boundedExportInspection,
  );
  await output.save(options.outputPath);
  const report = {
    status: "success",
    outputPath: options.outputPath,
    inputDir: options.inputDir,
    configPath: options.configPath,
    tables: records.map((record) => ({
      key: record.spec.key,
      sheet: record.spec.sheet,
      source: record.spec.file,
      rows: record.matrix.length - 1,
      columns: record.headers.length,
    })),
    missingOptional,
    sheets: ["README", "Data Dictionary", ...records.map((record) => record.spec.sheet)],
    superplotCount: config.superplots?.length ?? 0,
    ...verification,
  };
  if (options.verificationPath) {
    await fs.mkdir(path.dirname(options.verificationPath), { recursive: true });
    await fs.writeFile(
      options.verificationPath,
      `${JSON.stringify(report, null, 2)}\n`,
      "utf8",
    );
  }
  return report;
}

async function main() {
  try {
    const report = await buildWorkbook(parseArgs(process.argv.slice(2)));
    process.stdout.write(
      `${JSON.stringify({
        status: report.status,
        outputPath: report.outputPath,
        sheetCount: report.sheets.length,
        sheets: report.sheets,
        superplotCount: report.superplotCount,
        previewCount: report.previewFiles.length,
      })}\n`,
    );
  } catch (error) {
    const payload = {
      status: "error",
      code: error.code ?? "UNEXPECTED_ERROR",
      message: error.message,
      details: error.details ?? {},
    };
    process.stderr.write(`${JSON.stringify(payload)}\n`);
    process.exitCode = 1;
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}

export {
  WorkbookBuildError,
  assertNoFormulaErrors,
  assertSafeOutputTargets,
  buildWorkbook,
  exportXlsxWithBoundedInspection,
  validateConfig,
};
