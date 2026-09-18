import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import {
  buildProductionSummaryRows,
  deriveControlRows,
  pivotEndpointLongToWide,
  representativeOutputRows,
} from "../prepare_production_inputs.mjs";
import {
  assertNoFormulaErrors,
  assertSafeOutputTargets,
  exportXlsxWithBoundedInspection,
} from "../build_workbook.mjs";

const productionAdapter = JSON.parse(
  await fs.readFile(new URL("../production_adapter.json", import.meta.url), "utf8"),
);
assert.deepEqual(
  productionAdapter.summary.endpointRows.map(({ endpoint, displayLabel }) => ({
    endpoint,
    displayLabel,
  })),
  [
    {
      endpoint: "n_spots_all",
      displayLabel: "All detected MIAT spots per nucleus",
    },
    {
      endpoint: "threshold_positive_spots_per_nucleus_q95",
      displayLabel: "q95 QKI-associated MIAT spots per nucleus",
    },
    {
      endpoint: "miat_footprint_mass_all_union_deduplicated",
      displayLabel: "All-detected MIAT spot-pixel intensity per nucleus",
    },
    {
      endpoint: "miat_footprint_mass_q95_positive_union_deduplicated",
      displayLabel: "q95-associated MIAT spot-pixel intensity per nucleus",
    },
    {
      endpoint: "association_fraction_among_usable_q95",
      displayLabel: "q95 association fraction among null-usable MIAT spots",
    },
    {
      endpoint: "association_fraction_among_all_floor_spots_q95",
      displayLabel: "q95 association fraction among all floor-passing MIAT spots",
    },
    {
      endpoint: "threshold_usability_coverage_q95",
      displayLabel: "q95 null-usability coverage",
    },
  ],
  "Reader-facing summary rows must use the all-detected global endpoints",
);
assert.deepEqual(
  productionAdapter.summary.ratioRows.map((row) => ({
    numeratorEndpoint: row.numeratorEndpoint,
    denominatorEndpoint: row.denominatorEndpoint,
    displayLabel: row.displayLabel,
  })),
  [
    {
      numeratorEndpoint: "threshold_positive_spots_per_nucleus_q95",
      denominatorEndpoint: "n_spots_floor",
      displayLabel: "q95-associated spot-count change relative to floor-passing spot-count change",
    },
    {
      numeratorEndpoint: "miat_footprint_mass_q95_positive_union_deduplicated",
      denominatorEndpoint: "miat_footprint_mass_floor_union_deduplicated",
      displayLabel: "q95-associated MIAT spot-pixel intensity change relative to floor-passing MIAT spot-pixel intensity change",
    },
  ],
  "Each retained RoR denominator must match its canonical floor-passing count or union-deduplicated intensity endpoint",
);
const readerFacingAdapterText = [
  ...productionAdapter.summary.endpointRows,
  ...productionAdapter.summary.ratioRows,
]
  .flatMap((row) => [row.displayLabel, row.section, row.note])
  .filter(Boolean)
  .join("\n");
assert.doesNotMatch(readerFacingAdapterText, /exact-footprint mass|miat-mass/i);
assert.match(
  productionAdapter.summary.endpointRows[2].note,
  /union-deduplicated.*all detected miat spots/i,
);
assert.match(
  productionAdapter.summary.ratioRows[1].note,
  /q95-associated miat spot-pixel intensity ratio.*floor-passing miat spot-pixel intensity ratio/i,
);

const fovRows = [
  {
    cohort: "sampled_primary",
    endpoint: "n_spots_floor",
    image: "NT_1.vsi",
    image_key: "nt_1.vsi",
    slide: "1",
    arm: "NT",
    replicate: "1",
    fov: "1",
    biological_set: "S1_NT_1",
    is_control: "False",
    n_nuclei_total: "10",
    n_nuclei_finite: "10",
    value: "21.5",
  },
  {
    cohort: "sampled_primary",
    endpoint: "association_fraction_among_usable_q95",
    image: "NT_1.vsi",
    image_key: "nt_1.vsi",
    slide: "1",
    arm: "NT",
    replicate: "1",
    fov: "1",
    biological_set: "S1_NT_1",
    is_control: "False",
    n_nuclei_total: "10",
    n_nuclei_finite: "9",
    value: "0.125",
  },
  {
    cohort: "sampled_primary",
    endpoint: "n_spots_floor",
    image: "KD_1.vsi",
    image_key: "kd_1.vsi",
    slide: "1",
    arm: "KD",
    replicate: "1",
    fov: "1",
    biological_set: "S1_KD_1",
    is_control: "False",
    n_nuclei_total: "10",
    n_nuclei_finite: "10",
    value: "12.75",
  },
  {
    cohort: "sampled_primary",
    endpoint: "association_fraction_among_usable_q95",
    image: "KD_1.vsi",
    image_key: "kd_1.vsi",
    slide: "1",
    arm: "KD",
    replicate: "1",
    fov: "1",
    biological_set: "S1_KD_1",
    is_control: "False",
    n_nuclei_total: "10",
    n_nuclei_finite: "8",
    value: "0.075",
  },
];

const pivoted = pivotEndpointLongToWide(fovRows, {
  keyColumns: [
    "cohort", "image", "image_key", "slide", "arm", "replicate", "fov",
    "biological_set", "is_control", "n_nuclei_total",
  ],
  finiteCountColumn: "n_nuclei_finite",
});
assert.deepEqual(pivoted.headers.slice(-4), [
  "n_nuclei_finite__association_fraction_among_usable_q95",
  "n_nuclei_finite__n_spots_floor",
  "association_fraction_among_usable_q95",
  "n_spots_floor",
]);
assert.equal(pivoted.rows.length, 2);
assert.equal(pivoted.rows[0].at(-1), "21.5");
assert.equal(pivoted.rows[0].at(-2), "0.125");

assert.throws(
  () => pivotEndpointLongToWide([...fovRows, { ...fovRows[0] }], {
    keyColumns: ["cohort", "image_key"],
    finiteCountColumn: "n_nuclei_finite",
  }),
  /DUPLICATE_PIVOT_VALUE/,
);

const controls = deriveControlRows(
  [
    {
      image_key: "miat_647_qki_565__kd_sec_only_42.vsi",
      image: "MIAT_647_QKI_565__KD_Sec_Only_42.vsi",
      miat_call_count: "0",
      mask_qc_status: "failed",
      designation: "clean expected-negative zero-call example",
      biological_inference_eligible: "False",
    },
    {
      image_key: "miat_647_qki_565__nt_sec_only_21.vsi",
      image: "MIAT_647_QKI_565__NT_Sec_Only_21.vsi",
      miat_call_count: "304",
      mask_qc_status: "pass",
      designation: "artifact-bearing retained diagnostic",
      biological_inference_eligible: "False",
    },
  ],
  new Map([
    ["miat_647_qki_565__kd_sec_only_42.vsi", {
      spot_rows: 0, null_usable: 0, q90_positive: 0, q95_positive: 0, q99_positive: 0,
    }],
    ["miat_647_qki_565__nt_sec_only_21.vsi", {
      spot_rows: 304, null_usable: 159, q90_positive: 30, q95_positive: 13, q99_positive: 1,
    }],
  ]),
);
assert.equal(controls[0].status, "PASS");
assert.match(controls[0].notes, /expected-negative structural zero/i);
assert.equal(controls[1].status, "REVIEW");
assert.match(controls[1].notes, /artifact/i);
assert.ok(!Object.hasOwn(controls[0], "mask_qc_status"));

const representativeRows = representativeOutputRows(
  [{
    image_key: "miat_647_qki_565__nt_2_14.vsi",
    nucleus_id: "17",
    selection_role: "mixed",
  }],
  "F:\\POSTRUN\\FIGURES_V2",
);
assert.equal(
  representativeRows[0].output_file_png,
  "F:\\POSTRUN\\FIGURES_V2\\fields\\miat_647_qki_565__nt_2_14.vsi\\nucleus_17_mixed\\threshold_walkthrough.png",
);
assert.match(representativeRows[0].output_file_pdf, /threshold_walkthrough\.pdf$/);
assert.match(representativeRows[0].output_file_svg, /threshold_walkthrough\.svg$/);

const endpointInference = [
  {
    cohort: "sampled_primary",
    endpoint: "association_fraction_among_usable_q95",
    excluded_biological_set: "",
    mean_nt: "0.1125",
    mean_kd: "0.0868",
    ratio_kd_over_nt: "0.7714",
    percent_change_kd_vs_nt: "-22.86",
    permutation_p_exact_two_sided: "0.03",
  },
  {
    cohort: "sampled_primary",
    endpoint: "association_fraction_among_all_floor_spots_q95",
    excluded_biological_set: "",
    mean_nt: "0.0666",
    mean_kd: "0.0628",
    ratio_kd_over_nt: "0.9417",
    percent_change_kd_vs_nt: "-5.83",
    permutation_p_exact_two_sided: "0.575",
  },
];
const ratioRows = [
  {
    cohort: "sampled_primary",
    numerator_endpoint: "threshold_positive_spots_per_nucleus_q95",
    denominator_endpoint: "n_spots_floor",
    numerator_ratio_kd_over_nt: "0.6646",
    denominator_ratio_kd_over_nt: "0.6155",
    ratio_of_ratios: "1.0799",
    ratio_of_ratios_ci95_low: "0.9004",
    ratio_of_ratios_ci95_high: "1.2951",
    p_two_sided: "0.4075",
  },
  {
    cohort: "sampled_primary",
    numerator_endpoint: "miat_footprint_mass_q95_positive_union_deduplicated",
    denominator_endpoint: "miat_footprint_mass_floor_union_deduplicated",
    numerator_ratio_kd_over_nt: "0.5419",
    denominator_ratio_kd_over_nt: "0.4799",
    ratio_of_ratios: "1.1293",
    ratio_of_ratios_ci95_low: "0.8298",
    ratio_of_ratios_ci95_high: "1.5368",
    p_two_sided: "0.4394",
  },
];
const summaryRows = buildProductionSummaryRows(endpointInference, ratioRows, {
  endpointRows: [
    {
      endpoint: "association_fraction_among_usable_q95",
      displayLabel: "q95 association fraction among null-usable MIAT spots",
      note: "Interpret with the conservative all-floor denominator because null unusability differs by arm.",
    },
    {
      endpoint: "association_fraction_among_all_floor_spots_q95",
      displayLabel: "q95 association fraction among all floor-detected MIAT spots",
      note: "Conservative denominator retaining null-unusable spots.",
    },
  ],
  ratioRows: ratioRows.map((row) => ({
    numeratorEndpoint: row.numerator_endpoint,
    denominatorEndpoint: row.denominator_endpoint,
    displayLabel: `ratio ${row.numerator_endpoint}`,
  })),
});
assert.equal(summaryRows.length, 4);
assert.equal(summaryRows[0].primary_exact_permutation_p, "0.03");
assert.match(summaryRows[0].note, /unusability differs by arm/i);
assert.equal(summaryRows[2].ratio_of_ratios, "1.0799");
assert.equal(summaryRows[3].ratio_of_ratios, "1.1293");

assert.doesNotThrow(() => assertNoFormulaErrors(
  '{"kind":"notice","message":"Cell search matched 0 entries."}',
));
assert.throws(
  () => assertNoFormulaErrors(
    '{"kind":"match","sheet":"Summary","address":"A1","value":"#REF!"}',
  ),
  /FORMULA_ERRORS_FOUND/,
);
assert.throws(
  () => assertNoFormulaErrors('{"kind":"notice","message":"scan completed"}'),
  /FORMULA_ERROR_SCAN_UNCERTAIN/,
);

const originalInspectCalls = [];
const fakeWorkbook = {
  async inspect(options) {
    originalInspectCalls.push(options);
    if (options?.maxChars === 0) throw new RangeError("Invalid string length");
    return { ndjson: "original bounded inspect" };
  },
};
const fakeExporter = async (workbook) => {
  const inspection = await workbook.inspect({
    kind: "workbook,sheet,table,region,formula,thread,conditionalFormatting,computedStyle,definedName,drawing",
    maxChars: 0,
  });
  return { attachedInspection: inspection.ndjson };
};
const boundedExport = await exportXlsxWithBoundedInspection(
  fakeWorkbook,
  '{"kind":"notice","message":"bounded verified export inspection"}',
  fakeExporter,
);
assert.equal(
  boundedExport.attachedInspection,
  '{"kind":"notice","message":"bounded verified export inspection"}',
);
await assert.rejects(
  fakeWorkbook.inspect({ kind: "workbook", maxChars: 0 }),
  /Invalid string length/,
  "The workbook inspect method must be restored after export",
);
assert.equal(originalInspectCalls.length, 1);

const safetyRoot = await fs.mkdtemp(path.join(os.tmpdir(), "miat-qki-workbook-safety-"));
const outputPath = path.join(safetyRoot, "existing.xlsx");
const verificationPath = path.join(safetyRoot, "existing-verification.json");
const previewDir = path.join(safetyRoot, "previews");
await fs.writeFile(outputPath, "existing", "utf8");
await fs.writeFile(verificationPath, "existing", "utf8");
await fs.mkdir(previewDir);
await fs.writeFile(path.join(previewDir, "summary.png"), "existing", "utf8");
await assert.rejects(
  assertSafeOutputTargets({ outputPath, verificationPath, previewDir, overwrite: false }, ["summary.png"]),
  /OUTPUT_EXISTS/,
);
await assert.doesNotReject(
  assertSafeOutputTargets({ outputPath, verificationPath, previewDir, overwrite: true }, ["summary.png"]),
);
await fs.writeFile(path.join(previewDir, "unexpected.txt"), "do not clobber", "utf8");
await assert.rejects(
  assertSafeOutputTargets({ outputPath, verificationPath, previewDir, overwrite: true }, ["summary.png"]),
  /UNSAFE_PREVIEW_DIRECTORY/,
);

process.stdout.write(`${JSON.stringify({
  status: "passed",
  pivotRows: pivoted.rows.length,
  controlStatuses: controls.map((row) => row.status),
  summaryRows: summaryRows.length,
  formulaGate: "verified",
  boundedExportInspection: "verified",
  safeOverwrite: "verified",
}, null, 2)}\n`);
