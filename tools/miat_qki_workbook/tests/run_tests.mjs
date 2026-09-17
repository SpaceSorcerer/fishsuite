import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const testDir = path.dirname(fileURLToPath(import.meta.url));
const root = path.dirname(testDir);
const fixtureDir = path.join(testDir, "fixtures");
const outputDir = path.join(testDir, "output");
const scratchRoot = path.join(testDir, "tmp");
const builder = path.join(root, "build_workbook.mjs");
const baseConfig = path.join(root, "schema.default.json");
const config = path.join(outputDir, "schema.compact-style-test.json");
const workbookPath = path.join(outputDir, "synthetic_miat_qki_postrun.xlsx");
const previewDir = path.join(outputDir, "previews");
const verificationPath = path.join(outputDir, "verification.json");

function runBuilder(args) {
  return spawnSync(process.execPath, [builder, ...args], {
    cwd: root,
    encoding: "utf8",
    maxBuffer: 10 * 1024 * 1024,
  });
}

async function copyFixtures(destination, options = {}) {
  await fs.mkdir(destination, { recursive: true });
  for (const filename of await fs.readdir(fixtureDir)) {
    if (options.omit === filename) continue;
    const source = path.join(fixtureDir, filename);
    const target = path.join(destination, filename);
    if (options.rewrite?.filename === filename) {
      const original = await fs.readFile(source, "utf8");
      await fs.writeFile(target, options.rewrite.transform(original), "utf8");
    } else {
      await fs.copyFile(source, target);
    }
  }
}

function expectFailure(result, code, detail) {
  assert.notEqual(result.status, 0, `Expected builder failure ${code}`);
  assert.match(result.stderr, new RegExp(code));
  if (detail) assert.match(result.stderr, new RegExp(detail));
}

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(scratchRoot, { recursive: true });
const testConfig = JSON.parse(await fs.readFile(baseConfig, "utf8"));
testConfig.tables.find((table) => table.key === "nucleus").compactStyle = true;
await fs.writeFile(config, `${JSON.stringify(testConfig, null, 2)}\n`, "utf8");

const success = runBuilder([
  "--input-dir", fixtureDir,
  "--config", config,
  "--output", workbookPath,
  "--preview-dir", previewDir,
  "--verification-json", verificationPath,
  "--overwrite", "true",
]);
assert.equal(success.status, 0, `Workbook build failed:\n${success.stderr}\n${success.stdout}`);
const resultLine = success.stdout
  .trim()
  .split(/\r?\n/)
  .reverse()
  .find((line) => line.startsWith('{"status":'));
assert.ok(resultLine, `Builder did not emit a result record:\n${success.stdout}`);
const buildSummary = JSON.parse(resultLine);
assert.equal(buildSummary.sheetCount, 11);
assert.equal(buildSummary.superplotCount, 2);
assert.equal(buildSummary.previewCount, 11);

const workbookStat = await fs.stat(workbookPath);
assert.ok(workbookStat.size > 20_000, "Synthetic workbook is unexpectedly small");
const blob = await FileBlob.load(workbookPath);
const workbook = await SpreadsheetFile.importXlsx(blob);
const expectedSheets = [
  "README",
  "Data Dictionary",
  "Summary",
  "Nucleus",
  "FOV",
  "Set",
  "Inference",
  "Correlation",
  "Representative",
  "Control",
  "QC",
];
for (const sheetName of expectedSheets) {
  assert.ok(workbook.worksheets.getItem(sheetName), `Missing worksheet ${sheetName}`);
}

const sheetInspect = await workbook.inspect({
  kind: "sheet",
  include: "id,name",
  maxChars: 8000,
});
const summaryInspect = await workbook.inspect({
  kind: "table,formula,drawing",
  sheetId: "Summary",
  range: "A1:O55",
  maxChars: 8000,
  tableMaxRows: 20,
  tableMaxCols: 15,
  options: { maxResults: 150 },
});
const formulaErrorInspect = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "post-export formula error scan",
  maxChars: 8000,
});

const summarySheet = workbook.worksheets.getItem("Summary");
const formulas = summarySheet.getRange("A1:O55").formulas.flat();
assert.ok(
  formulas.filter((value) => typeof value === "string" && value.startsWith("=")).length >= 12,
  "Expected formula-linked superplot helper cells",
);
assert.equal(summarySheet.charts.items.length, 2, "Expected two native superplot charts");
assert.equal(
  workbook.worksheets.getItem("FOV").getRange("F2:G2").format.numberFormat,
  "#,##0.000",
  "Per-nucleus FOV averages must retain decimal precision",
);
assert.notEqual(
  workbook.worksheets.getItem("Nucleus").getRange("G2:G2").format.numberFormat,
  "#,##0",
  "Compact raw sheets must not expand per-cell numeric formats",
);
assert.equal(
  workbook.worksheets.getItem("Nucleus").tables.items.length,
  1,
  "Compact raw sheets must retain their formal Excel table and autofilter",
);
assert.deepEqual(
  workbook.worksheets.getItem("Control").getRange("D2:E2").values,
  [[0, 0]],
  "Control null-usable and q95-positive counts must remain numeric",
);
assert.equal(
  workbook.worksheets.getItem("Control").getRange("D2:E2").format.numberFormat,
  "#,##0",
  "Control counts must use integer display formatting",
);
assert.equal(
  workbook.worksheets.getItem("Summary").getRange("F2:F2").format.wrapText,
  true,
  "Long summary interpretations must wrap",
);
assert.equal(
  workbook.worksheets.getItem("Summary").getRange("A:A").format.columnWidth,
  38,
  "Summary endpoint labels must not be clipped by the superplot helper layout",
);

const formulaErrorPattern = /#REF!|#DIV\/0!|#VALUE!|#NAME\?|#N\/A/;
for (const sheetName of expectedSheets) {
  const sheet = workbook.worksheets.getItem(sheetName);
  const used = sheet.getUsedRange(true) ?? sheet.getUsedRange();
  if (!used) continue;
  const errors = used.values
    .flat()
    .filter((value) => typeof value === "string" && formulaErrorPattern.test(value));
  assert.deepEqual(errors, [], `Formula errors found on ${sheetName}`);
}

const dictionaryValues = workbook
  .worksheets
  .getItem("Data Dictionary")
  .getUsedRange(true)
  .values
  .flat();
assert.ok(
  dictionaryValues.includes("unexpected_extra_metric"),
  "Schema evolution check failed: extra source column is absent from Data Dictionary",
);

for (const sheetName of expectedSheets) {
  const safeName = sheetName
    .toLowerCase()
    .replaceAll(/[^a-z0-9]+/g, "_")
    .replaceAll(/^_|_$/g, "");
  const previewPath = path.join(previewDir, `${safeName}.png`);
  const stat = await fs.stat(previewPath);
  assert.ok(stat.size > 1_000, `Preview for ${sheetName} is unexpectedly small`);
}

const verification = JSON.parse(await fs.readFile(verificationPath, "utf8"));
assert.equal(verification.status, "success");
assert.equal(verification.tables.length, 9);
assert.deepEqual(verification.sheets, expectedSheets);
assert.equal(verification.previewFiles.length, 11);

const refusal = runBuilder([
  "--input-dir", fixtureDir,
  "--config", config,
  "--output", workbookPath,
  "--preview-dir", previewDir,
  "--verification-json", verificationPath,
]);
expectFailure(refusal, "OUTPUT_EXISTS", "Refusing to overwrite");

const missingTableDir = await fs.mkdtemp(path.join(scratchRoot, "missing-table-"));
await copyFixtures(missingTableDir, { omit: "qc.csv" });
const missingTable = runBuilder([
  "--input-dir", missingTableDir,
  "--config", config,
  "--output", path.join(missingTableDir, "must_not_exist.xlsx"),
]);
expectFailure(missingTable, "MISSING_REQUIRED_TABLES", "qc.csv");

const missingColumnDir = await fs.mkdtemp(path.join(scratchRoot, "missing-column-"));
await copyFixtures(missingColumnDir, {
  rewrite: {
    filename: "set.csv",
    transform: (text) => {
      const lines = text.trimEnd().split(/\r?\n/);
      const headers = lines[0].split(",");
      const removeIndex = headers.indexOf("association_fraction_among_usable");
      assert.ok(removeIndex >= 0);
      return `${lines
        .map((line) =>
          line
            .split(",")
            .filter((_, index) => index !== removeIndex)
            .join(","),
        )
        .join("\n")}\n`;
    },
  },
});
const missingColumn = runBuilder([
  "--input-dir", missingColumnDir,
  "--config", config,
  "--output", path.join(missingColumnDir, "must_not_exist.xlsx"),
]);
expectFailure(
  missingColumn,
  "MISSING_REQUIRED_COLUMNS",
  "association_fraction_among_usable",
);

process.stdout.write(
  `${JSON.stringify({
    status: "passed",
    workbook: workbookPath,
    workbookBytes: workbookStat.size,
    sheets: expectedSheets,
    formulaLinkedCells: formulas.filter(
      (value) => typeof value === "string" && value.startsWith("="),
    ).length,
    charts: summarySheet.charts.items.length,
    previews: expectedSheets.length,
    missingTableFailure: "MISSING_REQUIRED_TABLES",
    missingColumnFailure: "MISSING_REQUIRED_COLUMNS",
    existingOutputFailure: "OUTPUT_EXISTS",
    sheetInspectRecords: sheetInspect.ndjson.trim().split(/\r?\n/).length,
    summaryInspectRecords: summaryInspect.ndjson.trim().split(/\r?\n/).length,
    formulaErrorScan: formulaErrorInspect.ndjson,
  }, null, 2)}\n`,
);
