import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

const pageUrl = new URL("../app/page.tsx", import.meta.url);
const cssUrl = new URL("../app/globals.css", import.meta.url);
const nextConfigUrl = new URL("../next.config.ts", import.meta.url);

async function exists(url) {
  try {
    await access(url);
    return true;
  } catch {
    return false;
  }
}

test("keeps the pooled predictive forecast contract intact", async () => {
  const pageSource = await readFile(pageUrl, "utf8");

  assert.match(pageSource, /Minimum visual similarity/);
  assert.match(pageSource, /Target sell-through/);
  assert.match(pageSource, /Ranked by how closely the product photos match/);
  assert.match(pageSource, /No candidate meets the/);
  assert.match(pageSource, /const eligible = ranked\.filter/);
  assert.match(pageSource, /function newsvendorOrder/);
  assert.match(pageSource, /function forecastDemand/);
  assert.match(pageSource, /Why this recommendation/);
  assert.match(pageSource, /sellThroughTradeoff/);
  assert.doesNotMatch(pageSource, /Forecast demand: <b>/);
  assert.match(pageSource, /wideUncertainty/);
  assert.match(pageSource, /medianDemand/);
  assert.match(pageSource, /quantityCapped/);
  assert.match(pageSource, /highCapped/);
  assert.match(pageSource, /function ceilingFor/);
  assert.match(pageSource, /packRoundedUncapped/);
  assert.doesNotMatch(pageSource, /Attribute weight|Inventory strategy|Machine-learning sales forecast|Sales backtest WAPE/);
  assert.doesNotMatch(pageSource, /colourFamily|Colour family/);
  // Keep the planner export focused on decisions. These model-internal CSV
  // columns previously exposed competing forecast values and unexplained
  // percentile jargon, making a single recommendation look contradictory.
  assert.doesNotMatch(pageSource, /"Forecast demand \(average\)"/);
  assert.doesNotMatch(pageSource, /"Forecast low \(p10\)"/);
  assert.doesNotMatch(pageSource, /"Forecast high \(p90\)"/);
  assert.doesNotMatch(pageSource, /"Range tail hit ceiling"/);
  assert.match(pageSource, /"Product ID"/);
  assert.match(pageSource, /"Overall similarity \(%\)"/);
  assert.match(pageSource, /"Colour similarity \(%\)"/);
  assert.match(pageSource, /"Pattern similarity \(%\)"/);
  assert.match(pageSource, /"Style similarity \(%\)"/);
  assert.match(pageSource, /"Texture similarity \(%\)"/);
  assert.match(pageSource, /"Expected sales \(units\)"/);
  assert.match(pageSource, /"Recommended order \(units\)"/);
  assert.match(pageSource, /"Final order \(units\)"/);
  assert.doesNotMatch(pageSource, /"Estimated demand \(units\)"/);
  assert.doesNotMatch(pageSource, /"Analogue actual sales"|"Analogue original order"|"AI recommended quantity"/);
});

test("ships no catalogue of its own and renders an empty state until a build is activated", async () => {
  const pageSource = await readFile(pageUrl, "utf8");

  // The planner is upload-only: no bundled artifact, and no local image route
  // reading a developer's DATA/ directory.
  assert.equal(await exists(new URL("../app/generated-data.json", import.meta.url)), false);
  assert.equal(await exists(new URL("../app/product-images", import.meta.url)), false);
  assert.doesNotMatch(pageSource, /generated-data/);
  assert.doesNotMatch(pageSource, /product-images/);
  assert.doesNotMatch(pageSource, /"bundled"/);

  assert.match(pageSource, /const EMPTY_DATASET: Dataset/);
  assert.match(pageSource, /hasActiveBuild/);
  assert.match(pageSource, /no-build-state/);
  assert.match(pageSource, /Upload a catalogue to start planning/);
  // Both data-backed views must be gated, or an empty dataset renders a
  // workspace full of zeroes instead of the empty state.
  assert.match(pageSource, /tab === "compare" && hasActiveBuild/);
  assert.match(pageSource, /tab === "portfolio" && hasActiveBuild/);
  // Every artifact-derived lookup has to be refreshed per build; capturing the
  // demand model once at module load silently reused the previous build's priors.
  assert.match(pageSource, /demandModel = dataset\.meta\.model\.demandModel/);
});

test("ships the dual-mode resumable upload workflow", async () => {
  const [pageSource, cssSource, nextConfig] = await Promise.all([
    readFile(pageUrl, "utf8"),
    readFile(cssUrl, "utf8"),
    readFile(nextConfigUrl, "utf8"),
  ]);

  assert.match(pageSource, /full_replace/);
  assert.match(pageSource, /reuse_historical/);
  assert.match(pageSource, /Reuse trained historical/);
  assert.match(pageSource, /Build recommendations/);
  assert.match(pageSource, /Upload catalogue data and product images to create a validated recommendation build/);
  assert.match(pageSource, /Catalogue templates/);
  assert.match(pageSource, /Correct columns with one example row included/);
  assert.match(pageSource, /Download CSV template/);
  assert.match(pageSource, /historical-catalogue\.csv" download/);
  assert.match(pageSource, /upcoming-catalogue\.csv" download/);
  assert.doesNotMatch(pageSource, />Historical CSV template</);
  assert.doesNotMatch(pageSource, />Upcoming CSV template</);
  assert.doesNotMatch(pageSource, />New analysis</);
  assert.match(pageSource, /webkitdirectory/);
  assert.match(pageSource, /uploadPool/);
  assert.match(pageSource, /complete-upload/);
  assert.match(pageSource, /EventSource/);
  assert.match(pageSource, /validationReportUrl/);
  assert.match(pageSource, /CSV, XLSX, XLSM, XLS, XLSB, or ODS/);
  assert.match(cssSource, /\.upload-workspace/);
  assert.match(nextConfig, /TURTLE_API_URL/);
  assert.match(nextConfig, /source: "\/api\/:path\*"/);
});

test("keeps an analysis mounted while navigating between planner tabs", async () => {
  const pageSource = await readFile(pageUrl, "utf8");

  // The live EventSource and upload state belong to NewAnalysis. It must only
  // be hidden between tabs; conditional rendering would unmount and reset it.
  assert.match(pageSource, /<NewAnalysis\s+active=\{tab === "upload"\}/);
  assert.match(pageSource, /className="upload-workspace page-wrap" hidden=\{!active\}/);
  assert.doesNotMatch(pageSource, /\{tab === "upload" && \(\s*<NewAnalysis/);
});

test("keeps internal run and build IDs out of planner-facing labels", async () => {
  const [pageSource, cssSource] = await Promise.all([
    readFile(pageUrl, "utf8"),
    readFile(cssUrl, "utf8"),
  ]);

  assert.doesNotMatch(pageSource, /historical\.id\.slice\(0, 8\)/);
  assert.doesNotMatch(pageSource, /run\.id\.slice\(0, 8\)/);
  assert.doesNotMatch(pageSource, /activeBuildId\.slice\(0, 8\)/);
  assert.doesNotMatch(pageSource, /buildId\.slice\(0, 8\)/);
  assert.doesNotMatch(pageSource, /Current analysis/);
  assert.match(pageSource, /Last updated/);
  assert.doesNotMatch(pageSource, /Recommendations last updated/);
  assert.match(pageSource, /New recommendations activated/);
  assert.match(pageSource, /Upload complete — ready for analysis/);
  assert.match(pageSource, /uploaded successfully\. Select Start analysis to begin image processing\./);
  assert.match(pageSource, /DISPLAY_MODEL_VERSION = "1\.0"/);
  assert.doesNotMatch(pageSource, /v\{historical\.modelVersion\}/);
  assert.match(pageSource, /timeZone: "Asia\/Kolkata"/);
  assert.doesNotMatch(pageSource, /timeZoneName/);
  assert.doesNotMatch(pageSource, /formatAnalysisDate\(run\.createdAt\)/);
  assert.match(pageSource, /Last updated · \$\{formatAnalysisDate\(activeBuildCreatedAt\)\}/);
  assert.match(pageSource, /className="historical-updated"/);
  assert.match(cssSource, /grid-template-columns: minmax\(350px, 1\.35fr\) repeat\(3, minmax\(0, 1fr\)\)/);
  assert.match(cssSource, /\.historical-summary \.historical-updated\s*\{\s*white-space: nowrap !important;/);
});

test("recovers live progress after refreshes and interrupted event streams", async () => {
  const pageSource = await readFile(pageUrl, "utf8");

  assert.match(pageSource, /ACTIVE_ANALYSIS_RUN_KEY/);
  assert.match(pageSource, /localStorage\.getItem\(ACTIVE_ANALYSIS_RUN_KEY\)/);
  assert.match(pageSource, /fetch\("\/api\/runs\/active"/);
  assert.match(pageSource, /new EventSource\(`\/api\/runs\/\$\{runId\}\/events`\)/);
  assert.match(pageSource, /async function pollRun\(\)/);
  assert.match(pageSource, /setInterval\(\(\) => void pollRun\(\), RUN_POLL_INTERVAL_MS\)/);
  assert.match(pageSource, /Live progress was interrupted; checking the run automatically/);
  assert.match(pageSource, /terminalHandled = true;\s*progressEvents\?\.close\(\);/);
  assert.match(pageSource, /if \(!disposed && !terminalHandled\)/);

  // Tracking belongs to a lifecycle effect so Fast Refresh can recreate it;
  // the button handler should only transition the server into the queued state.
  const startBody = pageSource.slice(
    pageSource.indexOf("async function startAnalysis("),
    pageSource.indexOf("async function cancelAnalysis("),
  );
  assert.doesNotMatch(startBody, /new EventSource/);
});

test("shows useful analysis timing instead of an unavailable ETA", async () => {
  const pageSource = await readFile(pageUrl, "utf8");

  assert.match(pageSource, /startedAt\?: string \| null/);
  assert.match(pageSource, /completedAt\?: string \| null/);
  assert.match(pageSource, /Time elapsed/);
  assert.match(pageSource, /Total analysis time/);
  assert.match(pageSource, /analysisElapsedSeconds/);
  assert.match(pageSource, /setElapsedClock\(Date\.now\(\)\), 1000/);
  assert.doesNotMatch(pageSource, /Time remaining|Estimating…/);
});

test("separates uploading from analysing, and never invites a click on running work", async () => {
  const [pageSource, cssSource] = await Promise.all([
    readFile(pageUrl, "utf8"),
    readFile(cssUrl, "utf8"),
  ]);

  // Two named steps, not one button that silently does both.
  assert.match(pageSource, /"Upload files"/);
  assert.match(pageSource, /"Start analysis"/);
  assert.match(pageSource, /async function uploadFiles\(/);
  assert.match(pageSource, /async function startAnalysis\(/);
  // complete-upload belongs to step two only; uploading must not trigger the build.
  const uploadBody = pageSource.slice(
    pageSource.indexOf("async function uploadFiles("),
    pageSource.indexOf("async function startAnalysis("),
  );
  assert.doesNotMatch(uploadBody, /complete-upload/);

  // Every phase the primary button can be in.
  for (const phase of ["idle", "uploading", "upload-failed", "uploaded", "analysing", "done"]) {
    assert.ok(pageSource.includes(`"${phase}"`), `missing upload phase: ${phase}`);
  }
  // Running work must not be styled as a call to action, and must be blocked.
  assert.match(pageSource, /primaryDisabled \? "" : "primary"/);
  assert.match(pageSource, /const running = phase === "uploading" \|\| phase === "analysing"/);
  assert.match(pageSource, /disabled=\{primaryDisabled\}/);
  assert.match(cssSource, /\.button:disabled/);
  assert.match(cssSource, /\.button-spinner/);
  // A resumable upload must survive a failure rather than restart from zero.
  assert.match(pageSource, /Resume upload/);
  assert.doesNotMatch(pageSource, /Successful replacement permanently removes the superseded active data/);
  assert.doesNotMatch(pageSource, /Uploading only stages the files; no data is replaced until the analysis succeeds/);
});
