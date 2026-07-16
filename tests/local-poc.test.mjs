import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const artifactUrl = new URL("../app/generated-data.json", import.meta.url);
const pageUrl = new URL("../app/page.tsx", import.meta.url);
const stylesUrl = new URL("../app/globals.css", import.meta.url);

test("keeps the local POC and fitted model contract intact", async () => {
  const [artifactText, pageSource, stylesSource] = await Promise.all([
    readFile(artifactUrl, "utf8"),
    readFile(pageUrl, "utf8"),
    readFile(stylesUrl, "utf8"),
  ]);
  const artifact = JSON.parse(artifactText);

  assert.equal(artifact.meta.model.version, "2.3.2");
  assert.equal(artifact.meta.model.demandLibrary, "scikit-learn");
  assert.equal(artifact.meta.visionModel.modelId, "patrickjohncyh/fashion-clip");
  assert.equal(artifact.meta.model.topK, 8);
  assert.equal(Object.values(artifact.meta.matchConfidenceCounts).reduce((sum, value) => sum + value, 0), artifact.meta.upcomingItems);
  assert.equal(Object.values(artifact.meta.demandUncertaintyCounts).reduce((sum, value) => sum + value, 0), artifact.meta.upcomingItems);
  assert.equal(artifact.meta.attributeAudit.activeCount, 9);
  assert.deepEqual(Object.keys(artifact.meta.model.attributeWeights), [
    "category", "sleeve", "provision", "pattern", "lifecycle", "fit", "fabric", "colour", "price",
  ]);
  assert.deepEqual(artifact.meta.attributeAudit.excludedConstants.map(({ historicalColumn }) => historicalColumn), ["CAT2", "CAT5"]);
  assert.ok(artifact.upcoming.every(({ recommendation }) => (
    recommendation.confidence === recommendation.matchConfidence
    && ["Narrow", "Moderate", "Wide"].includes(recommendation.demandUncertainty)
    && Number.isFinite(recommendation.uncertaintyRatio)
  )));
  assert.ok(artifact.upcoming.some(({ recommendation }) => (
    recommendation.matchConfidence === "High"
    && recommendation.demandUncertainty === "Wide"
  )));
  assert.ok(artifact.upcoming.every(({ matches }) => matches.every(({ attributeBreakdown }) => (
    "lifecycle" in attributeBreakdown && !("range" in attributeBreakdown) && !("fashion" in attributeBreakdown)
  ))));
  assert.match(pageSource, /FashionCLIP retrieval \+ scikit-learn demand model/);
  assert.match(pageSource, /Match confidence/);
  assert.match(pageSource, /Demand uncertainty/);
  assert.match(pageSource, /Top historical analogue/);
  assert.match(pageSource, /Analogue-based demand/);
  assert.match(pageSource, /Historical catalog details/);
  assert.match(pageSource, /Select for match evidence/);
  assert.match(pageSource, /historical-spec-grid/);
  assert.doesNotMatch(pageSource, /Strong attribute matches/);
  assert.match(pageSource, /All \{Object\.keys\(focusedMatch\.attributeBreakdown\)\.length\} upcoming and historical comparison fields/);
  assert.match(pageSource, /attributeValueReaders/);
  assert.match(pageSource, /Season family/);
  assert.match(pageSource, /Workbook attribute audit/);
  assert.match(pageSource, /informative fields retained/);
  assert.match(pageSource, /Excluded constant/);
  assert.match(pageSource, /Exact match/);
  assert.match(pageSource, /Upcoming/);
  assert.match(pageSource, /Historical/);
  assert.match(pageSource, /View all product details/);
  assert.match(pageSource, /Colour variant code/);
  assert.match(pageSource, /FashionCLIP ready/);
  assert.match(pageSource, /Constant range codes are intentionally hidden/);
  assert.match(pageSource, /queue-commercial/);
  assert.match(stylesSource, /\.historical-spec-grid\s*\{[^}]*grid-template-columns:\s*repeat\(2,/s);
  assert.match(stylesSource, /\.historical-spec-grid > div\s*\{[^}]*border-radius:\s*7px/s);
  assert.match(stylesSource, /\.hero-product-image\s*\{[^}]*object-fit:\s*contain/s);
  assert.match(stylesSource, /\.match-image\s*\{[^}]*height:\s*320px[^}]*object-fit:\s*contain/s);
  assert.match(stylesSource, /\.recommendation-metrics\s*\{[^}]*grid-template-columns:\s*repeat\(2,/s);
  assert.match(stylesSource, /\.recommendation-metrics small,[^}]*overflow-wrap:\s*anywhere/s);
  assert.match(stylesSource, /\.queue-commercial\s*\{[^}]*justify-content:\s*space-between/s);
  assert.match(stylesSource, /\.product-details-grid\s*\{[^}]*grid-template-columns:\s*repeat\(2,/s);
});
