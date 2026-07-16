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
  assert.match(pageSource, /MatchAttributeCatalog/);
  assert.match(pageSource, /Product attributes/);
  assert.match(pageSource, /\$\{context\} product attributes/);
  assert.doesNotMatch(pageSource, /Upcoming match attributes/);
  assert.doesNotMatch(pageSource, /Historical match attributes/);
  assert.match(pageSource, /View all 9/);
  assert.match(pageSource, /Show key attributes/);
  assert.doesNotMatch(pageSource, /\+5 Show all 9/);
  assert.doesNotMatch(pageSource, /4 of 9 shown/);
  assert.doesNotMatch(pageSource, /9 of 9 shown/);
  assert.match(pageSource, /match-card-select/);
  assert.match(pageSource, /Select \$\{historical\.id\} for match evidence/);
  assert.doesNotMatch(pageSource, /View match evidence/);
  assert.doesNotMatch(pageSource, /Strong attribute matches/);
  assert.match(pageSource, /Both catalogs show the same four primary fields/);
  assert.match(pageSource, /attributeValueReaders/);
  assert.match(pageSource, /catalogAttributeOrder/);
  assert.match(pageSource, /const catalogAttributeOrder = \[\s*"colour",\s*"price",\s*"pattern",\s*"fabric",/s);
  assert.match(pageSource, /Season family/);
  assert.match(pageSource, /Workbook attribute audit/);
  assert.match(pageSource, /informative fields retained/);
  assert.match(pageSource, /Excluded constant/);
  assert.match(pageSource, /Exact match/);
  assert.match(pageSource, /Upcoming/);
  assert.match(pageSource, /Historical/);
  assert.doesNotMatch(pageSource, /product-details-grid/);
  assert.doesNotMatch(pageSource, /historical-spec-grid/);
  assert.doesNotMatch(pageSource, /View all product details/);
  assert.match(pageSource, /queue-commercial/);
  assert.match(stylesSource, /\.catalog-attribute-grid\s*\{[^}]*grid-template-columns:\s*repeat\(2,/s);
  assert.match(stylesSource, /\.match-grid\.match-grid-5\s*\{[^}]*repeat\(5,\s*max\(210px,\s*calc\(\(100% - 30px\) \/ 4\)\)\)/s);
  assert.match(stylesSource, /\.match-grid\.match-grid-8\s*\{[^}]*repeat\(8,\s*max\(210px,\s*calc\(\(100% - 30px\) \/ 4\)\)\)/s);
  assert.match(stylesSource, /\.match-grid\.match-grid-5,\s*\.match-grid\.match-grid-8\s*\{[^}]*overflow-x:\s*auto/s);
  assert.match(stylesSource, /\.catalog-attribute-grid dd\s*\{[^}]*overflow-wrap:\s*break-word[^}]*word-break:\s*normal/s);
  assert.match(stylesSource, /\.catalog-attribute-toggle\s*\{[^}]*width:\s*auto/s);
  assert.match(stylesSource, /\.match-card-select\s*\{[^}]*width:\s*100%/s);
  assert.match(stylesSource, /\.match-card \.catalog-attribute-grid > div\s*\{[^}]*display:\s*flex[^}]*flex-direction:\s*column[^}]*min-height:\s*28px/s);
  assert.doesNotMatch(stylesSource, /\.match-card \.catalog-attribute-grid > div:nth-child/);
  assert.match(stylesSource, /\.match-performance\s*\{[^}]*margin-top:\s*5px[^}]*padding-top:\s*5px/s);
  assert.match(stylesSource, /\.match-performance span\s*\{[^}]*font-size:\s*10px/s);
  assert.match(stylesSource, /\.match-performance small\s*\{[^}]*white-space:\s*nowrap/s);
  assert.doesNotMatch(stylesSource, /\.match-card \.match-attribute-catalog-heading[^}]*font-size/s);
  assert.doesNotMatch(stylesSource, /\.match-card \.catalog-attribute-grid (?:dt|dd)\s*\{[^}]*font-size/s);
  assert.match(stylesSource, /\.match-card \.catalog-attribute-grid dd\s*\{[^}]*font-weight:\s*700[^}]*min-height:\s*0/s);
  assert.match(stylesSource, /\.match-card \.catalog-attribute-toggle\s*\{[^}]*min-height:\s*20px/s);
  assert.match(stylesSource, /\.hero-product-image\s*\{[^}]*object-fit:\s*contain/s);
  assert.match(stylesSource, /\.match-image\s*\{[^}]*height:\s*320px[^}]*object-fit:\s*contain/s);
  assert.match(stylesSource, /\.recommendation-metrics\s*\{[^}]*grid-template-columns:\s*repeat\(2,/s);
  assert.match(stylesSource, /\.recommendation-metrics small,[^}]*overflow-wrap:\s*anywhere/s);
  assert.match(stylesSource, /\.queue-commercial\s*\{[^}]*justify-content:\s*space-between/s);
  assert.doesNotMatch(stylesSource, /\.product-details-grid\s*\{/);
  assert.doesNotMatch(stylesSource, /\.historical-spec-grid\s*\{/);
});
