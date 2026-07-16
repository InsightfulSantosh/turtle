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
  assert.match(pageSource, /Why this analogue matched/);
  assert.match(pageSource, /Similarity scores/);
  assert.match(pageSource, /Attribute-by-attribute comparison/);
  assert.match(pageSource, /Upcoming style/);
  assert.match(pageSource, /Historical analogue/);
  assert.match(pageSource, /Image score uses FashionCLIP/);
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
  assert.match(stylesSource, /\.match-grid\.match-grid-3\s*\{[^}]*repeat\(3,\s*max\(200px,\s*calc\(\(100% - 36px\) \/ 4\)\)\)/s);
  assert.match(stylesSource, /\.match-grid\.match-grid-5\s*\{[^}]*repeat\(5,\s*max\(200px,\s*calc\(\(100% - 36px\) \/ 4\)\)\)/s);
  assert.match(stylesSource, /\.match-grid\.match-grid-8\s*\{[^}]*repeat\(8,\s*max\(200px,\s*calc\(\(100% - 36px\) \/ 4\)\)\)/s);
  assert.match(stylesSource, /\.match-grid\.match-grid-3,\s*\.match-grid\.match-grid-5,\s*\.match-grid\.match-grid-8\s*\{[^}]*column-gap:\s*12px[^}]*overflow-x:\s*auto[^}]*padding:\s*0 12px 6px/s);
  assert.match(stylesSource, /\.catalog-attribute-grid dd\s*\{[^}]*overflow-wrap:\s*break-word[^}]*word-break:\s*normal/s);
  assert.match(stylesSource, /\.catalog-attribute-toggle\s*\{[^}]*width:\s*auto/s);
  assert.match(stylesSource, /\.match-card-select\s*\{[^}]*width:\s*100%/s);
  assert.match(stylesSource, /\.match-card::after\s*\{[^}]*border:\s*2px solid transparent[^}]*inset:\s*0[^}]*pointer-events:\s*none[^}]*z-index:\s*3/s);
  assert.match(stylesSource, /\.match-card:hover::after,\s*\.match-card\.active::after\s*\{[^}]*border-color:\s*var\(--forest-2\)/s);
  assert.doesNotMatch(stylesSource, /\.match-card:hover,\s*\.match-card\.active\s*\{[^}]*transform:/s);
  assert.match(stylesSource, /\.match-card \.catalog-attribute-grid > div\s*\{[^}]*display:\s*flex[^}]*flex-direction:\s*column[^}]*min-height:\s*26px/s);
  assert.doesNotMatch(stylesSource, /\.match-card \.catalog-attribute-grid > div:nth-child/);
  assert.match(stylesSource, /\.match-performance\s*\{[^}]*margin-top:\s*4px[^}]*padding-top:\s*4px/s);
  assert.match(stylesSource, /\.match-performance span\s*\{[^}]*font-size:\s*10px/s);
  assert.match(stylesSource, /\.match-performance small\s*\{[^}]*white-space:\s*nowrap/s);
  assert.doesNotMatch(stylesSource, /\.match-card \.match-attribute-catalog-heading[^}]*font-size/s);
  assert.doesNotMatch(stylesSource, /\.match-card \.catalog-attribute-grid (?:dt|dd)\s*\{[^}]*font-size/s);
  assert.match(stylesSource, /\.match-card \.catalog-attribute-grid dd\s*\{[^}]*font-weight:\s*700[^}]*min-height:\s*0/s);
  assert.match(stylesSource, /\.match-card \.catalog-attribute-toggle\s*\{[^}]*min-height:\s*20px/s);
  assert.match(stylesSource, /\.evidence-header\s*\{[^}]*grid-template-columns:\s*minmax\(0, 1fr\) 230px/s);
  assert.match(stylesSource, /\.evidence-product-pair\s*\{[^}]*grid-template-columns:\s*minmax\(0, 1fr\) 28px minmax\(0, 1fr\)/s);
  assert.match(stylesSource, /\.attribute-evidence\s*\{[^}]*grid-template-columns:\s*repeat\(3,/s);
  assert.match(stylesSource, /\.hero-product-image\s*\{[^}]*aspect-ratio:\s*auto 3 \/ 4[^}]*border-radius:\s*13px[^}]*height:\s*auto[^}]*margin:\s*12px auto 0[^}]*max-width:\s*100%[^}]*object-fit:\s*contain[^}]*object-position:\s*center[^}]*width:\s*100%/s);
  assert.match(stylesSource, /\.image-fallback\.hero-product-image\s*\{[^}]*aspect-ratio:\s*3 \/ 4/s);
  assert.match(stylesSource, /\.match-image\s*\{[^}]*aspect-ratio:\s*5 \/ 6[^}]*height:\s*auto[^}]*object-fit:\s*contain/s);
  assert.match(stylesSource, /\.image-fallback\.match-image\s*\{[^}]*aspect-ratio:\s*5 \/ 6/s);
  assert.match(stylesSource, /\.recommendation-metrics\s*\{[^}]*grid-template-columns:\s*repeat\(2,/s);
  assert.match(stylesSource, /\.recommendation-metrics small,[^}]*overflow-wrap:\s*anywhere/s);
  assert.match(stylesSource, /\.queue-commercial\s*\{[^}]*justify-content:\s*space-between/s);
  assert.doesNotMatch(stylesSource, /\.product-details-grid\s*\{/);
  assert.doesNotMatch(stylesSource, /\.historical-spec-grid\s*\{/);
});
