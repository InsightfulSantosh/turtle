import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const artifactUrl = new URL("../app/generated-data.json", import.meta.url);
const pageUrl = new URL("../app/page.tsx", import.meta.url);
const stylesUrl = new URL("../app/globals.css", import.meta.url);

test("keeps the local application and fitted model contract intact", async () => {
  const [artifactText, pageSource, stylesSource] = await Promise.all([
    readFile(artifactUrl, "utf8"),
    readFile(pageUrl, "utf8"),
    readFile(stylesUrl, "utf8"),
  ]);
  const artifact = JSON.parse(artifactText);

  assert.equal(artifact.meta.model.version, "4.2.0");
  assert.equal(artifact.meta.dataMode, "real");
  assert.equal(artifact.meta.upcomingSeason, "SS27");
  assert.equal(artifact.meta.model.demandLibrary, "scikit-learn");
  assert.equal(artifact.meta.visionModel.modelId, "Marqo/marqo-fashionSigLIP");
  assert.equal(artifact.meta.visionModel.reranker.modelId, "facebook/dinov2-base");
  assert.equal(artifact.meta.visionModel.reranker.candidateCount, 50);
  assert.equal(artifact.meta.visionModel.reranker.sameItemTypeConstraint, true);
  assert.equal(artifact.meta.model.dinoRerankWeight, 0.5);
  assert.deepEqual(artifact.meta.visionModel.reranker.weightGrid, [0.5]);
  assert.equal(artifact.meta.visionModel.reranker.candidateIndex.metric, "inner-product on L2-normalized FashionSigLIP embeddings");
  assert.equal(artifact.meta.visionModel.reranker.appearance.segmentation.method, "adaptive-lab-border-foreground-mask");
  assert.deepEqual(artifact.meta.visionModel.reranker.appearance.weights, {
    neural: 0.7,
    colour: 0.2,
    texture: 0.1,
  });
  assert.match(artifact.meta.visualMethod, /Two-stage FashionSigLIP.*DINOv2/);
  assert.equal(artifact.meta.historicalImageCoverage, 508);
  assert.equal(artifact.meta.upcomingImageCoverage, 36);
  assert.ok([3, 5, 8].includes(artifact.meta.model.topK));
  assert.equal(artifact.meta.model.minimumVisualScore, 0.5);
  assert.equal(artifact.meta.model.minimumMatchConfidence, "Medium");
  assert.ok(artifact.meta.model.attributeWeight > 0);
  assert.ok(artifact.meta.model.visualWeight > 0);
  assert.equal(
    artifact.meta.model.attributeWeight + artifact.meta.model.visualWeight,
    1,
  );
  assert.equal(artifact.meta.model.modelSelection, "Temporal holdout + ParameterGrid");
  assert.equal(Object.values(artifact.meta.matchConfidenceCounts).reduce((sum, value) => sum + value, 0), artifact.meta.upcomingItems);
  assert.equal(Object.values(artifact.meta.demandUncertaintyCounts).reduce((sum, value) => sum + value, 0), artifact.meta.upcomingItems);
  assert.equal(artifact.meta.attributeAudit.activeCount, 5);
  assert.equal(artifact.meta.dataQuality.zeroSalesHistoricalRowsExcluded, 142);
  assert.equal(artifact.meta.dataQuality.upcomingRowsExcludedUnseenItem, 114);
  assert.ok(artifact.historical.every(({ salesTarget }) => salesTarget > 0));
  assert.ok(artifact.upcoming.every(({ itemType }) => itemType !== "OTJT"));
  assert.equal(
    artifact.upcoming.filter(({ imageUrl }) => Boolean(imageUrl)).length,
    artifact.meta.upcomingImageCoverage,
  );
  assert.equal(
    artifact.historical.filter(({ imageUrl }) => Boolean(imageUrl)).length,
    artifact.meta.historicalImageCoverage,
  );
  assert.deepEqual(Object.keys(artifact.meta.model.attributeWeights), [
    "item", "design", "category_type", "fabric", "colour",
  ]);
  assert.deepEqual(artifact.meta.attributeAudit.excludedConstants, []);
  assert.ok(artifact.upcoming.every(({ recommendation }) => (
    recommendation.confidence === recommendation.matchConfidence
    && ["Narrow", "Moderate", "Wide"].includes(recommendation.demandUncertainty)
    && Number.isFinite(recommendation.uncertaintyRatio)
    && recommendation.salesLow <= recommendation.expectedSales
    && recommendation.expectedSales <= recommendation.salesHigh
    && recommendation.expectedSales % 25 === 0
  )));
  const imageBackedUpcoming = artifact.upcoming.filter(({ imageUrl }) => Boolean(imageUrl));
  assert.ok(imageBackedUpcoming.some(({ recommendation }) => !recommendation.noSuitableMatch));
  assert.ok(imageBackedUpcoming.every(({ matches }) => matches.every((match) => (
    match.fashionVisualScore !== null
    && match.dinoVisualScore !== null
    && match.colourVisualScore !== null
    && match.textureVisualScore !== null
  ))));
  assert.ok(imageBackedUpcoming.every(({ matches, recommendation }) => (
    recommendation.noSuitableMatch
    || matches[0].visualScore >= artifact.meta.model.minimumVisualScore
  )));
  assert.ok(artifact.upcoming.every(({ recommendation }) => (
    !recommendation.noSuitableMatch
    || recommendation.expectedSales === recommendation.regressionSales
  )));
  assert.ok(artifact.upcoming.some(({ recommendation }) => (
    recommendation.matchConfidence === "Medium"
    && recommendation.demandUncertainty === "Wide"
  )));
  assert.ok(artifact.upcoming.every(({ matches }) => matches.every(({ attributeBreakdown }) => (
    Object.keys(attributeBreakdown).length === 5
    && ["item", "design", "category_type", "fabric", "colour"]
      .every((key) => key in attributeBreakdown)
  ))));
  assert.ok(artifact.historical.every((item) => !("mrp" in item)));
  assert.ok(artifact.upcoming.every((item) => !("mrp" in item)));
  assert.match(pageSource, /Visual AI retrieval \+ trained unit-sales forecasting/);
  assert.match(pageSource, /FashionSigLIP retrieves the top/);
  assert.match(pageSource, /garment-masked CIELAB colour/);
  assert.match(pageSource, /Match confidence/);
  assert.match(pageSource, /Sales uncertainty/);
  assert.match(pageSource, /Top historical analogue/);
  assert.match(pageSource, /No convincing product match found/);
  assert.match(pageSource, /No historical product is shown/);
  assert.match(pageSource, /No product match/);
  assert.match(pageSource, /decision\.noSuitableMatch/);
  assert.match(pageSource, /Expected customer sales/);
  assert.match(pageSource, /Recommended initial order/);
  assert.match(pageSource, /Analogue sales forecast/);
  assert.match(pageSource, /Inventory strategy/);
  assert.match(pageSource, /historical\.salesTarget/);
  assert.match(pageSource, /MatchAttributeCatalog/);
  assert.match(pageSource, /Product attributes/);
  assert.match(pageSource, /\$\{context\} product attributes/);
  assert.doesNotMatch(pageSource, /Upcoming match attributes/);
  assert.doesNotMatch(pageSource, /Historical match attributes/);
  assert.match(pageSource, /View all \$\{catalogAttributeOrder\.length\}/);
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
  assert.match(pageSource, /image score combines FashionSigLIP, DINOv2, masked garment colour and texture evidence/i);
  assert.match(pageSource, /const visibleUpcoming =/);
  assert.match(pageSource, /\.filter\(\(match\) => Boolean\(historyById\.get\(match\.historicalId\)\?\.imageUrl\)\)/);
  assert.match(pageSource, /NEXT_PUBLIC_TURTLE_API_URL/);
  assert.doesNotMatch(pageSource, /FashionCLIP|scikit-learn|PyTorch|Ridge sales|CatBoost|LightGBM|pgvector|MinTrace/);
  assert.match(pageSource, /attributeValueReaders/);
  assert.match(pageSource, /catalogAttributeOrder/);
  assert.match(pageSource, /const preferredCatalogAttributeOrder = \[\s*"colour",\s*"design",\s*"fabric",\s*"item",\s*"category_type",/s);
  assert.doesNotMatch(pageSource, /\bMRP\b|\bmrp\b|Price band/);
  assert.doesNotMatch(pageSource, /\bsleeve\b|\bprovision\b/i);
  assert.match(pageSource, /const catalogAttributeOrder = preferredCatalogAttributeOrder\.filter/);
  assert.match(pageSource, /Category Type/);
  assert.match(pageSource, /Workbook attribute audit/);
  assert.match(pageSource, /informative fields retained/);
  assert.match(pageSource, /Excluded field/);
  assert.match(pageSource, /Exact match/);
  assert.match(pageSource, /Upcoming/);
  assert.match(pageSource, /Historical/);
  assert.doesNotMatch(pageSource, /product-details-grid/);
  assert.doesNotMatch(pageSource, /historical-spec-grid/);
  assert.doesNotMatch(pageSource, /View all product details/);
  assert.match(pageSource, /queue-commercial/);
  assert.match(stylesSource, /\.catalog-attribute-grid\s*\{[^}]*grid-template-columns:\s*repeat\(2,/s);
  assert.match(stylesSource, /\.match-grid\.match-grid-3\s*\{[^}]*repeat\(3,\s*max\(200px,\s*calc\(\(100% - 36px\) \/ 4\)\)\)/s);
  assert.match(stylesSource, /\.no-product-match-state\s*\{/);
  assert.match(stylesSource, /\.no-match-pill\s*\{/);
  assert.match(stylesSource, /\.match-grid\.match-grid-5\s*\{[^}]*repeat\(5,\s*max\(200px,\s*calc\(\(100% - 36px\) \/ 4\)\)\)/s);
  assert.match(stylesSource, /\.match-grid\.match-grid-8\s*\{[^}]*repeat\(8,\s*max\(200px,\s*calc\(\(100% - 36px\) \/ 4\)\)\)/s);
  assert.match(stylesSource, /\.match-grid\.match-grid-3,\s*\.match-grid\.match-grid-5,\s*\.match-grid\.match-grid-8\s*\{[^}]*column-gap:\s*12px[^}]*overflow-x:\s*auto[^}]*padding:\s*0 12px 6px/s);
  assert.match(stylesSource, /\.catalog-attribute-grid dd\s*\{[^}]*overflow-wrap:\s*break-word[^}]*word-break:\s*normal/s);
  assert.match(stylesSource, /\.catalog-attribute-toggle\s*\{[^}]*width:\s*auto/s);
  assert.match(stylesSource, /\.match-card-select\s*\{[^}]*width:\s*100%/s);
  assert.match(stylesSource, /\.match-card::after\s*\{[^}]*border:\s*1px solid transparent[^}]*inset:\s*0[^}]*pointer-events:\s*none[^}]*z-index:\s*3/s);
  assert.match(stylesSource, /\.match-card:hover::after,\s*\.match-card\.active::after\s*\{[^}]*border-color:\s*var\(--forest-2\)/s);
  assert.doesNotMatch(stylesSource, /\.match-card:hover,\s*\.match-card\.active\s*\{[^}]*transform:/s);
  assert.match(stylesSource, /\.match-card \.catalog-attribute-grid > div\s*\{[^}]*display:\s*flex[^}]*flex-direction:\s*column[^}]*min-height:\s*26px/s);
  assert.doesNotMatch(stylesSource, /\.match-card \.catalog-attribute-grid > div:nth-child/);
  assert.match(stylesSource, /\.match-performance\s*\{[^}]*gap:\s*4px[^}]*margin-top:\s*4px[^}]*padding-top:\s*4px/s);
  assert.match(stylesSource, /\.match-performance span\s*\{[^}]*align-items:\s*flex-start[^}]*flex-direction:\s*column[^}]*min-width:\s*0/s);
  assert.match(stylesSource, /\.match-performance small\s*\{[^}]*display:\s*block[^}]*white-space:\s*nowrap/s);
  assert.match(stylesSource, /\.match-performance strong\s*\{[^}]*font-size:\s*11px[^}]*font-weight:\s*900/s);
  assert.doesNotMatch(stylesSource, /\.match-card \.match-attribute-catalog-heading[^}]*font-size/s);
  assert.doesNotMatch(stylesSource, /\.match-card \.catalog-attribute-grid (?:dt|dd)\s*\{[^}]*font-size/s);
  assert.match(stylesSource, /\.match-card \.catalog-attribute-grid dd\s*\{[^}]*font-weight:\s*700[^}]*min-height:\s*0/s);
  assert.match(stylesSource, /\.match-card \.catalog-attribute-toggle\s*\{[^}]*min-height:\s*20px/s);
  assert.match(stylesSource, /\.evidence-header\s*\{[^}]*grid-template-columns:\s*minmax\(0, 1fr\) 230px/s);
  assert.match(stylesSource, /\.evidence-product-pair\s*\{[^}]*grid-template-columns:\s*minmax\(0, 1fr\) 28px minmax\(0, 1fr\)/s);
  assert.match(stylesSource, /\.attribute-evidence\s*\{[^}]*grid-template-columns:\s*repeat\(3,/s);
  assert.match(stylesSource, /\.hero-product-image\s*\{[^}]*aspect-ratio:\s*auto 3 \/ 4[^}]*border-radius:\s*13px[^}]*height:\s*auto[^}]*margin:\s*12px auto 0[^}]*max-width:\s*100%[^}]*object-fit:\s*contain[^}]*object-position:\s*center[^}]*width:\s*100%/s);
  assert.match(stylesSource, /\.image-fallback\.hero-product-image\s*\{[^}]*aspect-ratio:\s*3 \/ 4/s);
  assert.match(stylesSource, /\.match-image\s*\{[^}]*aspect-ratio:\s*auto 3 \/ 4[^}]*height:\s*auto[^}]*object-fit:\s*contain/s);
  assert.match(stylesSource, /\.image-fallback\.match-image\s*\{[^}]*aspect-ratio:\s*5 \/ 6/s);
  assert.match(stylesSource, /\.recommendation-metrics\s*\{[^}]*grid-template-columns:\s*repeat\(2,/s);
  assert.match(stylesSource, /\.quantity-secondary\s*\{[^}]*border-radius:\s*12px[^}]*flex-direction:\s*column/s);
  assert.match(stylesSource, /\.upcoming-card,\s*\.recommendation-card,\s*\.settings-card\s*\{[^}]*align-self:\s*stretch[^}]*box-sizing:\s*border-box/s);
  assert.match(stylesSource, /\.recommendation-metrics small,[^}]*overflow-wrap:\s*anywhere/s);
  assert.match(stylesSource, /\.queue-commercial\s*\{[^}]*justify-content:\s*space-between/s);
  assert.doesNotMatch(stylesSource, /\.product-details-grid\s*\{/);
  assert.doesNotMatch(stylesSource, /\.historical-spec-grid\s*\{/);
});
