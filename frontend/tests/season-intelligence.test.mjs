import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const artifactUrl = new URL("../app/generated-data.json", import.meta.url);
const pageUrl = new URL("../app/page.tsx", import.meta.url);

test("keeps the pooled predictive forecast contract intact", async () => {
  const [artifactText, pageSource] = await Promise.all([
    readFile(artifactUrl, "utf8"),
    readFile(pageUrl, "utf8"),
  ]);
  const artifact = JSON.parse(artifactText);
  const expectedUpcoming = artifact.meta.previewSample ? 200 : 5550;
  const model = artifact.meta.model;
  const reranker = artifact.meta.visionModel.reranker;

  assert.equal(model.version, "5.1.0");
  assert.equal(model.evidencePolicy, "pooled_visual_analogue_forecast");
  assert.equal(model.noMachineLearningForecast, false);
  assert.equal(model.noAttributeMatching, true);
  assert.equal(model.visualOnlyRanking, true);
  assert.equal(model.topK, 4);
  assert.equal(model.targetSellThrough, 0.70);
  assert.equal(model.minimumVisualScore, 0.5);
  assert.ok(!("demandLibrary" in model));
  assert.ok(!("demandPipeline" in model));
  assert.ok(!("regressionBlend" in model));
  assert.ok(!("attributeWeight" in model));
  assert.ok(!("backtest" in model));

  assert.equal(reranker.sameItemTypeConstraint, true);
  assert.equal(reranker.sameDesignConstraint, false);
  assert.ok(!("sameColourFamilyConstraint" in reranker));
  assert.equal(reranker.patternGate.enabled, true);
  assert.deepEqual(
    reranker.appearance.itemTypeOverrides.OTTR.relativeBox,
    [0.16, 0.28, 0.84, 0.8],
  );
  assert.equal(artifact.upcoming.length, expectedUpcoming);
  assert.equal(artifact.meta.historicalImageCoverage, 508);
  assert.equal(artifact.meta.upcomingImageCoverage, expectedUpcoming);

  // The estimator must ship the fitted priors and per-product rates, because
  // the frontend repools the forecast whenever a planner moves a slider.
  assert.ok(model.demandModel, "artifact must publish the fitted demand model");
  assert.ok(model.demandModel.horizonWeeks > 0);
  assert.ok(model.demandModel.shrinkageTau > 0);
  assert.ok(model.demandModel.wideUncertaintyEffectiveN > 0);
  assert.ok(model.demandModel.wideUncertaintySkewRatio > 1);
  assert.ok(Object.keys(model.demandModel.groups).length > 1);
  assert.ok(model.demandModel.groups[""], "a global prior must always exist");
  assert.ok(
    artifact.historical.filter((item) => typeof item.weeklyLogRate === "number").length > 0,
    "historical products must carry a corrected weekly demand rate",
  );
  // A handful of source rows genuinely record no ageing and fall back to the
  // assumed window. Near-total coverage is what distinguishes a properly built
  // artifact from a pre-migration one where the fallback applied to everything.
  const withExposure = artifact.historical.filter((item) => item.ageingDays > 0).length;
  assert.ok(
    withExposure / artifact.historical.length > 0.95,
    `only ${withExposure}/${artifact.historical.length} products carry exposure data`,
  );

  const historyById = new Map(artifact.historical.map((item) => [item.id, item]));
  assert.ok(artifact.upcoming.every(({ matches }) => matches.length <= 4));
  assert.ok(artifact.upcoming.every(({ matches }) => matches.every((match) => (
    !("attributeScore" in match) && !("attributeBreakdown" in match)
  ))));

  const recommended = artifact.upcoming.filter(({ recommendation }) => !recommendation.noSuitableMatch);
  assert.ok(recommended.length > 0);
  assert.ok(artifact.upcoming.every(({ recommendation, matches }) => {
    if (recommendation.noSuitableMatch) {
      return recommendation.quantity === 0
        && recommendation.expectedSales === 0
        && !("demand" in recommendation);
    }
    return Math.round(matches[0].visualScore * 100) >= Math.round(model.minimumVisualScore * 100)
      && recommendation.evidencePolicy === "pooled_visual_analogue_forecast"
      && recommendation.demand.analoguesUsed >= 1
      // A forecast has a distribution; a lookup has one value. The bands are
      // strict unless the 2,000-unit buy cap clamps the upper end.
      && recommendation.salesLow < recommendation.salesHigh
      && recommendation.low < recommendation.high
      && recommendation.salesLow <= recommendation.expectedSales
      && recommendation.expectedSales <= recommendation.salesHigh
      && recommendation.low <= recommendation.quantity
      && recommendation.quantity <= recommendation.high;
  }));

  // The reported defect: the forecast was the analogue's own sales republished
  // under a second name, for every single product.
  const copies = recommended.filter(
    ({ recommendation }) => recommendation.expectedSales === recommendation.analogueSales,
  );
  assert.ok(
    copies.length / recommended.length < 0.25,
    `forecast still copies the analogue for ${copies.length}/${recommended.length} products`,
  );
  assert.ok(recommended.every(({ matches, recommendation }) => {
    const historical = historyById.get(matches[0].historicalId);
    return recommendation.analogueSales === Math.min(Math.round(historical.salesTarget / 25) * 25, 2000);
  }), "analogue sales must still report the selected analogue's own cleaned sales");

  assert.match(pageSource, /Minimum visual similarity/);
  assert.match(pageSource, /Target sell-through/);
  assert.match(pageSource, /Ranked by how closely the product photos match/);
  assert.match(pageSource, /No candidate meets the/);
  assert.match(pageSource, /const eligible = ranked\.filter/);
  assert.match(pageSource, /function newsvendorOrder/);
  assert.match(pageSource, /function forecastDemand/);
  assert.doesNotMatch(pageSource, /Attribute weight|Inventory strategy|Machine-learning sales forecast|Sales backtest WAPE/);
  assert.doesNotMatch(pageSource, /colourFamily|Colour family/);
});
