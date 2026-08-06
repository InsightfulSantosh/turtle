import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const artifactUrl = new URL("../app/generated-data.json", import.meta.url);
const pageUrl = new URL("../app/page.tsx", import.meta.url);

test("keeps the visual-only single-analogue contract intact", async () => {
  const [artifactText, pageSource] = await Promise.all([
    readFile(artifactUrl, "utf8"),
    readFile(pageUrl, "utf8"),
  ]);
  const artifact = JSON.parse(artifactText);
  const expectedUpcoming = artifact.meta.previewSample ? 200 : 5550;
  const model = artifact.meta.model;
  const reranker = artifact.meta.visionModel.reranker;

  assert.equal(model.version, "5.1.0");
  assert.equal(model.evidencePolicy, "single_top_visual_analogue");
  assert.equal(model.noMachineLearningForecast, true);
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

  const historyById = new Map(artifact.historical.map((item) => [item.id, item]));
  assert.ok(artifact.upcoming.every(({ matches }) => matches.length <= 4));
  assert.ok(artifact.upcoming.every(({ matches }) => matches.every((match) => (
    !("attributeScore" in match) && !("attributeBreakdown" in match)
  ))));
  assert.ok(artifact.upcoming.every(({ recommendation, matches }) => {
    if (recommendation.noSuitableMatch) {
      return recommendation.quantity === 0
        && recommendation.expectedSales === 0;
    }
    const historical = historyById.get(matches[0].historicalId);
    const expectedSales = Math.min(Math.round(historical.salesTarget / 25) * 25, 2000);
    const expectedOrder = Math.min(Math.round(expectedSales / 0.70 / 25) * 25, 2000);
    return Math.round(matches[0].visualScore * 100) >= Math.round(model.minimumVisualScore * 100)
      && recommendation.evidencePolicy === "single_top_visual_analogue"
      && recommendation.expectedSales === expectedSales
      && recommendation.quantity === expectedOrder;
  }));

  assert.match(pageSource, /Minimum visual similarity/);
  assert.match(pageSource, /Target sell-through/);
  assert.match(pageSource, /Showing the top/);
  assert.match(pageSource, /Ranked by how closely the product photos match/);
  assert.match(pageSource, /the recommended order is/);
  assert.match(pageSource, /No candidate meets the/);
  assert.match(pageSource, /const eligible = ranked\.filter/);
  assert.doesNotMatch(pageSource, /Attribute weight|Inventory strategy|Machine-learning sales forecast|Sales backtest WAPE/);
  assert.doesNotMatch(pageSource, /colourFamily|Colour family/);
});
