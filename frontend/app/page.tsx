"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type Confidence = "High" | "Medium" | "Low";
type Tab = "compare" | "portfolio" | "upload";

type HistoricalItem = {
  id: string;
  season: string;
  itemType: string;
  design: string;
  categoryType: string;
  fabric: string;
  colour: string;
  order: number;
  sellThrough: number;
  imageUrl?: string | null;
  salesTarget: number;
  qualityFlags: string[];
  /** Log of the censoring-corrected weekly demand rate; absent on legacy artifacts. */
  weeklyLogRate?: number | null;
};

type DemandPrior = {
  mu: number;
  sigma: number;
  rows: number;
};

type DemandModel = {
  horizonWeeks: number;
  shrinkageTau: number;
  similarityExponent: number;
  minimumLogSigma: number;
  maximumLogSigma: number;
  maximumAnalogues: number;
  wideUncertaintyEffectiveN: number;
  wideUncertaintySkewRatio: number;
  /** Scales the lognormal skew correction for the point forecast: 1.0 is the
   * full mean, 0.0 the median. See DemandPolicy.point_estimate_skew. */
  pointEstimateSkew: number;
  groups: Record<string, DemandPrior>;
};

/** Data-derived, per-item-type sanity cap — see BuyCeilings in demand.py for
 * why a flat constant cannot be right for every item type (this catalogue's
 * own historical maxima span ~44x, from ~150 to ~6,700 units). */
type BuyCeilings = {
  byItemType: Record<string, number>;
  globalCeiling: number;
};

type Match = {
  historicalId: string;
  visualScore: number | null;
  fashionVisualScore: number | null;
  dinoVisualScore: number | null;
  colourVisualScore: number | null;
  textureVisualScore: number | null;
};

type UpcomingItem = {
  id: string;
  itemType: string;
  design: string;
  categoryType: string;
  fabric: string;
  colour: string;
  imageUrl?: string | null;
  matches: Match[];
  /** Only the backend's own match confidence is read, and only to pick the
   * product the workspace opens on: every shipped number is re-solved
   * client-side so the planner's sliders stay live. */
  recommendation: { matchConfidence: Confidence };
};

type ComparableProduct = Pick<UpcomingItem,
  "itemType" | "design" | "categoryType" | "fabric" | "colour"
>;

/**
 * The slice of the published artifact this page actually reads. The artifact
 * carries considerably more (provenance, calibration, preprocessing and quality
 * blocks) — that contract is asserted by tests/season-intelligence.test.mjs and
 * documented in the README, not restated here as types nothing consumes.
 */
type Dataset = {
  meta: {
    upcomingSeason: string;
    upcomingItems: number;
    upcomingImageCoverage: number;
    missingUpcomingImages: string[];
    visionModel: {
      historicalCoverage: number;
      upcomingCoverage: number;
    };
    model: {
      minimumVisualScore: number;
      targetSellThrough?: number;
      /** Present only when the artifact was built with the predictive estimator. */
      demandModel?: DemandModel;
      buyCeilings?: BuyCeilings;
    };
  };
  historical: HistoricalItem[];
  upcoming: UpcomingItem[];
};

type RankedMatch = Match & { combinedScore: number };
type Decision = {
  ranked: RankedMatch[];
  eligible: RankedMatch[];
  selectedMatch?: RankedMatch;
  quantity: number;
  low: number;
  high: number;
  /** True when the buy ceiling truncated the model's own solve — a policy
   * limit, not evidence the target sell-through was actually reached. */
  quantityCapped: boolean;
  highCapped: boolean;
  buyCeiling: number;
  matchConfidence: Confidence;
  noSuitableMatch: boolean;
  expectedSales: number;
  salesLow: number;
  salesHigh: number;
  analogueSales: number;
  analogueQuantity: number;
  /** Null when the legacy copy-one-analogue rule produced the numbers. */
  forecast: DemandForecast | null;
};

/**
 * The workspace ships with no catalogue of its own: every number it shows comes
 * from a build the user uploaded through the New analysis tab. Until one is
 * activated the planner runs on this empty dataset and renders its empty state.
 */
const EMPTY_DATASET: Dataset = {
  meta: {
    upcomingSeason: "—",
    upcomingItems: 0,
    upcomingImageCoverage: 0,
    missingUpcomingImages: [],
    visionModel: { historicalCoverage: 0, upcomingCoverage: 0 },
    model: { minimumVisualScore: 0.5, targetSellThrough: 0.7 },
  },
  historical: [],
  upcoming: [],
};

/** Stands in for the selected product while no build is active, so the
 * workspace's hooks stay unconditional and its empty state is a render guard
 * rather than an early return. */
const PLACEHOLDER_UPCOMING: UpcomingItem = {
  id: "",
  itemType: "",
  design: "",
  categoryType: "",
  fabric: "",
  colour: "",
  imageUrl: null,
  matches: [],
  recommendation: { matchConfidence: "Low" },
};

let dataset = EMPTY_DATASET;
let historyById = new Map<string, HistoricalItem>();
let visibleUpcoming: UpcomingItem[] = [];
let productSegments: string[] = [];
let visualMatchingAvailable = false;
let buyCeilings: BuyCeilings | undefined;
let demandModel: DemandModel | undefined;

function installDataset(next: Dataset) {
  dataset = next;
  historyById = new Map(dataset.historical.map((item) => [item.id, item]));
  const imageBackedUpcoming = dataset.upcoming.filter((item) => Boolean(item.imageUrl));
  visibleUpcoming = imageBackedUpcoming.length > 0 ? imageBackedUpcoming : dataset.upcoming;
  productSegments = Array.from(new Set(visibleUpcoming.map((item) => item.itemType))).sort();
  visualMatchingAvailable =
    dataset.meta.visionModel.upcomingCoverage > 0 &&
    dataset.meta.visionModel.historicalCoverage > 0;
  buyCeilings = dataset.meta.model.buyCeilings;
  // Read through the live dataset rather than captured once at module load:
  // every activated build brings its own priors.
  demandModel = dataset.meta.model.demandModel;
}

installDataset(dataset);
const numberFormatter = new Intl.NumberFormat("en-IN");

const attributeValueReaders: Record<string, (item: ComparableProduct) => string> = {
  design: (product) => product.design,
  category_type: (product) => product.categoryType,
  fabric: (item) => item.fabric,
  colour: (item) => item.colour,
};

const attributeLabels: Record<string, string> = {
  design: "Design",
  category_type: "Category Type",
  fabric: "Fabric",
  colour: "Colour",
};

const catalogAttributeOrder = [
  "colour",
  "design",
  "fabric",
  "category_type",
] as const;

function attributeValue(item: ComparableProduct, key: string) {
  return attributeValueReaders[key]?.(item) || "Not provided";
}

function normalizeAttributeValue(value: string) {
  return value.trim().toLowerCase();
}

function attributeMatchStatus(upcomingValue: string, historicalValue: string): "exact" | "different" {
  return normalizeAttributeValue(upcomingValue) === normalizeAttributeValue(historicalValue)
    ? "exact"
    : "different";
}

function scorePercent(value: number | null) {
  return value === null ? "N/A" : `${Math.round(value * 100)}%`;
}

function sellThroughTradeoff(target: number): string {
  const percent = Math.round(target * 100);
  if (target <= 0.5) {
    return `A ${percent}% sell-through target favors a larger opening order to capture more demand, with higher leftover-stock risk.`;
  }
  if (target >= 0.75) {
    return `A ${percent}% sell-through target favors a smaller opening order and lower leftover-stock risk, with a greater chance of missed sales.`;
  }
  return `A ${percent}% sell-through target balances capturing demand with limiting leftover stock.`;
}

// Fallback only, for callers with no data-derived ceiling available (the
// legacy-formula edge case before any artifact ever carried buyCeilings).
// The live predictive path uses a per-item-type ceiling from
// dataset.meta.model.buyCeilings instead — see the BuyCeilings type comment
// for why a flat number cannot be right for every item type.
const FALLBACK_MAX_BUY = 2_000;

/** Pack-round without any cap, so callers can tell whether one bound. */
function packRoundedRaw(value: number) {
  return Math.round(value / 25) * 25;
}

/** Historical evidence is an observed fact, not a forecast — never capped. */
function packRoundedUncapped(value: number) {
  return Math.max(0, packRoundedRaw(value));
}

function packRoundedCapped(value: number, ceiling: number) {
  return Math.max(0, Math.min(ceiling, packRoundedRaw(value)));
}

function ceilingFor(item: ComparableProduct): number {
  if (!buyCeilings) return FALLBACK_MAX_BUY;
  // Rounded once here (matching model.py) so a float ceiling that isn't
  // perfectly integral can't disagree with itself between the clamp and any
  // reported value derived from it.
  return Math.round(buyCeilings.byItemType[item.itemType] ?? buyCeilings.globalCeiling);
}

/**
 * Predicted demand for a product, independent of what is bought. For a
 * predictive forecast this is the median (the typical, most likely outcome)
 * rather than the mean — showing both side by side, as an earlier version of
 * this card did — and an earlier version of the CSV alongside it — reads as
 * two conflicting answers to the same question. The mean stays on the
 * `Decision` as `expectedSales` (it's the statistically correct number to sum
 * across many products; the median is not), but neither the card nor the
 * export shows both: one product needs one number, and the one a human means
 * by "how many will sell" is the typical case, not an average pulled up by
 * tail risk.
 *
 * This is *demand*. Anything labelled as sales must go through
 * `salesFromOrder` instead.
 */
function headlineDemand(decision: Decision): number {
  return decision.forecast?.medianDemand ?? decision.expectedSales;
}

/**
 * What the planner actually sells from a given buy. Sales are min(demand,
 * order) — you cannot sell stock you did not buy — so a raise of the
 * sell-through target, which shrinks the order, must pull the sales figure
 * down with it. Reporting unclipped demand here let the card state impossible
 * outcomes: a 300-unit buy beside a 575-unit "sales forecast", a 192%
 * sell-through.
 *
 * Because min(·, order) is monotone, every quantile passes straight through
 * it, so clipping the median and both interval bounds at the order quantity is
 * exact rather than an approximation.
 *
 * Only the predictive path is clipped. On the legacy path these numbers are
 * the matched analogue's own observed sales — a historical fact about a
 * different product, which this buy has no business bounding.
 */
function salesFromOrder(decision: Decision, orderQuantity: number) {
  const demand = headlineDemand(decision);
  if (!decision.forecast) {
    return {
      headline: demand,
      low: decision.salesLow,
      high: decision.salesHigh,
      demandExceedsOrder: false,
    };
  }
  const clip = (value: number) => Math.min(value, orderQuantity);
  return {
    headline: clip(demand),
    low: clip(decision.salesLow),
    high: clip(decision.salesHigh),
    // Clipping alone would silently hide an undersized buy, so the card flags
    // when the order — not demand — is what limits sales.
    demandExceedsOrder: demand > orderQuantity,
  };
}

const P10_Z = 1.2815515655446004;

// Abramowitz & Stegun 7.1.26. The estimator only needs CDF accuracy to ~1e-7,
// which is far finer than a buy rounded to packs of 25.
function erf(value: number) {
  const sign = value < 0 ? -1 : 1;
  const x = Math.abs(value);
  const t = 1 / (1 + 0.3275911 * x);
  const y =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) *
      t *
      Math.exp(-x * x);
  return sign * y;
}

function normalCdf(value: number) {
  return 0.5 * (1 + erf(value / Math.SQRT2));
}

/** E[min(D, Q)] for lognormal demand: expected sales saturate above demand. */
function expectedSalesAt(orderQuantity: number, logMu: number, logSigma: number) {
  if (orderQuantity <= 0) return 0;
  if (logSigma <= 1e-9) return Math.min(Math.exp(logMu), orderQuantity);
  const z = (Math.log(orderQuantity) - logMu) / logSigma;
  const mean = Math.exp(logMu + (logSigma * logSigma) / 2);
  return mean * normalCdf(z - logSigma) + orderQuantity * (1 - normalCdf(z));
}

/**
 * The buy whose expected sell-through equals the planner target.
 * With logSigma = 0 this returns exactly demand / target, so the legacy rule
 * is this function's no-uncertainty special case.
 */
function newsvendorOrder(logMu: number, logSigma: number, targetSellThrough: number) {
  const target = Math.max(0.01, Math.min(0.99, targetSellThrough));
  let low = 1e-6;
  let high = Math.max(Math.exp(logMu + (logSigma * logSigma) / 2) / target, 1);
  for (let step = 0; step < 60; step += 1) {
    if (expectedSalesAt(high, logMu, logSigma) / high <= target) break;
    high *= 2;
  }
  for (let step = 0; step < 200; step += 1) {
    const middle = (low + high) / 2;
    if (expectedSalesAt(middle, logMu, logSigma) / middle > target) low = middle;
    else high = middle;
  }
  return (low + high) / 2;
}

function priorFor(item: ComparableProduct): DemandPrior | undefined {
  if (!demandModel) return undefined;
  // No row-count gate here: each group in demandModel.groups is already a
  // hierarchically shrunk prior (category blended toward item-type, blended
  // toward global — see machine_learning/demand.py), so whichever level is
  // present is safe to use directly.
  const keys = [`${item.itemType}|${item.categoryType}`, item.itemType, ""];
  for (const key of keys) {
    const prior = demandModel.groups[key];
    if (prior) return prior;
  }
  return demandModel.groups[""];
}

type DemandForecast = {
  expectedSales: number;
  medianDemand: number;
  wideUncertainty: boolean;
  salesLow: number;
  salesHigh: number;
  quantity: number;
  low: number;
  high: number;
  /** True when the model's own solve wanted more than the buy ceiling and
   * was truncated — a policy limit, not evidence the target was reached. */
  quantityCapped: boolean;
  highCapped: boolean;
  buyCeiling: number;
  analoguesUsed: number;
};

/**
 * Pool the accepted analogues into a shrunk predictive distribution.
 * Mirrors machine_learning/demand.py so the planner's sliders keep re-solving
 * the estimator rather than silently reverting to the legacy division.
 */
function forecastDemand(
  item: UpcomingItem,
  accepted: RankedMatch[],
  targetSellThrough: number,
): DemandForecast | null {
  if (!demandModel) return null;
  const prior = priorFor(item);
  if (!prior) return null;

  const weights: number[] = [];
  const logRates: number[] = [];
  for (const match of accepted.slice(0, demandModel.maximumAnalogues)) {
    const historical = historyById.get(match.historicalId);
    const logRate = historical?.weeklyLogRate;
    if (logRate === null || logRate === undefined) continue;
    weights.push(Math.pow(Math.max(match.combinedScore, 1e-6), demandModel.similarityExponent));
    logRates.push(logRate);
  }
  if (!logRates.length) return null;

  const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);
  const weightedMean = logRates.reduce((sum, rate, index) => sum + weights[index] * rate, 0) / totalWeight;
  // Kish effective sample size: four near-identical analogues carry more
  // evidence than four where one dominates the weighting.
  const effectiveN = (totalWeight * totalWeight) / weights.reduce((sum, weight) => sum + weight * weight, 0);
  const observedVariance =
    logRates.length > 1
      ? logRates.reduce((sum, rate, index) => sum + weights[index] * (rate - weightedMean) ** 2, 0) / totalWeight
      : prior.sigma * prior.sigma;

  const shrinkage = effectiveN / (effectiveN + demandModel.shrinkageTau);
  const logMuRate = shrinkage * weightedMean + (1 - shrinkage) * prior.mu;
  const coreVariance =
    (effectiveN * observedVariance + demandModel.shrinkageTau * prior.sigma * prior.sigma) /
    (effectiveN + demandModel.shrinkageTau);
  // A single analogue is not certainty: carrying the estimation variance of
  // the pooled mean is what stops one row from implying a point forecast.
  const logSigma = Math.max(
    demandModel.minimumLogSigma,
    Math.min(
      demandModel.maximumLogSigma,
      Math.sqrt(coreVariance * (1 + 1 / (effectiveN + demandModel.shrinkageTau))),
    ),
  );

  const logMu = logMuRate + Math.log(demandModel.horizonWeeks);
  const p10 = Math.exp(logMu - P10_Z * logSigma);
  const p90 = Math.exp(logMu + P10_Z * logSigma);
  const medianDemand = Math.exp(logMu);
  const meanDemand = Math.exp(logMu + (logSigma * logSigma) / 2);
  // The shipped point forecast is a *skew-corrected* mean, not the raw one:
  // sigma is inflated for honest interval coverage, and reusing it at full
  // strength for a point estimate overshoots. Mirrors demand.py exactly so a
  // slider move can't drift away from the backend's own numbers.
  const pointDemand = Math.exp(logMu + (demandModel.pointEstimateSkew * logSigma * logSigma) / 2);
  // mean/median = exp(logSigma^2/2) for a lognormal: this ratio grows purely
  // from uncertainty. When it (or the effective analogue count) crosses the
  // policy threshold, expectedSales (the mean, kept as the unbiased point
  // estimate the order is solved against) can sit far above the typical
  // outcome — flag it rather than showing both numbers with equal confidence.
  const skewRatio = meanDemand / Math.max(medianDemand, 1e-6);
  const wideUncertainty =
    effectiveN < demandModel.wideUncertaintyEffectiveN || skewRatio >= demandModel.wideUncertaintySkewRatio;
  const ceiling = ceilingFor(item);
  const rawQuantity = packRoundedRaw(newsvendorOrder(logMu, logSigma, targetSellThrough));
  const rawHigh = packRoundedRaw(newsvendorOrder(Math.log(Math.max(p90, 1e-6)), logSigma, targetSellThrough));
  return {
    expectedSales: packRoundedCapped(pointDemand, ceiling),
    medianDemand: packRoundedCapped(medianDemand, ceiling),
    wideUncertainty,
    salesLow: packRoundedCapped(p10, ceiling),
    salesHigh: packRoundedCapped(p90, ceiling),
    quantity: Math.max(0, Math.min(ceiling, rawQuantity)),
    low: packRoundedCapped(newsvendorOrder(Math.log(Math.max(p10, 1e-6)), logSigma, targetSellThrough), ceiling),
    high: Math.max(0, Math.min(ceiling, rawHigh)),
    quantityCapped: rawQuantity > ceiling,
    highCapped: rawHigh > ceiling,
    buyCeiling: ceiling,
    analoguesUsed: logRates.length,
  };
}

function makeDecision(
  item: UpcomingItem,
  minimumSimilarity = dataset.meta.model.minimumVisualScore,
  targetSellThrough = dataset.meta.model.targetSellThrough ?? 0.70,
  selectedHistoricalId?: string | null,
): Decision {
  const ranked = item.matches
    .filter((match) => {
      const historical = historyById.get(match.historicalId);
      if (!historical?.imageUrl) return false;
      return historical.itemType === item.itemType;
    })
    .map((match) => ({ ...match, combinedScore: match.visualScore ?? 0 }))
    .sort((left, right) => right.combinedScore - left.combinedScore);
  const displayedCriterion = Math.round(minimumSimilarity * 100);
  const eligible = ranked.filter(
    (match) => Math.round(match.combinedScore * 100) >= displayedCriterion,
  );
  const selectedMatch = eligible.find((match) => match.historicalId === selectedHistoricalId) ?? eligible[0];
  const topScore = selectedMatch?.combinedScore ?? 0;
  const displayedScore = Math.round(topScore * 100);
  const historical = selectedMatch ? historyById.get(selectedMatch.historicalId) : undefined;
  let matchConfidence: Confidence = "Low";
  if (topScore >= 0.84 && selectedMatch?.visualScore !== null && !(historical?.qualityFlags.length)) {
    matchConfidence = "High";
  } else if (displayedScore >= displayedCriterion && selectedMatch?.visualScore !== null) {
    matchConfidence = "Medium";
  }
  const topVisualScore = selectedMatch?.visualScore;
  const noSuitableMatch =
    !selectedMatch ||
    matchConfidence === "Low" ||
    topVisualScore === null ||
    topVisualScore === undefined ||
    displayedScore < displayedCriterion;
  // Historical evidence is an observed fact, not a forecast — never capped
  // (this catalogue's top sellers exceed 5,000 units, well past any of the
  // per-item-type ceilings below, and truncating a real historical number
  // would misrepresent what actually happened).
  const analogueSales = noSuitableMatch || !historical ? 0 : packRoundedUncapped(historical.salesTarget);
  // The forecast pools every accepted analogue. Selecting one analogue changes
  // which product is shown as evidence, not which single row the buy copies.
  const forecast = noSuitableMatch ? null : forecastDemand(item, eligible, targetSellThrough);
  const ceiling = ceilingFor(item);
  const legacyRawQuantity = noSuitableMatch
    ? 0
    : packRoundedRaw(analogueSales / Math.max(targetSellThrough, 0.01));
  const legacyQuantity = Math.max(0, Math.min(ceiling, legacyRawQuantity));
  const legacyCapped = legacyRawQuantity > ceiling;
  return {
    ranked,
    eligible,
    selectedMatch,
    quantity: forecast ? forecast.quantity : legacyQuantity,
    low: forecast ? forecast.low : legacyQuantity,
    high: forecast ? forecast.high : legacyQuantity,
    quantityCapped: forecast ? forecast.quantityCapped : legacyCapped,
    highCapped: forecast ? forecast.highCapped : legacyCapped,
    buyCeiling: forecast ? forecast.buyCeiling : ceiling,
    matchConfidence,
    noSuitableMatch,
    expectedSales: forecast ? forecast.expectedSales : analogueSales,
    salesLow: forecast ? forecast.salesLow : analogueSales,
    salesHigh: forecast ? forecast.salesHigh : analogueSales,
    analogueSales,
    analogueQuantity: noSuitableMatch || !historical ? 0 : packRoundedUncapped(historical.order),
    forecast,
  };
}

function ProductImage({
  src,
  alt,
  className = "",
  eager = false,
}: {
  src?: string | null;
  alt: string;
  className?: string;
  eager?: boolean;
}) {
  const [failedSrc, setFailedSrc] = useState<string | null>(null);
  // Every imageUrl the artifact publishes is an /api/builds/{id}/images/ path,
  // proxied to the analysis service, so it is already same-origin here.
  const failed = Boolean(src && failedSrc === src);

  if (!src || failed) {
    return (
      <div className={`image-fallback ${className}`} aria-label={`${alt}: image unavailable`}>
        <span>Image pending</span>
        <small>Attribute-only match</small>
      </div>
    );
  }

  return (
    // Product images are remote, dynamic catalogue URLs and cannot use a fixed Next image allowlist.
    // eslint-disable-next-line @next/next/no-img-element
    <img
      className={className}
      src={src}
      alt={alt}
      loading={eager ? "eager" : "lazy"}
      referrerPolicy="no-referrer"
      onError={() => setFailedSrc(src)}
    />
  );
}

function MatchAttributeCatalog({
  product,
  context,
}: {
  product: ComparableProduct;
  context: "Upcoming" | "Historical";
}) {
  return (
    <section className="match-attribute-catalog" aria-label={`${context} product attributes`}>
      <div className="match-attribute-catalog-heading">
        <small>Product attributes</small>
      </div>
      <dl className="catalog-attribute-grid">
        {catalogAttributeOrder.map((key) => {
          const value = attributeValue(product, key);
          return (
            <div key={key}>
              <dt>{attributeLabels[key]}</dt>
              <dd title={value}>{value}</dd>
            </div>
          );
        })}
      </dl>
    </section>
  );
}

function MatchConfidencePill({
  confidence,
  detailed = false,
}: {
  confidence: Confidence;
  detailed?: boolean;
}) {
  return (
    <span className={`confidence-pill ${confidence.toLowerCase()}`}>
      {confidence} {detailed ? "match confidence" : "match"}
    </span>
  );
}

function NoMatchPill() {
  return (
    <span className="no-match-pill">
      No convincing visual match
    </span>
  );
}

function ScoreRing({ score, label }: { score: number; label: string }) {
  const degrees = Math.round(score * 360);
  return (
    <div className="score-ring-wrap">
      <div
        className="score-ring"
        style={{ background: `conic-gradient(var(--lime) ${degrees}deg, var(--line) 0deg)` }}
        aria-label={`${label}: ${scorePercent(score)}`}
      >
        <span>{scorePercent(score)}</span>
      </div>
      <small>{label}</small>
    </div>
  );
}

type HistoricalSummary = {
  id: string;
  createdAt: string;
  productCount: number;
  imageCoverage: number;
  modelVersion: string;
};

type AnalysisRun = {
  id: string;
  mode: "full_replace" | "reuse_historical";
  status: "uploading" | "queued" | "processing" | "succeeded" | "failed" | "cancelled";
  stage: string;
  progress: number;
  message: string;
  error?: string | null;
  buildId?: string | null;
  processedCount?: number;
  totalCount?: number;
  cacheHits?: number;
  etaSeconds?: number | null;
  startedAt?: string | null;
  completedAt?: string | null;
  updatedAt?: string;
  /** Per-catalogue encode counters: a single blended total cannot show which
   * half of the build is running, and the two phases are very different sizes. */
  historicalProcessed?: number;
  historicalTotal?: number;
  upcomingProcessed?: number;
  upcomingTotal?: number;
  validationReportUrl?: string | null;
};

const stageLabels: Record<string, string> = {
  uploading: "Uploading files",
  queued: "Waiting for an analysis worker",
  validating: "Validating catalogues and images",
  indexing_history: "Encoding historical images",
  matching_upcoming: "Encoding upcoming images and matching",
  building_results: "Building recommendations",
  succeeded: "Recommendations ready",
  failed: "Analysis failed",
  cancelled: "Analysis cancelled",
};

function stageLabel(run: AnalysisRun) {
  return stageLabels[run.stage] ?? (run.message || run.stage);
}

function formatElapsedDuration(seconds: number) {
  const wholeSeconds = Math.max(0, Math.floor(seconds));
  if (wholeSeconds < 60) return `${wholeSeconds}s`;
  const minutes = Math.floor(wholeSeconds / 60);
  const remainder = wholeSeconds % 60;
  if (minutes < 60) return remainder ? `${minutes}m ${remainder}s` : `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  return remainingMinutes ? `${hours}h ${remainingMinutes}m` : `${hours}h`;
}

function analysisElapsedSeconds(run: AnalysisRun, now: number) {
  if (!run.startedAt) return null;
  const start = Date.parse(run.startedAt);
  const terminal = run.status === "succeeded" || run.status === "failed" || run.status === "cancelled";
  const endValue = run.completedAt ?? (terminal ? run.updatedAt : undefined);
  const end = endValue ? Date.parse(endValue) : now;
  if (!Number.isFinite(start) || !Number.isFinite(end)) return null;
  return Math.max(0, (end - start) / 1000);
}

const analysisDateFormatter = new Intl.DateTimeFormat("en-IN", {
  day: "numeric",
  month: "short",
  year: "numeric",
  hour: "numeric",
  minute: "2-digit",
  timeZone: "Asia/Kolkata",
});

function formatAnalysisDate(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "Available" : analysisDateFormatter.format(date);
}

type UploadEntry = {
  catalog: "historical" | "upcoming";
  kind: "catalogue" | "images";
  file: File;
};

const catalogueAccept = ".csv,.xlsx,.xlsm,.xls,.xlsb,.ods";
const DISPLAY_MODEL_VERSION = "1.0";

function formatBytes(value: number) {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

/**
 * The planner has no data of its own, so a failed API call is not a detail —
 * it is the whole screen. The Next.js rewrite reports an unreachable analysis
 * service as a bare `500 Internal Server Error` with no body, which says
 * nothing about the actual cause, so name it here.
 */
function describeServiceFailure(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  if (/\b(500|502|503|504)\b/.test(message) || /failed to fetch|networkerror|load failed/i.test(message)) {
    return "Cannot reach the analysis service. Start it with `make api-dev` and reload.";
  }
  return message;
}

async function responseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.json().catch(() => null) as { detail?: string } | null;
    throw new Error(body?.detail || `${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

const UPLOAD_ATTEMPTS = 4;
const ACTIVE_ANALYSIS_RUN_KEY = "turtle.active-analysis-run";
const RUN_POLL_INTERVAL_MS = 2500;

async function uploadAttempt(
  runId: string,
  entry: UploadEntry,
  signal: AbortSignal,
  onBytes: (value: number) => void,
) {
  const filename = encodeURIComponent(entry.file.name);
  const url = `/api/runs/${runId}/uploads/${entry.catalog}/${entry.kind}/${filename}`;
  const head = await fetch(url, { method: "HEAD", signal });
  if (!head.ok) throw new Error(`${entry.file.name}: ${head.status} ${head.statusText}`);
  const offset = Number(head.headers.get("Upload-Offset") ?? 0);
  if (offset > entry.file.size) throw new Error(`${entry.file.name}: remote upload is larger than the local file`);
  if (offset === entry.file.size) return;
  const response = await fetch(url, {
    method: "PUT",
    headers: {
      "Content-Type": entry.file.type || "application/octet-stream",
      "Upload-Offset": String(offset),
      "Upload-Length": String(entry.file.size),
    },
    body: entry.file.slice(offset),
    signal,
  });
  if (!response.ok) throw new Error(`${entry.file.name}: ${response.status} ${response.statusText}`);
  onBytes(entry.file.size - offset);
}

/**
 * A dropped connection part-way through a large image used to fail the whole
 * run — hundreds of already-uploaded megabytes thrown away because one body
 * did not finish. The protocol is resumable by design, so retry instead: each
 * attempt re-reads the server's offset and sends only the remainder, and the
 * per-attempt byte counts still add up to exactly one file.
 */
async function uploadOne(
  runId: string,
  entry: UploadEntry,
  signal: AbortSignal,
  onBytes: (value: number) => void,
) {
  for (let attempt = 1; ; attempt += 1) {
    try {
      await uploadAttempt(runId, entry, signal, onBytes);
      return;
    } catch (error) {
      // A user cancellation must stay immediate, never retried.
      if (signal.aborted || attempt >= UPLOAD_ATTEMPTS) throw error;
      await new Promise((resolve) => setTimeout(resolve, 300 * attempt));
    }
  }
}

async function uploadPool(
  entries: UploadEntry[],
  runId: string,
  signal: AbortSignal,
  onBytes: (value: number) => void,
) {
  let cursor = 0;
  let failed: unknown = null;
  const workers = Array.from({ length: Math.min(4, entries.length) }, async () => {
    while (!failed && cursor < entries.length) {
      const entry = entries[cursor++];
      try {
        await uploadOne(runId, entry, signal, onBytes);
      } catch (error) {
        failed = error;
      }
    }
  });
  await Promise.all(workers);
  if (failed) throw failed;
}

function ImageUploadField({
  label,
  files,
  onFiles,
}: {
  label: string;
  files: File[];
  onFiles: (files: File[]) => void;
}) {
  const setSelection = (list: FileList | null) => onFiles(Array.from(list ?? []));
  return (
    <div className="upload-field image-upload-field">
      <span>{label}</span>
      <div className="upload-picker-actions">
        <label className="button secondary">
          Select image files
          <input hidden type="file" accept="image/jpeg,image/png,image/webp" multiple onChange={(event) => setSelection(event.target.files)} />
        </label>
        <label className="button secondary">
          Select image folder
          <input
            hidden
            type="file"
            accept="image/jpeg,image/png,image/webp"
            multiple
            {...({ webkitdirectory: "" } as Record<string, string>)}
            onChange={(event) => setSelection(event.target.files)}
          />
        </label>
      </div>
      <small>{files.length ? `${files.length} images · ${formatBytes(files.reduce((sum, file) => sum + file.size, 0))}` : "JPEG, PNG, or WebP; filename must match product_id"}</small>
    </div>
  );
}

function NewAnalysis({
  active,
  historical,
  onActivated,
}: {
  active: boolean;
  historical: HistoricalSummary | null;
  onActivated: (buildId: string) => Promise<void>;
}) {
  const [selectedMode, setMode] = useState<"full_replace" | "reuse_historical" | null>(null);
  const mode = selectedMode ?? (historical ? "reuse_historical" : "full_replace");
  const [historicalFile, setHistoricalFile] = useState<File | null>(null);
  const [historicalImages, setHistoricalImages] = useState<File[]>([]);
  const [upcomingFile, setUpcomingFile] = useState<File | null>(null);
  const [upcomingImages, setUpcomingImages] = useState<File[]>([]);
  const [run, setRun] = useState<AnalysisRun | null>(null);
  const [uploadedBytes, setUploadedBytes] = useState(0);
  const [totalBytes, setTotalBytes] = useState(0);
  const [uploadComplete, setUploadComplete] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [connectionWarning, setConnectionWarning] = useState("");
  const [elapsedClock, setElapsedClock] = useState(() => Date.now());
  const abortRef = useRef<AbortController | null>(null);
  const onActivatedRef = useRef(onActivated);
  const trackedRunId = run?.id;
  const trackedRunStatus = run?.status;

  useEffect(() => {
    onActivatedRef.current = onActivated;
  }, [onActivated]);

  useEffect(() => {
    if (!run?.startedAt || (run.status !== "queued" && run.status !== "processing")) return;
    const timer = window.setInterval(() => setElapsedClock(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [run?.id, run?.startedAt, run?.status]);

  /** Recover an analysis after a full reload. File inputs cannot be restored,
   * but once inference has started the run ID is all the browser needs. */
  useEffect(() => {
    const controller = new AbortController();
    async function restoreRun() {
      const storedRunId = window.localStorage.getItem(ACTIVE_ANALYSIS_RUN_KEY);
      let restored: AnalysisRun | null = null;
      if (storedRunId) {
        try {
          restored = await responseJson<AnalysisRun>(await fetch(`/api/runs/${storedRunId}`, {
            cache: "no-store",
            signal: controller.signal,
          }));
        } catch {
          if (controller.signal.aborted) return;
          window.localStorage.removeItem(ACTIVE_ANALYSIS_RUN_KEY);
        }
      }
      if (!restored) {
        restored = await responseJson<AnalysisRun | null>(await fetch("/api/runs/active", {
          cache: "no-store",
          signal: controller.signal,
        }));
      }
      if (restored) {
        setRun(restored);
        const stillRunning = restored.status === "queued" || restored.status === "processing";
        setBusy(stillRunning);
        if (!stillRunning) {
          window.localStorage.removeItem(ACTIVE_ANALYSIS_RUN_KEY);
          if (restored.status === "failed") setError(restored.error || restored.message);
        }
      }
    }
    void restoreRun().catch((caught) => {
      if (controller.signal.aborted) return;
      setConnectionWarning(`Could not restore the previous run: ${describeServiceFailure(caught)}`);
    });
    return () => controller.abort();
  }, []);

  /** Own progress tracking for the lifetime of the run, rather than inside the
   * button click. Effects reconnect after Fast Refresh/remount, while polling
   * guarantees forward progress even when an SSE proxy silently buffers. */
  useEffect(() => {
    if (!trackedRunId || (trackedRunStatus !== "queued" && trackedRunStatus !== "processing")) return;
    const runId = trackedRunId;
    window.localStorage.setItem(ACTIVE_ANALYSIS_RUN_KEY, runId);
    let disposed = false;
    let terminalHandled = false;
    let pollInFlight = false;
    let progressEvents: EventSource | null = null;

    function acceptUpdate(update: AnalysisRun) {
      if (disposed || terminalHandled) return;
      setRun(update);
      if (update.status === "queued" || update.status === "processing") return;

      terminalHandled = true;
      progressEvents?.close();
      setBusy(false);
      setConnectionWarning("");
      window.localStorage.removeItem(ACTIVE_ANALYSIS_RUN_KEY);
      if (update.status === "succeeded" && update.buildId) {
        void onActivatedRef.current(update.buildId).catch((caught) => {
          setError(caught instanceof Error ? caught.message : "The completed build could not be loaded");
        });
      } else if (update.status === "failed") {
        setError(update.error || update.message);
      }
    }

    const events = new EventSource(`/api/runs/${runId}/events`);
    progressEvents = events;
    events.onopen = () => {
      if (!disposed) setConnectionWarning("");
    };
    events.onmessage = (event) => {
      try {
        acceptUpdate(JSON.parse(event.data) as AnalysisRun);
      } catch {
        setConnectionWarning("Live progress was unreadable; checking the run automatically…");
      }
    };
    events.onerror = () => {
      if (!disposed && !terminalHandled) {
        setConnectionWarning("Live progress was interrupted; checking the run automatically…");
      }
      // EventSource reconnects itself. Polling below covers proxies that keep a
      // connection open but stop delivering incremental events.
    };

    async function pollRun() {
      if (disposed || terminalHandled || pollInFlight) return;
      pollInFlight = true;
      try {
        const update = await responseJson<AnalysisRun>(await fetch(`/api/runs/${runId}`, {
          cache: "no-store",
        }));
        setConnectionWarning("");
        acceptUpdate(update);
      } catch {
        if (!disposed) {
          setConnectionWarning("Progress updates are reconnecting; the analysis is still running on the server.");
        }
      } finally {
        pollInFlight = false;
      }
    }

    void pollRun();
    const pollTimer = window.setInterval(() => void pollRun(), RUN_POLL_INTERVAL_MS);
    return () => {
      disposed = true;
      events.close();
      window.clearInterval(pollTimer);
    };
  }, [trackedRunId, trackedRunStatus]);

  function missingSelection(): string {
    if (!upcomingFile || !upcomingImages.length) {
      return "Choose the upcoming catalogue and image files or folder.";
    }
    if (mode === "full_replace" && (!historicalFile || !historicalImages.length)) {
      return "Full replacement requires the historical catalogue and images.";
    }
    return "";
  }

  function uploadEntries(): UploadEntry[] {
    return [
      ...(mode === "full_replace" && historicalFile
        ? [{ catalog: "historical" as const, kind: "catalogue" as const, file: historicalFile }, ...historicalImages.map((file) => ({ catalog: "historical" as const, kind: "images" as const, file }))]
        : []),
      ...(upcomingFile ? [{ catalog: "upcoming" as const, kind: "catalogue" as const, file: upcomingFile }] : []),
      ...upcomingImages.map((file) => ({ catalog: "upcoming" as const, kind: "images" as const, file })),
    ];
  }

  /** Step one. Creates the run and transfers the files, then stops: the run
   * stays open on the server until the planner explicitly starts the analysis,
   * so a large upload is never silently followed by an hour of CPU inference. */
  async function uploadFiles() {
    const problem = missingSelection();
    if (problem) {
      setError(problem);
      return;
    }
    setError("");
    const entries = uploadEntries();
    const controller = new AbortController();
    abortRef.current = controller;
    setBusy(true);
    setUploadComplete(false);
    // A resumed upload keeps its byte total; only a brand-new run resets it.
    const existing = run;
    if (!existing) {
      setUploadedBytes(0);
      setTotalBytes(entries.reduce((sum, entry) => sum + entry.file.size, 0));
    }
    try {
      const active = existing ?? await responseJson<AnalysisRun>(await fetch("/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode }),
        signal: controller.signal,
      }));
      setRun(active);
      await uploadPool(entries, active.id, controller.signal, (delta) => setUploadedBytes((value) => value + delta));
      setUploadComplete(true);
      setRun(await responseJson<AnalysisRun>(await fetch(`/api/runs/${active.id}`)));
    } catch (caught) {
      // The run is deliberately left open so the same button can resume it:
      // every byte already accepted is still on the server.
      setError(caught instanceof Error ? caught.message : "Upload failed");
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  }

  /** Step two. Only reachable once every file is on the server. */
  async function startAnalysis() {
    if (!run) return;
    setError("");
    const controller = new AbortController();
    abortRef.current = controller;
    setBusy(true);
    try {
      const queued = await responseJson<AnalysisRun>(await fetch(`/api/runs/${run.id}/complete-upload`, {
        method: "POST",
        signal: controller.signal,
      }));
      window.localStorage.setItem(ACTIVE_ANALYSIS_RUN_KEY, queued.id);
      setRun(queued);
    } catch (caught) {
      setBusy(false);
      setError(caught instanceof Error ? caught.message : "Analysis failed");
    } finally {
      abortRef.current = null;
    }
  }

  async function cancelAnalysis() {
    abortRef.current?.abort();
    if (run?.id) {
      const cancelled = await responseJson<AnalysisRun>(await fetch(`/api/runs/${run.id}/cancel`, { method: "POST" }));
      setRun(cancelled);
    }
    window.localStorage.removeItem(ACTIVE_ANALYSIS_RUN_KEY);
    setBusy(false);
  }

  function resetForNewUpload() {
    setRun(null);
    setUploadComplete(false);
    setUploadedBytes(0);
    setTotalBytes(0);
    setError("");
    setConnectionWarning("");
    window.localStorage.removeItem(ACTIVE_ANALYSIS_RUN_KEY);
  }

  const uploadProgress = totalBytes ? uploadedBytes / totalBytes : 0;
  const displayedProgress = run?.status === "uploading" ? uploadProgress : run?.progress ?? 0;

  /**
   * The flow is two deliberate steps, so the primary action has to name the one
   * step it will actually perform. A single "Start analysis" button that also
   * silently uploaded a gigabyte gave no way to tell the phases apart, and
   * stayed highlighted while work was already running.
   */
  const phase: "idle" | "uploading" | "upload-failed" | "uploaded" | "analysing" | "done" =
    run === null
      ? "idle"
      : run.status === "uploading"
        ? (busy ? "uploading" : uploadComplete ? "uploaded" : "upload-failed")
        : run.status === "queued" || run.status === "processing"
          ? "analysing"
          : "done";
  const running = phase === "uploading" || phase === "analysing";
  const primaryAction = {
    idle: { label: "Upload files", onClick: uploadFiles },
    uploading: { label: "Uploading…", onClick: uploadFiles },
    "upload-failed": { label: "Resume upload", onClick: uploadFiles },
    uploaded: { label: "Start analysis", onClick: startAnalysis },
    analysing: { label: "Analysis running…", onClick: startAnalysis },
    done: { label: "Start another analysis", onClick: resetForNewUpload },
  }[phase];
  const primaryDisabled = running || (phase === "idle" && Boolean(missingSelection()));
  const elapsedSeconds = run ? analysisElapsedSeconds(run, elapsedClock) : null;
  const analysisIsRunning = run?.status === "queued" || run?.status === "processing";

  return (
    <section className="upload-workspace page-wrap" hidden={!active}>
      <div className="upload-hero">
        <span className="eyebrow">Versioned catalogue analysis</span>
        <h1>New analysis</h1>
        <p>The current planner stays available until every new recommendation is complete and validated.</p>
        <div className="template-links">
          <a href="/templates/historical-catalogue.csv">Historical CSV template</a>
          <a href="/templates/upcoming-catalogue.csv">Upcoming CSV template</a>
        </div>
      </div>
      <div className="analysis-mode-grid">
        <button className={mode === "full_replace" ? "active" : ""} onClick={() => setMode("full_replace")}> 
          <strong>Replace historical + upcoming</strong>
          <span>Upload both catalogues and rebuild the historical visual evidence.</span>
        </button>
        <button disabled={!historical} className={mode === "reuse_historical" ? "active" : ""} onClick={() => setMode("reuse_historical")}> 
          <strong>Reuse trained historical</strong>
          <span>{historical ? "Upload only the new upcoming catalogue." : "Available after the first successful full build."}</span>
        </button>
      </div>
      {historical && (
        <div className="historical-summary">
          <span><small>Historical catalogue</small><strong className="historical-updated">Updated {formatAnalysisDate(historical.createdAt)}</strong></span>
          <span><small>Products</small><strong>{historical.productCount}</strong></span>
          <span><small>Images</small><strong>{historical.imageCoverage}</strong></span>
          <span><small>Model</small><strong>v{DISPLAY_MODEL_VERSION}</strong></span>
        </div>
      )}
      <div className="upload-catalog-grid">
        {mode === "full_replace" && (
          <article className="upload-catalog-card">
            <h2><i>1</i> Historical catalogue</h2>
            <label className="upload-field"><span>Historical catalogue file</span><input type="file" accept={catalogueAccept} onChange={(event) => setHistoricalFile(event.target.files?.[0] ?? null)} /><small>CSV, XLSX, XLSM, XLS, XLSB, or ODS using the canonical columns</small></label>
            <ImageUploadField label="Historical product images" files={historicalImages} onFiles={setHistoricalImages} />
          </article>
        )}
        <article className="upload-catalog-card">
          <h2><i>{mode === "full_replace" ? 2 : 1}</i> Upcoming catalogue</h2>
          <label className="upload-field"><span>Upcoming catalogue file</span><input type="file" accept={catalogueAccept} onChange={(event) => setUpcomingFile(event.target.files?.[0] ?? null)} /><small>CSV, XLSX, XLSM, XLS, XLSB, or ODS using the canonical columns</small></label>
          <ImageUploadField label="Upcoming product images" files={upcomingImages} onFiles={setUpcomingImages} />
        </article>
      </div>
      {run && (
        <div className={`run-status ${run.status}`} aria-busy={running}>
          <div>
            <strong>
              {phase === "uploaded"
                ? "Upload complete — ready for analysis"
                : phase === "upload-failed"
                  ? "Upload interrupted"
                  : stageLabel(run)}
            </strong>
            <span>{Math.round((phase === "uploaded" ? 1 : displayedProgress) * 100)}%</span>
          </div>
          <progress max="1" value={phase === "uploaded" ? 1 : displayedProgress} />
          {run.status === "uploading" ? (
            <small>
              {phase === "uploaded"
                ? `${formatBytes(totalBytes)} uploaded successfully. Select Start analysis to begin image processing.`
                : `Uploaded ${formatBytes(uploadedBytes)} of ${formatBytes(totalBytes)}`}
            </small>
          ) : (
            <>
              <dl className="encode-counters">
                <div>
                  <dt>Historical encoded</dt>
                  <dd>
                    {run.mode === "reuse_historical"
                      ? "Reused"
                      : `${run.historicalProcessed ?? 0} / ${run.historicalTotal ?? 0}`}
                  </dd>
                </div>
                <div>
                  <dt>Upcoming encoded</dt>
                  <dd>{`${run.upcomingProcessed ?? 0} / ${run.upcomingTotal ?? 0}`}</dd>
                </div>
                <div>
                  <dt>{analysisIsRunning ? "Time elapsed" : "Total analysis time"}</dt>
                  <dd>{elapsedSeconds === null ? "Not available" : formatElapsedDuration(elapsedSeconds)}</dd>
                </div>
              </dl>
              <small>
                {run.totalCount
                  ? `${run.processedCount ?? 0}/${run.totalCount} images total · ${run.cacheHits ?? 0} cache hits`
                  : "Preparing image processing…"}
              </small>
            </>
          )}
        </div>
      )}
      {error && <div className="upload-error">{error}{run?.validationReportUrl && <> · <a href={run.validationReportUrl}>Download validation report</a></>}</div>}
      {connectionWarning && <div className="upload-warning" role="status">{connectionWarning}</div>}
      <div className="upload-submit-row">
        {/* Muted, never "primary", while work is in flight: a highlighted
            call-to-action invites a click on something already running. */}
        <button
          className={`button ${primaryDisabled ? "" : "primary"}`}
          disabled={primaryDisabled}
          aria-busy={running}
          onClick={primaryAction.onClick}
        >
          {running && <span className="button-spinner" aria-hidden="true" />}
          {primaryAction.label}
        </button>
        {(running || phase === "uploaded" || phase === "upload-failed") && (
          <button className="button secondary" onClick={cancelAnalysis}>
            {phase === "analysing" ? "Cancel analysis" : "Discard upload"}
          </button>
        )}
      </div>
    </section>
  );
}

function App() {
  const initialItem = visibleUpcoming.find(
    (item) => item.recommendation.matchConfidence === "High" && item.imageUrl,
  ) ?? visibleUpcoming[0] ?? PLACEHOLDER_UPCOMING;
  const [tab, setTab] = useState<Tab>(visibleUpcoming.length > 0 ? "compare" : "upload");
  const [dataRevision, setDataRevision] = useState(0);
  const [activeBuildId, setActiveBuildId] = useState("");
  const [activeBuildCreatedAt, setActiveBuildCreatedAt] = useState("");
  const [activeHistorical, setActiveHistorical] = useState<HistoricalSummary | null>(null);
  const [selectedId, setSelectedId] = useState(initialItem.id);
  const [queueSearch, setQueueSearch] = useState("");
  const [segment, setSegment] = useState("All");
  const [matchConfidenceFilter, setMatchConfidenceFilter] = useState("All");
  const [focusedHistoricalId, setFocusedHistoricalId] = useState<string | null>(null);
  const [minimumSimilarity, setMinimumSimilarity] = useState(
    dataset.meta.model.minimumVisualScore,
  );
  const [targetSellThrough, setTargetSellThrough] = useState(
    dataset.meta.model.targetSellThrough ?? 0.70,
  );
  const [overrides, setOverrides] = useState<Record<string, number>>({});
  const [approvedIds, setApprovedIds] = useState<Record<string, boolean>>({});
  const [toast, setToast] = useState("");
  const [serviceError, setServiceError] = useState("");
  const tabChosenByUser = useRef(false);

  function chooseTab(next: Tab) {
    tabChosenByUser.current = true;
    setTab(next);
  }

  const hasActiveBuild = visibleUpcoming.length > 0;
  const selected = visibleUpcoming.find((item) => item.id === selectedId) ?? initialItem;
  const decision = useMemo(
    () => {
      void dataRevision;
      return makeDecision(
        selected,
        minimumSimilarity,
        targetSellThrough,
        focusedHistoricalId,
      );
    },
    [selected, minimumSimilarity, targetSellThrough, focusedHistoricalId, dataRevision],
  );
  const selectedMatches = decision.eligible.slice(0, 4);

  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(""), 2600);
    return () => window.clearTimeout(timer);
  }, [toast]);

  async function refreshActive(expectedBuildId?: string) {
    const activeResponse = await fetch("/api/active", { cache: "no-store" });
    // 404 is the documented "nothing uploaded yet" answer, not a failure: it is
    // the empty state, and must not be reported as a broken service.
    if (activeResponse.status === 404 && !expectedBuildId) {
      setActiveBuildId("");
      setActiveBuildCreatedAt("");
      return;
    }
    const active = await responseJson<{
      id: string;
      createdAt: string;
      artifactUrl: string;
    }>(activeResponse);
    const artifactUrl = expectedBuildId ? `/api/builds/${expectedBuildId}/artifact` : active.artifactUrl;
    const artifact = await responseJson<Dataset>(await fetch(artifactUrl, { cache: "no-store" }));
    installDataset(artifact);
    const nextInitial = visibleUpcoming.find((item) => item.recommendation.matchConfidence === "High" && item.imageUrl) ?? visibleUpcoming[0];
    if (nextInitial) setSelectedId(nextInitial.id);
    setFocusedHistoricalId(null);
    setMinimumSimilarity(dataset.meta.model.minimumVisualScore);
    setTargetSellThrough(dataset.meta.model.targetSellThrough ?? 0.70);
    setActiveBuildId(expectedBuildId ?? active.id);
    setActiveBuildCreatedAt(active.createdAt);
    setDataRevision((value) => value + 1);
    const summary = await responseJson<HistoricalSummary | null>(await fetch("/api/historical/active", { cache: "no-store" }));
    setActiveHistorical(summary);
  }

  useEffect(() => {
    const timer = window.setTimeout(() => {
      refreshActive()
        .then(() => {
          setServiceError("");
          // The first render has no dataset — the build is fetched — so the
          // workspace opens on New analysis. Once a build does load, move to
          // Compare, unless the user has already picked a tab themselves.
          if (visibleUpcoming.length > 0 && !tabChosenByUser.current) setTab("compare");
        })
        .catch((error) => {
          // Reaching here means the request itself failed, not that the
          // catalogue is empty — most often the analysis service is not
          // running, which the Next.js proxy reports as a bare 500.
          setServiceError(describeServiceFailure(error));
        });
    }, 0);
    return () => window.clearTimeout(timer);
  }, []);

  const portfolio = useMemo(
    () => {
      void dataRevision;
      return visibleUpcoming.map((item) => ({
        item,
        decision: makeDecision(item, minimumSimilarity, targetSellThrough),
      }));
    },
    [minimumSimilarity, targetSellThrough, dataRevision],
  );

  const queueItems = useMemo(() => {
    const query = queueSearch.trim().toUpperCase();
    return portfolio.filter(({ item, decision: itemDecision }) => {
      const searchable = [
        item.id,
        item.design,
        item.colour,
        item.categoryType,
        item.fabric,
      ]
        .join(" ")
        .toUpperCase();
      return (
        (!query || searchable.includes(query)) &&
        (segment === "All" || item.itemType === segment) &&
        (matchConfidenceFilter === "All" || itemDecision.matchConfidence === matchConfidenceFilter)
      );
    });
  }, [portfolio, queueSearch, segment, matchConfidenceFilter]);

  const focusedMatch = decision.selectedMatch;
  const focusedHistory = focusedMatch ? historyById.get(focusedMatch.historicalId) : undefined;
  const finalQuantity = overrides[selected.id] ?? decision.quantity;
  const sales = salesFromOrder(decision, finalQuantity);
  const isApproved = Boolean(approvedIds[selected.id]);

  const totalOrder = portfolio.reduce(
    (sum, { item, decision: itemDecision }) => sum + (overrides[item.id] ?? itemDecision.quantity),
    0,
  );

  function chooseItem(id: string) {
    setSelectedId(id);
    setFocusedHistoricalId(null);
    chooseTab("compare");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function unapproveItem(id: string) {
    setApprovedIds((current) => {
      if (!(id in current)) return current;
      const next = { ...current };
      delete next[id];
      return next;
    });
  }

  function chooseAnalogue(historicalId: string) {
    setFocusedHistoricalId(historicalId);
    setOverrides((current) => {
      const next = { ...current };
      delete next[selected.id];
      return next;
    });
    unapproveItem(selected.id);
  }

  function changeMinimumSimilarity(value: number) {
    setMinimumSimilarity(value);
    setOverrides({});
    setApprovedIds({});
  }

  function changeTargetSellThrough(value: number) {
    setTargetSellThrough(value);
    setOverrides({});
    setApprovedIds({});
  }

  function applyOverride(value: number) {
    if (!Number.isFinite(value) || value < 0) return;
    setOverrides((current) => ({ ...current, [selected.id]: Math.round(value) }));
    unapproveItem(selected.id);
    setToast(`Planner quantity saved for ${selected.id}`);
  }

  function resetOverride() {
    setOverrides((current) => {
      const next = { ...current };
      delete next[selected.id];
      return next;
    });
    unapproveItem(selected.id);
    setToast("System recommendation restored");
  }

  function approveOrder() {
    setApprovedIds((current) => ({ ...current, [selected.id]: true }));
    setToast(`${selected.id} approved at ${numberFormatter.format(finalQuantity)} units`);
  }

  function exportCsv() {
    const scoreValue = (value: number | null | undefined) =>
      value === null || value === undefined ? "" : Math.round(value * 100);

    // One column per question a planner actually asks of this file. The four
    // similarity sub-scores stay because they explain why the past product was
    // selected. Competing demand statistics — the mean alongside the median
    // and the p10/p90 band — stay out because they make one forecast look like
    // several contradictory answers. The range remains available on the card,
    // where the evidence context explains it.
    const rows = [
      [
        "Product ID",
        "Product type",
        "Design",
        "Colour",
        "Category",
        "Match status",
        "Similar past product",
        "Overall similarity (%)",
        "Colour similarity (%)",
        "Pattern similarity (%)",
        "Style similarity (%)",
        "Texture similarity (%)",
        "Match confidence",
        "Expected sales (units)",
        "Past product sales (units)",
        "Past product order (units)",
        "Recommended order (units)",
        // Audit trail rather than an on-card warning: a capped order is a
        // policy limit, not a target the model actually reached. That does not
        // belong in the planner's face, but it does belong in offline analysis.
        "Maximum order (units)",
        "Order limit applied",
        // Always the number to order — the planner's figure when they set one,
        // the model's otherwise. An override-only column left most rows blank
        // and made every reader recompute the fallback by hand.
        "Final order (units)",
        "Approval status",
      ],
      ...portfolio.map(({ item, decision: itemDecision }) => {
        const top = itemDecision.noSuitableMatch
          ? undefined
          : itemDecision.ranked[0];
        const finalOrder = overrides[item.id] ?? itemDecision.quantity;
        const expectedSales = salesFromOrder(itemDecision, finalOrder).headline;
        return [
          item.id,
          item.itemType,
          item.design,
          item.colour,
          item.categoryType,
          itemDecision.noSuitableMatch ? "No product match" : "Matched",
          top?.historicalId ?? "",
          scoreValue(top?.visualScore),
          scoreValue(top?.colourVisualScore),
          scoreValue(top?.dinoVisualScore),
          scoreValue(top?.fashionVisualScore),
          scoreValue(top?.textureVisualScore),
          itemDecision.matchConfidence,
          expectedSales,
          itemDecision.analogueSales,
          itemDecision.analogueQuantity,
          itemDecision.quantity,
          itemDecision.buyCeiling,
          itemDecision.quantityCapped ? "Yes" : "No",
          finalOrder,
          approvedIds[item.id] ? "Approved" : "Pending",
        ];
      }),
    ];
    const csv = rows
      .map((row) => row.map((value) => `"${String(value).replaceAll('"', '""')}"`).join(","))
      .join("\n");
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = "turtle-season-recommendations.csv";
    link.click();
    URL.revokeObjectURL(url);
    setToast("Recommendation export prepared");
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <button className="brand" onClick={() => chooseTab("compare")} aria-label="Open comparison workspace">
          <span className="brand-mark">T</span>
          <span>
            <strong>TURTLE</strong>
            <small>Season Intelligence</small>
          </span>
        </button>
        <nav className="main-nav" aria-label="Main navigation">
          {([
            ["compare", "Compare"],
            ["portfolio", "Portfolio"],
            ["upload", "New analysis"],
          ] as [Tab, string][]).map(([key, label]) => (
            <button key={key} className={tab === key ? "active" : ""} onClick={() => chooseTab(key)}>
              {label}
            </button>
          ))}
        </nav>
        <div className="top-actions">
          <span className="sync-state">
            <i /> {activeBuildId
              ? `Last updated · ${formatAnalysisDate(activeBuildCreatedAt)}`
              : "No recommendations available"}
          </span>
          <span className="avatar" aria-label="Planner profile">SD</span>
        </div>
      </header>

      {(tab === "compare" || tab === "portfolio") && !hasActiveBuild && (
        <section className="page-wrap no-build-state" aria-live="polite">
          <span className="eyebrow">{serviceError ? "Analysis service unavailable" : "No recommendations available"}</span>
          <h1>{serviceError ? "The planner cannot load its data" : "Upload a catalogue to start planning"}</h1>
          {serviceError ? (
            <p className="service-error">{serviceError}</p>
          ) : (
            <p>
              The planner reads only the analysis you upload. Add a historical and an
              upcoming catalogue with their product images in <strong>New analysis</strong>;
              the comparison and portfolio views open as soon as the build activates.
            </p>
          )}
          <button className="button primary" onClick={() => chooseTab("upload")}>
            Start a new analysis
          </button>
        </section>
      )}

      {tab === "compare" && hasActiveBuild && (
        <div className="comparison-shell">
          <aside className="queue-panel">
            <div className="queue-heading">
              <div>
                <span className="eyebrow">Upcoming assortment</span>
                <h2>{queueItems.length} styles</h2>
              </div>
              <span className="season-chip">{dataset.meta.upcomingSeason}</span>
            </div>
            <label className="search-box">
              <span aria-hidden="true">⌕</span>
              <input
                value={queueSearch}
                onChange={(event) => setQueueSearch(event.target.value)}
                placeholder="Search style, design, colour, category or fabric"
                aria-label="Search upcoming styles by style code, design, colour, category type or fabric"
                title="Searches style code, design, colour, category type and fabric"
                type="search"
                autoComplete="off"
                spellCheck={false}
              />
              {queueSearch && (
                <button
                  type="button"
                  className="search-clear"
                  onClick={() => setQueueSearch("")}
                  aria-label="Clear search"
                >
                  ×
                </button>
              )}
            </label>
            <div className="queue-filters">
              <select value={segment} onChange={(event) => setSegment(event.target.value)} aria-label="Filter category">
                <option>All</option>
                {productSegments.map((itemType) => (
                  <option key={itemType}>{itemType}</option>
                ))}
              </select>
              <select value={matchConfidenceFilter} onChange={(event) => setMatchConfidenceFilter(event.target.value)} aria-label="Filter match confidence">
                <option value="All">All matches</option>
                <option value="High">High match</option>
                <option value="Medium">Medium match</option>
                <option value="Low">No convincing match</option>
              </select>
            </div>
            <div className="queue-list">
              {queueItems.map(({ item, decision: itemDecision }) => (
                <button
                  key={item.id}
                  className={`queue-item ${selected.id === item.id ? "selected" : ""}`}
                  onClick={() => {
                    setSelectedId(item.id);
                    setFocusedHistoricalId(null);
                  }}
                >
                  <ProductImage src={item.imageUrl} alt={item.id} className="queue-image" />
                  <span className="queue-copy">
                    <strong>{item.id}</strong>
                    <small>{item.design} · {item.colour}</small>
                    <span className="queue-commercial">
                      <span>{item.categoryType}</span>
                    </span>
                    <span className="mini-signals">
                      {itemDecision.noSuitableMatch ? (
                        <span className="mini-confidence no-match">
                          No product match
                        </span>
                      ) : (
                        <span className={`mini-confidence ${itemDecision.matchConfidence.toLowerCase()}`}>
                          {itemDecision.matchConfidence} match
                        </span>
                      )}
                    </span>
                  </span>
                  <span className="queue-qty">{numberFormatter.format(itemDecision.quantity)}<small>units</small></span>
                </button>
              ))}
              {!queueItems.length && <div className="empty-state">No styles match these filters.</div>}
            </div>
          </aside>

          <section className="workspace">
            <div className="workspace-heading">
              <div>
                <span className="eyebrow">Decision workspace</span>
                <h1>{selected.id}</h1>
              </div>
              <div className="workspace-stepper" aria-label="Decision progress">
                <span
                  className={decision.noSuitableMatch ? "blocked" : "done"}
                  title={decision.noSuitableMatch
                    ? "No visual analogue cleared the similarity threshold"
                    : "A historical analogue was matched"}
                >
                  1 <small>{decision.noSuitableMatch ? "No match" : "Matched"}</small>
                </span>
                <i />
                <span className={isApproved ? "done" : "current"} aria-current={isApproved ? undefined : "step"}>
                  2 <small>Review</small>
                </span>
                <i />
                <span className={isApproved ? "current" : ""} aria-current={isApproved ? "step" : undefined}>
                  3 <small>{isApproved ? "Approved" : "Approve"}</small>
                </span>
              </div>
            </div>

            <div className="decision-grid">
              <article className="upcoming-card">
                <div className="card-label-row">
                  <span className="card-label">Upcoming style</span>
                  {!selected.imageUrl && <span className="warning-chip">Image missing</span>}
                </div>
                <ProductImage src={selected.imageUrl} alt={`Upcoming ${selected.id}`} className="hero-product-image" eager />
                <MatchAttributeCatalog key={selected.id} product={selected} context="Upcoming" />
              </article>

              <article className="recommendation-card">
                <div className="recommendation-topline">
                  <span className="card-label" title="AI order quantity recommendation">AI order quantity recommendation</span>
                  {/* Every status chip sits on its own row under the heading, and
                      the row is always present, so the card's height does not
                      shift as chips come and go between products. */}
                  <div className="recommendation-signals">
                    {decision.noSuitableMatch
                      ? <NoMatchPill />
                      : <MatchConfidencePill confidence={decision.matchConfidence} />}
                    {decision.forecast?.wideUncertainty && (
                      <span
                        className="warning-chip"
                        title="Thin or conflicting evidence: the average forecast is pulled well above the typical outcome. Treat the range, not the headline number, as the estimate."
                      >
                        Wide uncertainty
                      </span>
                    )}
                    {decision.quantityCapped && (
                      <span
                        className="warning-chip"
                        title={`The model wanted more than the ${numberFormatter.format(decision.buyCeiling)}-unit maximum order for ${selected.itemType} to reach ${Math.round(targetSellThrough * 100)}% expected sell-through. This number is a policy limit, not a target that was actually reached.`}
                      >
                        Capped at max order
                      </span>
                    )}
                    {sales.demandExceedsOrder && (
                      <span
                        className="warning-chip"
                        title={`Forecast demand is ${numberFormatter.format(headlineDemand(decision))} units, above this ${numberFormatter.format(finalQuantity)}-unit order. Sales are limited by the order, not by demand — lower the target sell-through or raise the planner quantity to capture more.`}
                      >
                        Demand exceeds order
                      </span>
                    )}
                  </div>
                </div>
                <div className="quantity-hero">
                  <div className="quantity-primary">
                    <small>{decision.forecast ? "Initial order to place" : "Order prediction"}</small>
                    <div>
                      <strong>{numberFormatter.format(finalQuantity)}</strong>
                      <span>units</span>
                    </div>
                    {!decision.forecast && (
                      <p>Selected product&apos;s sales ÷ {Math.round(targetSellThrough * 100)}% target sell-through</p>
                    )}
                  </div>
                  <div className="quantity-secondary">
                    <small>{decision.forecast ? "Typical sales forecast" : "Matched product sales"}</small>
                    <strong>{numberFormatter.format(sales.headline)} units</strong>
                    {decision.forecast ? (
                      <div
                        className="forecast-evidence"
                        title="The model estimates an 80% chance that sales will fall within this range."
                      >
                        <div className="forecast-range">
                          <span className="forecast-range-label">Likely sales range <em>80%</em></span>
                          <strong className="forecast-range-value">
                            {numberFormatter.format(sales.low)}–{numberFormatter.format(sales.high)} <small>units</small>
                          </strong>
                        </div>
                        <span className="forecast-evidence-note">
                          Based on <b>{decision.forecast.analoguesUsed}</b> similar past style{decision.forecast.analoguesUsed === 1 ? "" : "s"}
                        </span>
                      </div>
                    ) : (
                      <span>Cleaned observed sales from the one selected historical product</span>
                    )}
                  </div>
                </div>
                <div className="match-confidence-row">
                  <span>Selected analogue similarity</span>
                  <div className="match-confidence-track" aria-label={`Selected historical match score ${scorePercent(focusedMatch?.combinedScore ?? 0)}`}>
                    <span style={{ width: `${Math.round((focusedMatch?.combinedScore ?? 0) * 100)}%` }} />
                  </div>
                  <strong>{scorePercent(focusedMatch?.combinedScore ?? 0)}</strong>
                </div>
                <div className="rationale-box">
                  <span className="rationale-icon">✦</span>
                  <div className="rationale-copy">
                    <strong>Why this recommendation</strong>
                    <p>
                      {decision.noSuitableMatch ? (
                        <>
                          No historical product cleared the visual-match threshold. No sales or order quantity is generated; planner review is required.
                        </>
                      ) : decision.forecast ? (
                        <>
                          {sellThroughTradeoff(targetSellThrough)}
                          {decision.forecast.wideUncertainty && " Historical evidence is limited, so use the likely sales range above rather than treating the point forecast as certain."}
                          {decision.quantityCapped && " The maximum-order policy is binding, so the selected sell-through target may not be achieved."}
                        </>
                      ) : (
                        <>
                          This rule-based order scales the selected past product&apos;s observed sales to the chosen sell-through target. Review the historical match before approval.
                        </>
                      )}
                    </p>
                  </div>
                </div>
                <div className="recommendation-metrics">
                  <div className="analogue-tile">
                    {!decision.noSuitableMatch && focusedHistory?.imageUrl ? (
                      <ProductImage src={focusedHistory.imageUrl} alt={`Historical analogue ${focusedHistory.id}`} className="analogue-thumb" />
                    ) : (
                      <strong>{decision.noSuitableMatch ? "No accepted product" : focusedMatch?.historicalId}</strong>
                    )}
                  </div>
                  <div className="metrics-sales">
                    <small>Historical sales evidence</small>
                    <strong>{decision.noSuitableMatch ? "Not used" : `${numberFormatter.format(decision.analogueSales)} units`}</strong>
                  </div>
                  <div className="metrics-order"><small>Historical original order</small><strong>{decision.noSuitableMatch ? "Not used" : `${numberFormatter.format(decision.analogueQuantity)} units`}</strong></div>
                  <div className="override-row">
                    <label>
                      Planner quantity
                      <input
                        type="number"
                        min="0"
                        step="25"
                        value={finalQuantity}
                        onChange={(event) => applyOverride(Number(event.target.value))}
                      />
                    </label>
                    <div className="override-actions">
                      {overrides[selected.id] && <button className="text-button" onClick={resetOverride}>Reset</button>}
                      <button
                        className={`button primary ${isApproved ? "approved" : ""}`}
                        onClick={approveOrder}
                      >
                        {isApproved ? "✓ Approved" : "Approve order"}
                      </button>
                    </div>
                  </div>
                </div>
                {focusedHistory && (
                  <div className="workbook-audit-panel">
                    <div className="workbook-audit-heading">
                      <div>
                        <span>Attribute reference</span>
                      </div>
                      <span className="analogue-count-chip">Not used in matching</span>
                    </div>
                    <div className="attribute-evidence-legend">
                      <span>Upcoming</span>
                      <i aria-hidden="true">→</i>
                      <span>Historical</span>
                    </div>
                    <div className="attribute-evidence">
                      {catalogAttributeOrder.map((key) => {
                        const upcomingValue = attributeValue(selected, key);
                        const historicalValue = attributeValue(focusedHistory, key);
                        const matchStatus = attributeMatchStatus(upcomingValue, historicalValue);
                        return (
                          <div className="attribute-comparison" key={key}>
                            <div className="attribute-comparison-heading">
                              <span>{attributeLabels[key] ?? key}</span>
                              <strong className={`attribute-match-status ${matchStatus}`}>
                                {matchStatus === "exact" ? "Exact match" : "Different"}
                              </strong>
                            </div>
                            <div className="attribute-value-pair">
                              <span title={upcomingValue}>{upcomingValue}</span>
                              <i aria-hidden="true">→</i>
                              <span title={historicalValue}>{historicalValue}</span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </article>

              <aside className="settings-card">
                <div className="card-label-row">
                  <span className="card-label">Decision policy</span>
                  <span className="live-chip">Live</span>
                </div>
                <label className="range-control">
                  <span><b>Minimum visual similarity</b><strong>{scorePercent(minimumSimilarity)}</strong></span>
                  <input
                    type="range"
                    min="10"
                    max="90"
                    step="1"
                    value={Math.round(minimumSimilarity * 100)}
                    onChange={(event) => changeMinimumSimilarity(Number(event.target.value) / 100)}
                    aria-label="Minimum visual similarity"
                  />
                  <small className="setting-help">The selected analogue must meet this score before it can drive an order.</small>
                </label>
                <label className="range-control">
                  <span><b>Target sell-through</b><strong>{Math.round(targetSellThrough * 100)}%</strong></span>
                  <input
                    type="range"
                    min="10"
                    max="90"
                    step="1"
                    value={Math.round(targetSellThrough * 100)}
                    onChange={(event) => changeTargetSellThrough(Number(event.target.value) / 100)}
                    aria-label="Target sell-through"
                  />
                  {/* Written as a live sentence rather than a static definition: the
                      figure tracks the slider, and the two clauses tell the planner
                      which way to move it and what they trade away by doing so. */}
                  <small className="setting-help">
                    {demandModel ? (
                      <>
                        Aim to sell <b>{Math.round(targetSellThrough * 100)}%</b> of what you order.
                        Raise it for a smaller, safer order; lower it for a larger one that chases
                        more demand.
                      </>
                    ) : (
                      <>
                        Aim to sell <b>{Math.round(targetSellThrough * 100)}%</b> of what you order.
                        The order is the matched product sales divided by this target.
                      </>
                    )}
                  </small>
                </label>
                {focusedHistory && focusedMatch ? (
                  <aside className="evidence-summary" aria-label="Similarity score summary">
                    <div className="evidence-summary-heading">
                      <span>Similarity scores</span>
                      <small>Higher is closer</small>
                    </div>
                    <div className="evidence-scores evidence-scores-overall">
                      <ScoreRing score={focusedMatch.combinedScore} label="Overall match" />
                    </div>
                    <div className="evidence-scores evidence-scores-components">
                      <ScoreRing score={focusedMatch.colourVisualScore ?? 0} label="Colour" />
                      <ScoreRing score={focusedMatch.dinoVisualScore ?? 0} label="Pattern" />
                      <ScoreRing score={focusedMatch.fashionVisualScore ?? 0} label="Style" />
                      <ScoreRing score={focusedMatch.textureVisualScore ?? 0} label="Texture" />
                    </div>
                    <p>
                      {visualMatchingAvailable
                        ? `Ranked by how closely the product photos match in style, colour and pattern.${selected.itemType === "OTTR" ? " For trousers, only the garment is compared — footwear in the photo is ignored." : ""} Attribute details like fabric or category aren't used to rank.`
                        : "No visual match is available for this product yet."}
                    </p>
                  </aside>
                ) : (
                  <aside className="evidence-summary evidence-summary-empty" aria-label="Similarity score summary">
                    <div className="evidence-summary-heading">
                      <span>Similarity scores</span>
                    </div>
                    <p>
                      <span aria-hidden="true">∅</span>
                      No historical product cleared the visual-match threshold, so there is no similarity score to show.
                    </p>
                  </aside>
                )}
              </aside>
            </div>

            <section className="matches-section">
              <div className="section-heading">
                <div>
                  <span className="eyebrow">
                    Recommended historical analogues
                  </span>
                  <h2>
                    {demandModel
                      ? "Every accepted analogue feeds the forecast; select one to compare it"
                      : "Select one product to calculate the order quantity"}
                  </h2>
                  {decision.noSuitableMatch && (
                    <p className="section-supporting-copy">
                      {decision.ranked.length === 0
                        ? "No candidate survived the visual colour and pattern gates."
                        : `No candidate meets the ${scorePercent(minimumSimilarity)} criterion. Lower the criterion to reveal weaker matches.`}
                    </p>
                  )}
                </div>
                <div className="analogue-header-tools">
                  {decision.noSuitableMatch && (
                    <span className="analogue-count-chip rejected">0 accepted</span>
                  )}
                  <div className="score-legend" title="Ranked by visual similarity in colour, pattern, style and texture">
                    <span><i className="visual" /> Visual AI similarity</span>
                  </div>
                </div>
              </div>
              {selectedMatches.length === 0 ? (
                <div className="no-product-match-state" role="status">
                  <span aria-hidden="true">∅</span>
                  <div>
                    <strong>{decision.ranked.length === 0 ? "No gated visual candidate is available" : `No candidate meets the ${scorePercent(minimumSimilarity)} criterion`}</strong>
                    <p>
                      {decision.ranked.length === 0
                        ? "No historical image passed the visual colour and pattern gates."
                        : "Lower the minimum visual similarity to reveal weaker candidates."}
                      {" "}No sales or order quantity is generated.
                    </p>
                  </div>
                </div>
              ) : (
                <div className="match-grid match-grid-4">
                  {selectedMatches.map((match, index) => {
                    const historical = historyById.get(match.historicalId);
                    if (!historical) return null;
                    return (
                      <article
                        key={match.historicalId}
                        className={`match-card ${focusedMatch?.historicalId === match.historicalId ? "active" : ""} ${Math.round(match.combinedScore * 100) < Math.round(minimumSimilarity * 100) ? "below-threshold" : ""}`}
                      >
                        <button
                          type="button"
                          className="match-card-select"
                          aria-label={`Select ${historical.id} for order quantity recommendation`}
                          onClick={() => chooseAnalogue(match.historicalId)}
                        >
                          <span className="rank">#{index + 1}</span>
                          <div className="match-image-frame">
                            <ProductImage src={historical.imageUrl} alt={historical.id} className="match-image" />
                          </div>
                          <div className="match-copy">
                            <div className="match-title"><strong>{historical.id}</strong><span>{scorePercent(match.combinedScore)}</span></div>
                            {Math.round(match.combinedScore * 100) < Math.round(minimumSimilarity * 100) && <small className="threshold-note">Below current similarity criterion</small>}
                            <div className="dual-bars">
                              <span><i style={{ width: `${(match.visualScore ?? 0) * 100}%` }} /></span>
                            </div>
                            <div className="match-selection-status">
                              {focusedMatch?.historicalId === match.historicalId
                                ? "Selected for order recommendation"
                                : "Select this analogue"}
                            </div>
                          </div>
                        </button>
                        <div className="match-card-catalog">
                          <div className="match-performance">
                            <span><small>Order</small><strong>{numberFormatter.format(historical.order)}</strong></span>
                            <span><small>Sales</small><strong>{numberFormatter.format(historical.salesTarget)}</strong></span>
                            <span><small>Sell-through</small><strong>{scorePercent(historical.sellThrough)}</strong></span>
                          </div>
                          <MatchAttributeCatalog key={historical.id} product={historical} context="Historical" />
                        </div>
                      </article>
                    );
                  })}
                </div>
              )}
            </section>

          </section>
        </div>
      )}

      {tab === "portfolio" && hasActiveBuild && (
        <section className="portfolio-page page-wrap">
          <div className="page-heading">
            <div><span className="eyebrow">Upcoming assortment</span><h1>Portfolio recommendation</h1><p>Review expected sales, analogue evidence, uncertainty and recommended initial orders across the complete {dataset.meta.upcomingSeason} assortment.</p></div>
            <button className="button primary" onClick={exportCsv}>Export recommendation file</button>
          </div>
          <div className="kpi-grid">
            <article><span>Total styles</span><strong>{dataset.meta.upcomingItems}</strong><small>{dataset.meta.upcomingImageCoverage} with images</small></article>
            <article>
              <span>Visual AI coverage</span>
              <strong>{visualMatchingAvailable ? scorePercent(dataset.meta.upcomingImageCoverage / dataset.meta.upcomingItems) : "Pending mapping"}</strong>
              <small>{visualMatchingAvailable ? `${dataset.meta.missingUpcomingImages.length} linked-image exceptions` : "Attribute-only recommendations active"}</small>
            </article>
            <article><span>Single-match decisions</span><strong>{portfolio.filter(({ decision }) => !decision.noSuitableMatch).length}</strong><small>accepted visual analogues</small></article>
            <article className="accent"><span>Order prediction</span><strong>{numberFormatter.format(totalOrder)}</strong><small>units across {dataset.meta.upcomingSeason}</small></article>
          </div>
          <div className="portfolio-toolbar">
            <label className="search-box wide">
              <span aria-hidden="true">⌕</span>
              <input
                value={queueSearch}
                onChange={(event) => setQueueSearch(event.target.value)}
                placeholder="Search style code, design, colour, category or fabric"
                aria-label="Search portfolio by style code, design, colour, category type or fabric"
                title="Searches style code, design, colour, category type and fabric"
                type="search"
                autoComplete="off"
                spellCheck={false}
              />
              {queueSearch && (
                <button
                  type="button"
                  className="search-clear"
                  onClick={() => setQueueSearch("")}
                  aria-label="Clear search"
                >
                  ×
                </button>
              )}
            </label>
            <select value={segment} onChange={(event) => setSegment(event.target.value)} aria-label="Filter portfolio category">
              <option>All</option>
              {productSegments.map((itemType) => <option key={itemType}>{itemType}</option>)}
            </select>
            <select value={matchConfidenceFilter} onChange={(event) => setMatchConfidenceFilter(event.target.value)} aria-label="Filter portfolio match confidence"><option value="All">All matches</option><option value="High">High match</option><option value="Medium">Medium match</option><option value="Low">No convincing match</option></select>
            <span>{queueItems.length} results</span>
          </div>
          <div className="portfolio-table-wrap">
            <table className="portfolio-table">
              <thead><tr><th>Upcoming style</th><th>Product attributes</th><th>Top historical analogue</th><th>Match score</th><th>Decision signals</th><th>{demandModel ? "Sales prediction" : "Expected sales"}</th><th>Order prediction</th><th>Planner order</th><th><span className="sr-only">Actions</span></th></tr></thead>
              <tbody>
                {queueItems.length === 0 && (
                  <tr className="table-empty-row">
                    <td colSpan={9}>No styles match these filters.</td>
                  </tr>
                )}
                {queueItems.map(({ item, decision: itemDecision }) => {
                  const top = itemDecision.noSuitableMatch
                    ? undefined
                    : itemDecision.ranked[0];
                  const historical = top ? historyById.get(top.historicalId) : undefined;
                  // This column is headed "Sales prediction", so it is bounded by
                  // the buy the same way the workspace card is.
                  const itemSales = salesFromOrder(itemDecision, overrides[item.id] ?? itemDecision.quantity);
                  return (
                    <tr key={item.id}>
                      <td><div className="table-product"><ProductImage src={item.imageUrl} alt={item.id} className="table-image" /><span><strong>{item.id}</strong><small>{item.colour}</small></span></div></td>
                      <td><strong>{item.design}</strong><small>{item.categoryType} · {item.fabric}</small></td>
                      <td>
                        {historical ? (
                          <div className="table-product"><ProductImage src={historical.imageUrl} alt={historical.id} className="table-image" /><span><strong>{historical.id}</strong><small>{historical.season} · ST {scorePercent(historical.sellThrough)}</small></span></div>
                        ) : (
                          <strong className="no-match-table-label">No product match</strong>
                        )}
                      </td>
                      <td>
                        {top ? (
                          <strong className="match-score">{scorePercent(top.combinedScore)}</strong>
                        ) : (
                          <><strong className="no-match-table-label">Below threshold</strong><small>Candidate suppressed</small></>
                        )}
                      </td>
                      <td><div className="table-signals">{itemDecision.noSuitableMatch ? <NoMatchPill /> : <MatchConfidencePill confidence={itemDecision.matchConfidence} detailed />}</div></td>
                      <td>
                        <strong>{numberFormatter.format(itemSales.headline)}</strong>
                        <small>
                          {itemDecision.noSuitableMatch
                            ? "not generated"
                            : itemDecision.forecast
                              ? `${numberFormatter.format(itemSales.low)}–${numberFormatter.format(itemSales.high)}`
                              : "selected product actual"}
                        </small>
                      </td>
                      <td>
                        <strong>{numberFormatter.format(itemDecision.quantity)}</strong>
                        <small>
                          {itemDecision.noSuitableMatch
                            ? "planner review"
                            : itemDecision.forecast
                              ? `${Math.round(targetSellThrough * 100)}% expected ST`
                              : `sales ÷ ${Math.round(targetSellThrough * 100)}% ST`}
                        </small>
                      </td>
                      <td><strong>{overrides[item.id] ? numberFormatter.format(overrides[item.id]) : "—"}</strong><small>{overrides[item.id] ? "Adjusted" : "Pending"}</small></td>
                      <td><button className="row-action" onClick={() => chooseItem(item.id)}>Review →</button></td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* Keep this workspace mounted across navigation. Unmounting it loses the
          selected files, upload counters, and live EventSource connection. */}
      <NewAnalysis
        active={tab === "upload"}
        historical={activeHistorical}
        onActivated={async (buildId) => {
          await refreshActive(buildId);
          setTab("compare");
          setToast("New recommendations activated");
        }}
      />

      {toast && <div className="toast" role="status">✓ {toast}</div>}
    </main>
  );
}

export default App;
