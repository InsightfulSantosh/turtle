"use client";

import { useEffect, useMemo, useState } from "react";
import dataJson from "./generated-data.json";

type Confidence = "High" | "Medium" | "Low";
type Tab = "compare" | "portfolio";

type HistoricalItem = {
  id: string;
  season: string;
  itemType: string;
  style: string;
  colourCode: string;
  design: string;
  categoryType: string;
  fabric: string;
  colour: string;
  order: number;
  dispatch: number;
  sales: number;
  sellThrough: number;
  imageUrl?: string | null;
  hasVisualFeature: boolean;
  salesTarget: number;
  qualityFlags: string[];
  /** Log of the censoring-corrected weekly demand rate; absent on legacy artifacts. */
  weeklyLogRate?: number | null;
  demandCensored?: boolean;
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
  censorSellThrough: number;
  groupShrinkageRows: number;
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
  multiplier: number;
  floor: number;
};

type Match = {
  historicalId: string;
  visualScore: number | null;
  fashionVisualScore: number | null;
  dinoVisualScore: number | null;
  colourVisualScore: number | null;
  textureVisualScore: number | null;
  hybridScore: number;
};

type UpcomingItem = {
  id: string;
  itemType: string;
  style: string;
  colourCode: string;
  design: string;
  categoryType: string;
  fabric: string;
  season: string;
  colour: string;
  imageUrl?: string | null;
  hasVisualFeature: boolean;
  matches: Match[];
  recommendation: {
    quantity: number;
    low: number;
    high: number;
    confidence: Confidence;
    matchConfidence: Confidence;
    noSuitableMatch: boolean;
    expectedSales: number;
    salesLow: number;
    salesHigh: number;
    analogueSales: number;
    analogueQuantity: number;
    evidencePolicy: string;
    topMatchScore: number;
    modelVersion: string;
  };
  modelFlags: string[];
};

type ComparableProduct = Pick<UpcomingItem,
  "itemType" | "design" | "categoryType" | "fabric" | "colour"
>;

type Dataset = {
  meta: {
    title: string;
    generatedAt: string;
    dataMode: "sample" | "real";
    upcomingSeason: string;
    historicalItems: number;
    upcomingItems: number;
    historicalImageCoverage: number;
    upcomingImageCoverage: number;
    missingUpcomingImages: string[];
    confidenceCounts: Record<Confidence, number>;
    matchConfidenceCounts: Record<Confidence, number>;
    visualMethod: string;
    visionModel: {
      modelId: string;
      modelRevision: string | null;
      embeddingDimension: number;
      device: string;
      historicalCoverage: number;
      upcomingCoverage: number;
      reranker?: {
        modelId: string;
        modelRevision: string | null;
        embeddingDimension: number;
        device: string;
        candidateCount: number;
        weightGrid: number[];
        sameItemTypeConstraint: boolean;
        sameDesignConstraint?: boolean;
        visualOnlyRanking?: boolean;
        patternGate?: {
          enabled: boolean;
          method: string;
          scope: string;
          maximumDistance: number;
          policy: string;
        };
        candidateIndex?: {
          engine: string;
          metric: string;
          fallback: string;
        };
        appearance?: {
          colourDescriptor: {
            space: string;
            method: string;
            paletteSize: number;
            distance: string;
            normalisationScaleDeltaE: number;
            fullImage: boolean;
            region?: string;
          };
          colourGate?: {
            enabled: boolean;
            maximumDistance: number;
            maximumDeltaE?: number;
            policy: string;
          };
          itemTypeOverrides?: Record<string, {
            analysisRegion: string;
            relativeBox: number[];
            usedFor: string[];
            displayedImageModified: boolean;
            patternHardGateDesigns: string[];
          }>;
          textureDescriptor: {
            method: string;
            dimension: number;
          };
          weights: Record<"neural" | "colour" | "texture", number>;
        };
      } | null;
    };
    model: {
      version: string;
      status: string;
      algorithm: string;
      evidencePolicy: string;
      salesPolicy: string;
      orderPolicy: string;
      noMachineLearningForecast: boolean;
      noAttributeMatching: boolean;
      visualOnlyRanking?: boolean;
      dinoRerankWeight?: number;
      minimumVisualScore: number;
      targetSellThrough?: number;
      minimumMatchConfidence: Confidence;
      noMatchPolicy: string;
      topK: number;
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

const dataset = dataJson as unknown as Dataset;
const historyById = new Map(dataset.historical.map((item) => [item.id, item]));
const imageBackedUpcoming = dataset.upcoming.filter((item) => Boolean(item.imageUrl));
const visibleUpcoming =
  imageBackedUpcoming.length > 0 ? imageBackedUpcoming : dataset.upcoming;
const productSegments = Array.from(
  new Set(visibleUpcoming.map((item) => item.itemType)),
).sort();
const visualMatchingAvailable =
  dataset.meta.visionModel.upcomingCoverage > 0 &&
  dataset.meta.visionModel.historicalCoverage > 0;
const numberFormatter = new Intl.NumberFormat("en-IN");

const attributeValueReaders: Record<string, (item: ComparableProduct) => string> = {
  item: (product) => product.itemType,
  design: (product) => product.design,
  category_type: (product) => product.categoryType,
  fabric: (item) => item.fabric,
  colour: (item) => item.colour,
};

const attributeLabels: Record<string, string> = {
  item: "Item",
  design: "Design",
  category_type: "Category Type",
  fabric: "Fabric",
  colour: "Colour",
};

const preferredCatalogAttributeOrder = [
  "colour",
  "design",
  "fabric",
  "category_type",
] as const;
const catalogAttributeOrder = preferredCatalogAttributeOrder;

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

const buyCeilings = dataset.meta.model.buyCeilings;

function ceilingFor(item: ComparableProduct): number {
  if (!buyCeilings) return FALLBACK_MAX_BUY;
  // Rounded once here (matching model.py) so a float ceiling that isn't
  // perfectly integral can't disagree with itself between the clamp and any
  // reported value derived from it.
  return Math.round(buyCeilings.byItemType[item.itemType] ?? buyCeilings.globalCeiling);
}

/**
 * The single number shown to a planner as "Sales prediction" everywhere on
 * screen. For a predictive forecast this is the median (the typical, most
 * likely outcome) rather than the mean — showing both side by side, as an
 * earlier version of this card did, reads as two conflicting answers to the
 * same question. The mean stays available as `expectedSales` for anyone
 * doing portfolio-level arithmetic (it's the statistically correct number to
 * sum across many products; the median is not), but a single product's card
 * only needs one number, and the one a human means by "how many will sell"
 * is the typical case, not an average pulled up by tail risk.
 */
function headlineDemand(decision: Decision): number {
  return decision.forecast?.medianDemand ?? decision.expectedSales;
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

const demandModel = dataset.meta.model.demandModel;

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
  skewRatio: number;
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
  analogueWeight: number;
  logSigma: number;
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
    skewRatio,
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
    analogueWeight: shrinkage,
    logSigma,
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
  const apiBase = (
    process.env.NEXT_PUBLIC_TURTLE_API_URL ?? "http://localhost:8080"
  ).replace(/\/$/, "");
  const resolvedSrc = src?.startsWith("/v1/product-images/")
    ? src.replace("/v1/product-images/", "/product-images/")
    : src?.startsWith("/product-images/")
      ? src
      : src?.startsWith("/")
        ? `${apiBase}${src}`
        : src;
  const failed = Boolean(resolvedSrc && failedSrc === resolvedSrc);

  if (!resolvedSrc || failed) {
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
      src={resolvedSrc}
      alt={alt}
      loading={eager ? "eager" : "lazy"}
      referrerPolicy="no-referrer"
      onError={() => setFailedSrc(resolvedSrc)}
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

function App() {
  const initialItem = visibleUpcoming.find(
    (item) => item.recommendation.matchConfidence === "High" && item.imageUrl,
  ) ?? visibleUpcoming[0];
  const [tab, setTab] = useState<Tab>("compare");
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

  const selected = visibleUpcoming.find((item) => item.id === selectedId) ?? initialItem;
  const decision = useMemo(
    () => makeDecision(
      selected,
      minimumSimilarity,
      targetSellThrough,
      focusedHistoricalId,
    ),
    [selected, minimumSimilarity, targetSellThrough, focusedHistoricalId],
  );
  const selectedMatches = decision.eligible.slice(0, 4);

  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(""), 2600);
    return () => window.clearTimeout(timer);
  }, [toast]);

  const portfolio = useMemo(
    () =>
      visibleUpcoming.map((item) => ({
        item,
        decision: makeDecision(item, minimumSimilarity, targetSellThrough),
      })),
    [minimumSimilarity, targetSellThrough],
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
  const isApproved = Boolean(approvedIds[selected.id]);

  const totalOrder = portfolio.reduce(
    (sum, { item, decision: itemDecision }) => sum + (overrides[item.id] ?? itemDecision.quantity),
    0,
  );

  function chooseItem(id: string) {
    setSelectedId(id);
    setFocusedHistoricalId(null);
    setTab("compare");
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

    const rows = [
      [
        "Upcoming item",
        "Item",
        "Design",
        "Colour",
        "Category Type",
        "Product match status",
        "Top historical match",
        "Visual AI similarity (overall)",
        "Colour similarity",
        "Pattern similarity",
        "Style similarity",
        "Texture similarity",
        "Match confidence",
        "Forecast demand (typical)",
        "Forecast demand (average)",
        "Forecast low (p10)",
        "Forecast high (p90)",
        "Analogue actual sales",
        "Analogue original order",
        "AI recommended quantity",
        // Audit trail rather than an on-card warning: a capped order is a
        // policy limit, not a target the model actually reached, and a capped
        // range tail understates upside. Neither belongs in the planner's
        // face, but both belong in offline analysis.
        "Buy ceiling",
        "Order hit ceiling",
        "Range tail hit ceiling",
        "Planner quantity",
        "Approval status",
      ],
      ...portfolio.map(({ item, decision: itemDecision }) => {
        const top = itemDecision.noSuitableMatch
          ? undefined
          : itemDecision.ranked[0];
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
          headlineDemand(itemDecision),
          itemDecision.expectedSales,
          itemDecision.salesLow,
          itemDecision.salesHigh,
          itemDecision.analogueSales,
          itemDecision.analogueQuantity,
          itemDecision.quantity,
          itemDecision.buyCeiling,
          itemDecision.quantityCapped ? "Yes" : "No",
          itemDecision.highCapped ? "Yes" : "No",
          overrides[item.id] ?? "",
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
        <button className="brand" onClick={() => setTab("compare")} aria-label="Open comparison workspace">
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
          ] as [Tab, string][]).map(([key, label]) => (
            <button key={key} className={tab === key ? "active" : ""} onClick={() => setTab(key)}>
              {label}
            </button>
          ))}
        </nav>
        <div className="top-actions">
          <span className="sync-state"><i /> Recommendations up to date</span>
          <span className="avatar" aria-label="Planner profile">SD</span>
        </div>
      </header>

      {tab === "compare" && (
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
                  <div>
                    <span className="card-label">AI order quantity recommendation</span>
                  </div>
                  <div className="signal-pills">
                    {decision.noSuitableMatch
                      ? <NoMatchPill />
                      : <MatchConfidencePill confidence={decision.matchConfidence} detailed />}
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
                        title={`The model wanted more than the ${numberFormatter.format(decision.buyCeiling)}-unit ${selected.itemType} buy ceiling to reach ${Math.round(targetSellThrough * 100)}% expected sell-through. This number is a policy limit, not a target that was actually reached.`}
                      >
                        Capped at max buy
                      </span>
                    )}
                  </div>
                </div>
                <div className="quantity-hero">
                  <div className="quantity-primary">
                    <small>Order prediction</small>
                    <div>
                      <strong>{numberFormatter.format(finalQuantity)}</strong>
                      <span>units</span>
                    </div>
                    <p>
                      {decision.forecast
                        ? `${Math.round(targetSellThrough * 100)}% sell-through · ${numberFormatter.format(decision.low)}–${numberFormatter.format(decision.high)}`
                        : `Selected product's sales ÷ ${Math.round(targetSellThrough * 100)}% target sell-through`}
                    </p>
                  </div>
                  <div className="quantity-secondary">
                    <small>{decision.forecast ? "Sales prediction" : "Matched product sales"}</small>
                    <strong>{numberFormatter.format(headlineDemand(decision))} units</strong>
                    <span>
                      {decision.forecast
                        ? `80%: ${numberFormatter.format(decision.salesLow)}–${numberFormatter.format(decision.salesHigh)} · ${decision.forecast.analoguesUsed} analogue${decision.forecast.analoguesUsed === 1 ? "" : "s"}`
                        : "Cleaned observed sales from the one selected historical product"}
                    </span>
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
                  <p>
                    {decision.noSuitableMatch ? (
                      <>
                        No historical product cleared the visual-match threshold. No sales or order quantity is generated; planner review is required.
                      </>
                    ) : (
                      decision.forecast ? (
                        <>
                          Forecast: <b>{numberFormatter.format(headlineDemand(decision))} units</b> ({numberFormatter.format(decision.salesLow)}–{numberFormatter.format(decision.salesHigh)}, {decision.forecast.analoguesUsed} {decision.forecast.analoguesUsed === 1 ? "analogue" : "analogues"}).{decision.forecast.wideUncertainty ? " Evidence is thin, so treat the range as the estimate." : ""}{" "}
                          {decision.quantityCapped ? (
                            <>
                              Capped at the {selected.itemType} ceiling of <b>{numberFormatter.format(decision.buyCeiling)} units</b>.
                            </>
                          ) : (
                            <>
                              Order <b>{numberFormatter.format(decision.quantity)} units</b> for <b>{Math.round(targetSellThrough * 100)}% sell-through</b>.
                            </>
                          )}
                        </>
                      ) : (
                        <>
                          Matched product sold <b>{numberFormatter.format(decision.expectedSales)} units</b>. At <b>{Math.round(targetSellThrough * 100)}% target sell-through</b>, order <b>{numberFormatter.format(decision.quantity)} units</b>.
                        </>
                      )
                    )}
                  </p>
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
                  <small className="setting-help">
                    {demandModel
                      ? "The buy is solved so expected sales reach this share of it, given the forecast range. Rounded to packs of 25."
                      : "Order prediction = selected product sales ÷ target sell-through, rounded to packs of 25."}
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

      {tab === "portfolio" && (
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
                        <strong>{numberFormatter.format(headlineDemand(itemDecision))}</strong>
                        <small>
                          {itemDecision.noSuitableMatch
                            ? "not generated"
                            : itemDecision.forecast
                              ? `${numberFormatter.format(itemDecision.salesLow)}–${numberFormatter.format(itemDecision.salesHigh)}`
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

      {toast && <div className="toast" role="status">✓ {toast}</div>}
    </main>
  );
}

export default App;
