"use client";

import { useEffect, useMemo, useState } from "react";
import dataJson from "./generated-data.json";

type Confidence = "High" | "Medium" | "Low";
type DemandUncertainty = "Narrow" | "Moderate" | "Wide";
type Tab = "compare" | "portfolio" | "method";

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
  normalizedDemand: number;
  qualityFlags: string[];
};

type Match = {
  historicalId: string;
  attributeScore: number;
  visualScore: number | null;
  fashionVisualScore: number | null;
  dinoVisualScore: number | null;
  colourVisualScore: number | null;
  textureVisualScore: number | null;
  hybridScore: number;
  attributeBreakdown: Record<string, number>;
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
    demandUncertainty: DemandUncertainty;
    uncertaintyRatio: number;
    expectedSales: number;
    salesLow: number;
    salesHigh: number;
    analogueSales: number;
    regressionSales: number;
    salesIntervalHalfWidth: number;
    analogueQuantity: number;
    regressionQuantity: number;
    intervalHalfWidth: number;
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
    imageMappingStatus: string;
    historicalItems: number;
    upcomingItems: number;
    historicalImageCoverage: number;
    upcomingImageCoverage: number;
    missingUpcomingImages: string[];
    confidenceCounts: Record<Confidence, number>;
    matchConfidenceCounts: Record<Confidence, number>;
    demandUncertaintyCounts: Record<DemandUncertainty, number>;
    visualMethod: string;
    attributeAudit: {
      historicalSourceRange: string;
      upcomingSourceRange: string;
      activeCount: number;
      policy: string;
      activeAttributes: Array<{
        key: string;
        label: string;
        historicalColumn: string;
        upcomingColumn: string;
        weight: number;
        historicalUnique: number;
        upcomingUnique: number;
        method: string;
      }>;
      excludedConstants: Array<{
        label: string;
        historicalColumn: string;
        upcomingColumn: string;
        reason: string;
      }>;
      excludedNonComparisonFields: Array<{
        label: string;
        historicalColumn: string;
        upcomingColumn: string;
        reason: string;
      }>;
    };
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
        candidateIndex?: {
          engine: string;
          metric: string;
          fallback: string;
        };
        appearance?: {
          segmentation: {
            enabled: boolean;
            method: string;
            maskedImages: number;
            fallbackImages: number;
            meanForegroundCoverage: number;
            meanMaskConfidence: number;
          };
          colourDescriptor: {
            space: string;
            binsPerChannel: number;
            maskOnly: boolean;
          };
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
      trainingRows: number;
      validationRows: number;
      targetSellThrough: number;
      algorithm: string;
      demandLibrary: string;
      demandPipeline: string;
      modelSelection: string;
      attributeWeights: Record<string, number>;
      attributeWeightGrid: number[];
      attributeWeight: number;
      visualWeight: number;
      dinoRerankWeight?: number;
      minimumVisualScore: number;
      minimumMatchConfidence: Confidence;
      noMatchPolicy: string;
      topK: number;
      regressionBlend: number;
      ridgeAlpha: number;
      salesConformalHalfWidth: number;
      conformalHalfWidth: number;
      evaluation: string;
      interval: string;
      backtest: {
        wape: number;
        mae: number;
        bias: number;
        intervalCoverage: number;
      };
    };
    dataQuality: {
      duplicateHistoricalRowsRemoved: number;
      upcomingWithoutHistoricalItem: number;
      zeroSalesHistoricalRowsExcluded: number;
      upcomingRowsExcludedUnseenItem: number;
      dispatchAboveOrder: number;
      salesAboveDispatch: number;
      sellThroughAbove100: number;
    };
  };
  historical: HistoricalItem[];
  upcoming: UpcomingItem[];
};

type RankedMatch = Match & { combinedScore: number };
type Decision = {
  ranked: RankedMatch[];
  quantity: number;
  low: number;
  high: number;
  matchConfidence: Confidence;
  noSuitableMatch: boolean;
  demandUncertainty: DemandUncertainty;
  uncertaintyRatio: number;
  expectedSales: number;
  salesLow: number;
  salesHigh: number;
  analogueSales: number;
  regressionSales: number;
  salesIntervalHalfWidth: number;
  analogueQuantity: number;
  regressionQuantity: number;
  intervalHalfWidth: number;
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
  "item",
  "category_type",
] as const;
const activeAttributeKeys = new Set(
  dataset.meta.attributeAudit.activeAttributes.map((attribute) => attribute.key),
);
const catalogAttributeOrder = preferredCatalogAttributeOrder.filter(
  (key) => activeAttributeKeys.has(key),
);

function attributeValue(item: ComparableProduct, key: string) {
  return attributeValueReaders[key]?.(item) || "Not provided";
}

function attributeMatchLabel(score: number) {
  if (score >= 0.995) return "Exact match";
  if (score >= 0.62) return "Related match";
  if (score > 0) return "Partial match";
  return "Different";
}

function attributeMatchTone(score: number) {
  if (score >= 0.995) return "exact";
  if (score >= 0.62) return "related";
  if (score > 0) return "partial";
  return "different";
}

function clamp(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(maximum, value));
}

function roundPack(value: number, pack = 25) {
  return Math.round(value / pack) * pack;
}

function scorePercent(value: number | null) {
  return value === null ? "N/A" : `${Math.round(value * 100)}%`;
}

function makeDecision(
  item: UpcomingItem,
  attributeWeight: number,
  visualWeight: number,
  targetSellThrough: number,
  topK: number,
): Decision {
  const ranked = item.matches
    .filter((match) => Boolean(historyById.get(match.historicalId)?.imageUrl))
    .map((match) => {
      const visualAvailable = match.visualScore !== null;
      const denominator = visualAvailable
        ? attributeWeight + visualWeight
        : attributeWeight;
      const combinedScore = visualAvailable
        ? (match.attributeScore * attributeWeight +
            (match.visualScore ?? 0) * visualWeight) /
          Math.max(denominator, 1)
        : match.attributeScore;
      return { ...match, combinedScore };
    })
    .sort((left, right) => right.combinedScore - left.combinedScore);

  const top = ranked.slice(0, topK);
  let numerator = 0;
  let denominator = 0;
  top.forEach((match) => {
    const historical = historyById.get(match.historicalId);
    if (!historical) return;
    const weight = Math.max(match.combinedScore, 0.01) ** 2;
    numerator += historical.salesTarget * weight;
    denominator += weight;
  });

  const analogueSales = numerator / Math.max(denominator, 0.01);
  const topScore = top[0]?.combinedScore ?? 0;
  const averageTop =
    top.slice(0, 3).reduce((sum, match) => sum + match.combinedScore, 0) /
    Math.max(top.slice(0, 3).length, 1);
  const salesIntervalHalfWidth = dataset.meta.model.salesConformalHalfWidth *
    (1 + Math.max(0, 0.7 - topScore));
  const issueCount = top.slice(0, 3).reduce((sum, match) => {
    const historical = historyById.get(match.historicalId);
    return sum + (historical?.qualityFlags.length ?? 0);
  }, 0);
  let matchConfidence: Confidence = "Low";
  if (
    topScore >= 0.84 &&
    averageTop >= 0.72 &&
    top[0]?.visualScore !== null &&
    issueCount === 0
  ) {
    matchConfidence = "High";
  } else if (topScore >= 0.62 && averageTop >= 0.52) {
    matchConfidence = "Medium";
  }
  const topVisualScore = top[0]?.visualScore;
  const noSuitableMatch =
    !top.length ||
    matchConfidence === "Low" ||
    topVisualScore === null ||
    topVisualScore === undefined ||
    topVisualScore < dataset.meta.model.minimumVisualScore;
  const regressionSales = item.recommendation.regressionSales;
  const blend = dataset.meta.model.regressionBlend;
  const rawExpectedSales = noSuitableMatch
    ? regressionSales
    : analogueSales * (1 - blend) + regressionSales * blend;
  const expectedSales = clamp(roundPack(rawExpectedSales), 0, 2000);
  const sellThroughPolicy = Math.max(targetSellThrough / 100, 0.01);
  const quantity = clamp(
    roundPack(expectedSales / sellThroughPolicy),
    100,
    2000,
  );
  const uncertaintyRatio = salesIntervalHalfWidth / Math.max(expectedSales, 1);
  const demandUncertainty: DemandUncertainty =
    uncertaintyRatio <= 0.20 ? "Narrow" :
      uncertaintyRatio <= 0.40 ? "Moderate" : "Wide";
  const usesValidatedDefault =
    attributeWeight === Math.round(dataset.meta.model.attributeWeight * 100) &&
    visualWeight === Math.round(dataset.meta.model.visualWeight * 100) &&
    targetSellThrough === Math.round(dataset.meta.model.targetSellThrough * 100) &&
    topK === dataset.meta.model.topK;

  if (usesValidatedDefault) {
    return {
      ranked,
      quantity: item.recommendation.quantity,
      low: item.recommendation.low,
      high: item.recommendation.high,
      matchConfidence: item.recommendation.matchConfidence,
      noSuitableMatch: item.recommendation.noSuitableMatch,
      demandUncertainty: item.recommendation.demandUncertainty,
      uncertaintyRatio: item.recommendation.uncertaintyRatio,
      expectedSales: item.recommendation.expectedSales,
      salesLow: item.recommendation.salesLow,
      salesHigh: item.recommendation.salesHigh,
      analogueSales: item.recommendation.analogueSales,
      regressionSales: item.recommendation.regressionSales,
      salesIntervalHalfWidth: item.recommendation.salesIntervalHalfWidth,
      analogueQuantity: item.recommendation.analogueQuantity,
      regressionQuantity: item.recommendation.regressionQuantity,
      intervalHalfWidth: item.recommendation.intervalHalfWidth,
    };
  }

  return {
    ranked,
    quantity,
    low: clamp(roundPack(Math.max(expectedSales - salesIntervalHalfWidth, 0) / sellThroughPolicy), 100, 2000),
    high: clamp(roundPack((expectedSales + salesIntervalHalfWidth) / sellThroughPolicy), 100, 2000),
    matchConfidence,
    noSuitableMatch,
    demandUncertainty,
    uncertaintyRatio,
    expectedSales,
    salesLow: clamp(roundPack(expectedSales - salesIntervalHalfWidth), 0, 2000),
    salesHigh: clamp(roundPack(expectedSales + salesIntervalHalfWidth), 0, 2000),
    analogueSales: roundPack(analogueSales),
    regressionSales: roundPack(regressionSales),
    salesIntervalHalfWidth: roundPack(salesIntervalHalfWidth),
    analogueQuantity: roundPack(analogueSales / sellThroughPolicy),
    regressionQuantity: roundPack(regressionSales / sellThroughPolicy),
    intervalHalfWidth: roundPack(salesIntervalHalfWidth / sellThroughPolicy),
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
  const [expanded, setExpanded] = useState(false);
  const visibleAttributes = expanded
    ? catalogAttributeOrder
    : catalogAttributeOrder.slice(0, 4);

  return (
    <section className="match-attribute-catalog" aria-label={`${context} product attributes`}>
      <div className="match-attribute-catalog-heading">
        <small>Product attributes</small>
        <button
          type="button"
          className="catalog-attribute-toggle"
          aria-expanded={expanded}
          onClick={() => setExpanded((current) => !current)}
        >
          {expanded ? "Show key attributes" : `View all ${catalogAttributeOrder.length}`}
        </button>
      </div>
      <dl className="catalog-attribute-grid">
        {visibleAttributes.map((key) => {
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

function UncertaintyPill({
  uncertainty,
  detailed = false,
}: {
  uncertainty: DemandUncertainty;
  detailed?: boolean;
}) {
  return (
    <span className={`uncertainty-pill ${uncertainty.toLowerCase()}`}>
      {uncertainty} {detailed ? "sales uncertainty" : "uncertainty"}
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
  const [uncertaintyFilter, setUncertaintyFilter] = useState("All");
  const [attributeWeight, setAttributeWeight] = useState(Math.round(dataset.meta.model.attributeWeight * 100));
  const [visualWeight, setVisualWeight] = useState(Math.round(dataset.meta.model.visualWeight * 100));
  const [targetSellThrough, setTargetSellThrough] = useState(Math.round(dataset.meta.model.targetSellThrough * 100));
  const [topK, setTopK] = useState(dataset.meta.model.topK);
  const [focusedHistoricalId, setFocusedHistoricalId] = useState<string | null>(null);
  const [overrides, setOverrides] = useState<Record<string, number>>({});
  const [toast, setToast] = useState("");

  const selected = visibleUpcoming.find((item) => item.id === selectedId) ?? initialItem;
  const decision = useMemo(
    () => makeDecision(selected, attributeWeight, visualWeight, targetSellThrough, topK),
    [selected, attributeWeight, visualWeight, targetSellThrough, topK],
  );
  const selectedMatches = decision.noSuitableMatch
    ? []
    : decision.ranked.slice(0, topK);

  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(""), 2600);
    return () => window.clearTimeout(timer);
  }, [toast]);

  const portfolio = useMemo(
    () =>
      visibleUpcoming.map((item) => ({
        item,
        decision: makeDecision(item, attributeWeight, visualWeight, targetSellThrough, topK),
      })),
    [attributeWeight, visualWeight, targetSellThrough, topK],
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
        (matchConfidenceFilter === "All" || itemDecision.matchConfidence === matchConfidenceFilter) &&
        (uncertaintyFilter === "All" || itemDecision.demandUncertainty === uncertaintyFilter)
      );
    });
  }, [portfolio, queueSearch, segment, matchConfidenceFilter, uncertaintyFilter]);

  const focusedMatch =
    selectedMatches.find((match) => match.historicalId === focusedHistoricalId) ??
    selectedMatches[0];
  const focusedHistory = focusedMatch ? historyById.get(focusedMatch.historicalId) : undefined;
  const finalQuantity = overrides[selected.id] ?? decision.quantity;

  const totalBuy = portfolio.reduce(
    (sum, { item, decision: itemDecision }) => sum + (overrides[item.id] ?? itemDecision.quantity),
    0,
  );

  function chooseItem(id: string) {
    setSelectedId(id);
    setFocusedHistoricalId(null);
    setTab("compare");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function applyOverride(value: number) {
    if (!Number.isFinite(value) || value < 0) return;
    setOverrides((current) => ({ ...current, [selected.id]: Math.round(value) }));
    setToast(`Planner quantity saved for ${selected.id}`);
  }

  function resetOverride() {
    setOverrides((current) => {
      const next = { ...current };
      delete next[selected.id];
      return next;
    });
    setToast("System recommendation restored");
  }

  function exportCsv() {
    const rows = [
      [
        "Upcoming item",
        "Item",
        "Design",
        "Colour",
        "Category Type",
        "Fabric",
        "Product match status",
        "Top historical match",
        "Combined similarity",
        "Attribute similarity",
        "Visual AI similarity",
        "Match confidence",
        "Sales uncertainty",
        "Uncertainty half-width percentage",
        "Analogue expected sales",
        "Machine-learning expected sales",
        "Expected sales",
        "Expected sales low",
        "Expected sales high",
        "Recommended initial order",
        "Recommended order low",
        "Recommended order high",
        "Planner quantity",
        "Target sell-through",
        "Model version",
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
          item.fabric,
          itemDecision.noSuitableMatch ? "No product match" : "Matched",
          top?.historicalId ?? "",
          Math.round((top?.combinedScore ?? 0) * 100),
          Math.round((top?.attributeScore ?? 0) * 100),
          top?.visualScore === null ? "" : Math.round((top?.visualScore ?? 0) * 100),
          itemDecision.matchConfidence,
          itemDecision.demandUncertainty,
          Math.round(itemDecision.uncertaintyRatio * 100),
          itemDecision.analogueSales,
          itemDecision.regressionSales,
          itemDecision.expectedSales,
          itemDecision.salesLow,
          itemDecision.salesHigh,
          itemDecision.quantity,
          itemDecision.low,
          itemDecision.high,
          overrides[item.id] ?? "",
          targetSellThrough,
          dataset.meta.model.version,
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
            ["method", "Methodology"],
          ] as [Tab, string][]).map(([key, label]) => (
            <button key={key} className={tab === key ? "active" : ""} onClick={() => setTab(key)}>
              {label}
            </button>
          ))}
        </nav>
        <div className="top-actions">
          <span className="sync-state"><i /> AI model v{dataset.meta.model.version} ready</span>
          <button className="button secondary" onClick={exportCsv}>Export CSV</button>
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
              <span>⌕</span>
              <input
                value={queueSearch}
                onChange={(event) => setQueueSearch(event.target.value)}
                placeholder="Search style, colour, fabric"
                aria-label="Search upcoming styles"
              />
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
              <select value={uncertaintyFilter} onChange={(event) => setUncertaintyFilter(event.target.value)} aria-label="Filter sales uncertainty">
                <option value="All">All ranges</option>
                <option value="Narrow">Narrow range</option>
                <option value="Moderate">Moderate range</option>
                <option value="Wide">Wide range</option>
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
                      <span className={`mini-uncertainty ${itemDecision.demandUncertainty.toLowerCase()}`}>
                        {itemDecision.demandUncertainty} range
                      </span>
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
                <span className="eyebrow">Decision workspace / {selected.itemType}</span>
                <h1>{selected.id}</h1>
                <div className="style-tags">
                  <span>{selected.design}</span>
                  <span>{selected.categoryType}</span>
                  <span>{selected.fabric}</span>
                  <span>{selected.colour}</span>
                </div>
              </div>
              <div className="workspace-stepper" aria-label="Decision progress">
                <span className="done">1 <small>Matched</small></span>
                <i />
                <span className="current">2 <small>Review</small></span>
                <i />
                <span>3 <small>Approve</small></span>
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
                    <span className="card-label">AI buy recommendation</span>
                    <p>
                      {visualMatchingAvailable
                        ? "Visual AI retrieval + trained unit-sales forecasting"
                        : "Real-data attribute retrieval + trained unit-sales forecasting"}
                    </p>
                  </div>
                  <div className="signal-pills">
                    {decision.noSuitableMatch
                      ? <NoMatchPill />
                      : <MatchConfidencePill confidence={decision.matchConfidence} detailed />}
                    <UncertaintyPill uncertainty={decision.demandUncertainty} detailed />
                  </div>
                </div>
                <div className="quantity-hero">
                  <div className="quantity-primary">
                    <small>Recommended initial order</small>
                    <div>
                      <strong>{numberFormatter.format(finalQuantity)}</strong>
                      <span>units</span>
                    </div>
                    <p>Expected sales ÷ {targetSellThrough}% inventory policy</p>
                  </div>
                  <div className="quantity-secondary">
                    <small>Expected customer sales</small>
                    <strong>{numberFormatter.format(decision.expectedSales)} units</strong>
                    <span>80% forecast range {numberFormatter.format(decision.salesLow)}–{numberFormatter.format(decision.salesHigh)}</span>
                  </div>
                </div>
                <div className="match-confidence-track" aria-label={`Top historical match score ${scorePercent(decision.ranked[0]?.combinedScore ?? 0)}`}>
                  <span style={{ width: `${Math.round((decision.ranked[0]?.combinedScore ?? 0) * 100)}%` }} />
                </div>
                <div className="rationale-box">
                  <span className="rationale-icon">✦</span>
                  <p>
                    {decision.noSuitableMatch ? (
                      <>
                        No historical product cleared the visual-match guardrail.
                        The <b>{numberFormatter.format(decision.expectedSales)} unit sales forecast</b> therefore
                        uses the trained product-attribute model without analogue blending.
                      </>
                    ) : (
                      <>
                        The model forecasts <b>{numberFormatter.format(decision.expectedSales)} sales units</b> independently of inventory policy. The {targetSellThrough}% target converts that forecast into the recommended initial order; changing it does not retrain or alter expected sales.
                      </>
                    )}
                  </p>
                </div>
                <div className="recommendation-metrics">
                  <div>
                    <small>Top historical analogue</small>
                    <strong>{decision.noSuitableMatch ? "No product match" : decision.ranked[0]?.historicalId}</strong>
                  </div>
                  <div>
                    <small>Analogue sales forecast</small>
                    <strong>{decision.noSuitableMatch ? "Not used" : `${numberFormatter.format(decision.analogueSales)} units`}</strong>
                  </div>
                  <div><small>Machine-learning sales forecast</small><strong>{numberFormatter.format(decision.regressionSales)} units</strong></div>
                  <div><small>Sales backtest WAPE</small><strong>{scorePercent(dataset.meta.model.backtest.wape)}</strong></div>
                </div>
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
                  <button className="button primary" onClick={() => setToast(`${selected.id} approved at ${numberFormatter.format(finalQuantity)} units`)}>
                    Approve buy
                  </button>
                  {overrides[selected.id] && <button className="text-button" onClick={resetOverride}>Reset</button>}
                </div>
              </article>

              <aside className="settings-card">
                <div className="card-label-row">
                  <span className="card-label">Decision settings</span>
                  <span className="live-chip">Live</span>
                </div>
                <label className="range-control">
                  <span><b>Attribute weight</b><strong>{attributeWeight}%</strong></span>
                  <input type="range" min={visualMatchingAvailable ? 10 : 100} max={visualMatchingAvailable ? 90 : 100} value={attributeWeight} disabled={!visualMatchingAvailable} onChange={(event) => {
                    const value = Number(event.target.value);
                    setAttributeWeight(value);
                    setVisualWeight(100 - value);
                  }} />
                </label>
                <label className="range-control">
                  <span><b>Visual weight</b><strong>{visualWeight}%</strong></span>
                  <input type="range" min={visualMatchingAvailable ? 10 : 0} max={visualMatchingAvailable ? 90 : 0} value={visualWeight} disabled={!visualMatchingAvailable} onChange={(event) => {
                    const value = Number(event.target.value);
                    setVisualWeight(value);
                    setAttributeWeight(100 - value);
                  }} />
                </label>
                <label className="range-control">
                  <span><b>Inventory strategy</b><strong>{targetSellThrough}% ST</strong></span>
                  <input type="range" min="50" max="90" value={targetSellThrough} onChange={(event) => setTargetSellThrough(Number(event.target.value))} />
                  <small className="setting-help">Adjusts the recommended order, not the AI sales forecast.</small>
                </label>
                <label className="select-control analogue-count-control">
                  <span>
                    <b>Products used in recommendation</b>
                    <strong>
                      {decision.noSuitableMatch
                        ? "0 — visual guardrail active"
                        : topK === dataset.meta.model.topK
                          ? "Validated default"
                          : "Custom scenario"}
                    </strong>
                  </span>
                  <select aria-label="Products used in recommendation" value={topK} onChange={(event) => {
                    setTopK(Number(event.target.value));
                    setFocusedHistoricalId(null);
                  }}>
                    <option value="3">3 closest products</option>
                    <option value="5">5 closest products</option>
                    <option value="8">8 closest products</option>
                  </select>
                  <small>
                    {decision.noSuitableMatch
                      ? "Weak candidates are excluded; the forecast does not use a historical analogue."
                      : "Every selected product is shown below and contributes to the quantity calculation."}
                  </small>
                </label>
                <div className="method-note">
                  <span>{visualMatchingAvailable ? "Visual AI + demand intelligence" : "Attribute + demand intelligence"}</span>
                  <p>
                    {visualMatchingAvailable
                      ? "Fashion-domain visual embeddings, validation-selected matching, a trained sales model, and data-calibrated uncertainty."
                      : "Seven real-data product attributes, temporal validation, a trained sales model, and data-calibrated uncertainty. Image matching activates after identifier mapping."}
                  </p>
                </div>
              </aside>
            </div>

            <section className="matches-section">
              <div className="section-heading">
                <div>
                  <span className="eyebrow">
                    {decision.noSuitableMatch
                      ? "Visual match guardrail"
                      : "Ranked historical analogues"}
                  </span>
                  <h2>
                    {decision.noSuitableMatch
                      ? "No convincing product match found"
                      : "Why these styles are relevant"}
                  </h2>
                  <p className="section-supporting-copy">
                    {decision.noSuitableMatch
                      ? `The best visual candidate did not meet the ${scorePercent(dataset.meta.model.minimumVisualScore)} minimum and is not used as sales evidence.`
                      : `Showing all ${topK} products used in this recommendation, selected from ${dataset.meta.historicalItems} eligible historical records.`}
                  </p>
                </div>
                <div className="analogue-header-tools">
                  <span className={`analogue-count-chip ${decision.noSuitableMatch ? "rejected" : ""}`}>
                    {decision.noSuitableMatch ? "0 used" : `${topK} used`}
                  </span>
                  <div className="score-legend">
                    <span><i className="attr" /> Attribute</span>
                    {visualMatchingAvailable && <span><i className="visual" /> Visual AI</span>}
                  </div>
                </div>
              </div>
              {decision.noSuitableMatch ? (
                <div className="no-product-match-state" role="status">
                  <span aria-hidden="true">∅</span>
                  <div>
                    <strong>No historical product is shown</strong>
                    <p>
                      The nearest candidates remain below the visual confidence
                      threshold. The buy forecast uses the trained
                      product-attribute model and a wide uncertainty range.
                    </p>
                  </div>
                </div>
              ) : (
                <div className={`match-grid match-grid-${topK}`}>
                  {selectedMatches.map((match, index) => {
                    const historical = historyById.get(match.historicalId);
                    if (!historical) return null;
                    return (
                      <article
                        key={match.historicalId}
                        className={`match-card ${focusedMatch?.historicalId === match.historicalId ? "active" : ""}`}
                      >
                        <button
                          type="button"
                          className="match-card-select"
                          aria-label={`Select ${historical.id} for match evidence`}
                          onClick={() => setFocusedHistoricalId(match.historicalId)}
                        >
                          <span className="rank">#{index + 1}</span>
                          <ProductImage src={historical.imageUrl} alt={historical.id} className="match-image" />
                          <div className="match-copy">
                            <div className="match-title"><strong>{historical.id}</strong><span>{scorePercent(match.combinedScore)}</span></div>
                            <small>{historical.season} · {historical.design} · {historical.colour}</small>
                            <div className="dual-bars">
                              <span><i style={{ width: `${match.attributeScore * 100}%` }} /></span>
                              <span><i style={{ width: `${(match.visualScore ?? 0) * 100}%` }} /></span>
                            </div>
                          </div>
                        </button>
                        <div className="match-card-catalog">
                          <MatchAttributeCatalog product={historical} context="Historical" />
                          <div className="match-performance">
                            <span><small>Order</small><strong>{numberFormatter.format(historical.order)}</strong></span>
                            <span><small>Sell-through</small><strong>{scorePercent(historical.sellThrough)}</strong></span>
                          </div>
                        </div>
                      </article>
                    );
                  })}
                </div>
              )}
            </section>

            {focusedHistory && focusedMatch && (
              <section className="evidence-panel" aria-label={`${selected.id} compared with ${focusedHistory.id}`}>
                <div className="evidence-header">
                  <div className="evidence-intro">
                    <span className="eyebrow">Match evidence</span>
                    <h2>Why this analogue matched</h2>
                    <p>
                      {visualMatchingAvailable
                        ? "The overall similarity combines structured product attributes with deep visual similarity. Review every scored field below before using this historical style as sales evidence."
                        : "The current similarity uses the seven comparable fields in the real workbooks. Review every scored field below before using this historical style as sales evidence."}
                    </p>
                    <div className="evidence-product-pair">
                      <div className="evidence-product-card">
                        <ProductImage src={selected.imageUrl} alt={`Upcoming ${selected.id}`} className="evidence-product-image" />
                        <span>
                          <small>Upcoming style</small>
                          <strong>{selected.id}</strong>
                          <b>{selected.design} · {selected.colour}</b>
                        </span>
                      </div>
                      <i className="evidence-pair-arrow" aria-hidden="true">↔</i>
                      <div className="evidence-product-card">
                        <ProductImage src={focusedHistory.imageUrl} alt={`Historical ${focusedHistory.id}`} className="evidence-product-image" />
                        <span>
                          <small>Historical analogue</small>
                          <strong>{focusedHistory.id}</strong>
                          <b>{focusedHistory.design} · {focusedHistory.colour}</b>
                        </span>
                      </div>
                    </div>
                  </div>
                  <aside className="evidence-summary" aria-label="Similarity score summary">
                    <div className="evidence-summary-heading">
                      <span>Similarity scores</span>
                      <small>Higher is closer</small>
                    </div>
                    <div className="evidence-scores">
                      <ScoreRing score={focusedMatch.combinedScore} label="Overall" />
                      <ScoreRing score={focusedMatch.attributeScore} label="Attributes" />
                      {visualMatchingAvailable && <ScoreRing score={focusedMatch.visualScore ?? 0} label="Image" />}
                    </div>
                    <p>
                      {visualMatchingAvailable
                        ? "The image score combines FashionSigLIP, DINOv2, masked garment colour and texture evidence; the overall score uses the selected attribute and visual weights."
                        : "Image scoring is disabled because the supplied filenames do not map to the SS27 identifiers; the overall score is attribute-only."}
                    </p>
                  </aside>
                </div>
                <div className="evidence-attribute-heading">
                  <div>
                    <span>Attribute-by-attribute comparison</span>
                    <p>Upcoming value versus historical value across all {Object.keys(focusedMatch.attributeBreakdown).length} active matching fields.</p>
                  </div>
                  <small>Exact, related, partial or different</small>
                </div>
                <div className="attribute-evidence">
                  {Object.entries(focusedMatch.attributeBreakdown).map(([key, value]) => {
                    const upcomingValue = attributeValue(selected, key);
                    const historicalValue = attributeValue(focusedHistory, key);
                    return (
                      <div className="attribute-comparison" key={key}>
                        <div className="attribute-comparison-heading">
                          <span>{attributeLabels[key] ?? key}</span>
                          <strong className={`attribute-match-status ${attributeMatchTone(value)}`}>
                            {attributeMatchLabel(value)} <small>{scorePercent(value)}</small>
                          </strong>
                        </div>
                        <div className="attribute-value-pair">
                          <span title={upcomingValue}><small>Upcoming</small><b>{upcomingValue}</b></span>
                          <i aria-hidden="true">→</i>
                          <span title={historicalValue}><small>Historical</small><b>{historicalValue}</b></span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </section>
            )}
          </section>
        </div>
      )}

      {tab === "portfolio" && (
        <section className="portfolio-page page-wrap">
          <div className="page-heading">
            <div><span className="eyebrow">Upcoming assortment</span><h1>Portfolio recommendation</h1><p>Review expected sales, analogue evidence, uncertainty and recommended initial orders across the complete {dataset.meta.upcomingSeason} workbook.</p></div>
            <button className="button primary" onClick={exportCsv}>Export recommendation file</button>
          </div>
          <div className="kpi-grid">
            <article><span>Total styles</span><strong>{dataset.meta.upcomingItems}</strong><small>{dataset.meta.upcomingImageCoverage} with images</small></article>
            <article>
              <span>Visual AI coverage</span>
              <strong>{visualMatchingAvailable ? scorePercent(dataset.meta.upcomingImageCoverage / dataset.meta.upcomingItems) : "Pending mapping"}</strong>
              <small>{visualMatchingAvailable ? `${dataset.meta.missingUpcomingImages.length} linked-image exceptions` : "Attribute-only recommendations active"}</small>
            </article>
            <article><span>Temporal holdout WAPE</span><strong>{scorePercent(dataset.meta.model.backtest.wape)}</strong><small>MAE {numberFormatter.format(dataset.meta.model.backtest.mae)} units</small></article>
            <article className="accent"><span>Recommended buy</span><strong>{numberFormatter.format(totalBuy)}</strong><small>units across {dataset.meta.upcomingSeason}</small></article>
          </div>
          <div className="portfolio-toolbar">
            <label className="search-box wide"><span>⌕</span><input value={queueSearch} onChange={(event) => setQueueSearch(event.target.value)} placeholder="Search item, design, colour, category type or fabric" /></label>
            <select value={segment} onChange={(event) => setSegment(event.target.value)} aria-label="Filter portfolio category">
              <option>All</option>
              {productSegments.map((itemType) => <option key={itemType}>{itemType}</option>)}
            </select>
            <select value={matchConfidenceFilter} onChange={(event) => setMatchConfidenceFilter(event.target.value)} aria-label="Filter portfolio match confidence"><option value="All">All matches</option><option value="High">High match</option><option value="Medium">Medium match</option><option value="Low">No convincing match</option></select>
            <select value={uncertaintyFilter} onChange={(event) => setUncertaintyFilter(event.target.value)} aria-label="Filter portfolio sales uncertainty"><option value="All">All ranges</option><option value="Narrow">Narrow range</option><option value="Moderate">Moderate range</option><option value="Wide">Wide range</option></select>
            <span>{queueItems.length} results</span>
          </div>
          <div className="portfolio-table-wrap">
            <table className="portfolio-table">
              <thead><tr><th>Upcoming style</th><th>Product attributes</th><th>Top historical analogue</th><th>Match score</th><th>Decision signals</th><th>Expected sales</th><th>Recommended buy</th><th>Planner buy</th><th><span className="sr-only">Actions</span></th></tr></thead>
              <tbody>
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
                          <><strong className="match-score">{scorePercent(top.combinedScore)}</strong><small>Attr {scorePercent(top.attributeScore)} · Visual {scorePercent(top.visualScore)}</small></>
                        ) : (
                          <><strong className="no-match-table-label">Below threshold</strong><small>Candidate suppressed</small></>
                        )}
                      </td>
                      <td><div className="table-signals">{itemDecision.noSuitableMatch ? <NoMatchPill /> : <MatchConfidencePill confidence={itemDecision.matchConfidence} detailed />}<UncertaintyPill uncertainty={itemDecision.demandUncertainty} detailed /></div></td>
                      <td><strong>{numberFormatter.format(itemDecision.expectedSales)}</strong><small>{numberFormatter.format(itemDecision.salesLow)}–{numberFormatter.format(itemDecision.salesHigh)} forecast</small></td>
                      <td><strong>{numberFormatter.format(itemDecision.quantity)}</strong><small>{numberFormatter.format(itemDecision.low)}–{numberFormatter.format(itemDecision.high)}</small></td>
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

      {tab === "method" && (
        <section className="method-page page-wrap">
          <div className="page-heading method-heading">
            <div><span className="eyebrow">AI with measurable evidence</span><h1>How model v{dataset.meta.model.version} reaches a recommendation</h1><p>A real-workbook product-matching and learned sales-forecasting workflow with temporal validation, uncertainty, data guardrails, and planner control.</p></div>
            <div className="poc-badge"><span>AI v{dataset.meta.model.version}</span><small>Real data · {dataset.meta.upcomingSeason}</small></div>
          </div>
          <div className="workflow-grid">
            {[
              ["01", "Audit inputs", "Map both workbook schemas, remove constant or non-comparable fields, link images, and quarantine inconsistent outcome values."],
              ["02", "Retrieve visual analogues", dataset.meta.visionModel.reranker
                ? `FashionSigLIP retrieves the top ${dataset.meta.visionModel.reranker.candidateCount} same-item candidates with ${dataset.meta.visionModel.reranker.candidateIndex?.engine ?? "exact vector search"}; DINOv2, garment-masked CIELAB colour, and texture rerank visual detail before structured evidence is combined.`
                : `Create ${dataset.meta.attributeAudit.activeCount}-field structured evidence and compare mapped product images.`],
              ["03", "Learn retrieval", "Use a forward season holdout and parameter search to tune neighbour count, forecast blend, and regularization."],
              ["04", "Forecast unit sales", "Ensemble similarity-weighted historical sales with a trained, regularized machine-learning forecast."],
              ["05", "Convert forecast to a buy", "Keep expected sales independent, apply the sell-through inventory policy, then enforce pack and quantity limits."],
            ].map(([number, title, copy]) => <article key={number}><span>{number}</span><h3>{title}</h3><p>{copy}</p></article>)}
          </div>
          <section className="attribute-audit-card">
            <div className="attribute-audit-heading">
              <div>
                <span className="card-label">Workbook attribute audit</span>
                <h2>{dataset.meta.attributeAudit.activeCount} informative fields retained</h2>
                <p>{dataset.meta.attributeAudit.policy}</p>
              </div>
              <span className="audit-range">Historical {dataset.meta.attributeAudit.historicalSourceRange}<br />Upcoming {dataset.meta.attributeAudit.upcomingSourceRange}</span>
            </div>
            <div className="active-attribute-grid">
              {dataset.meta.attributeAudit.activeAttributes.map((attribute) => (
                <article key={attribute.key}>
                  <div><b>{attribute.label}</b><strong>{scorePercent(attribute.weight)}</strong></div>
                  <p>{attribute.historicalColumn} ↔ {attribute.upcomingColumn}</p>
                  <small>{attribute.method}</small>
                  <span>{attribute.historicalUnique} historical · {attribute.upcomingUnique} upcoming values</span>
                </article>
              ))}
            </div>
            <div className="excluded-attribute-row">
              {dataset.meta.attributeAudit.excludedConstants.map((attribute) => (
                <article key={attribute.label}>
                  <span>Excluded field</span>
                  <b>{attribute.label}</b>
                  <small>{attribute.historicalColumn} ↔ {attribute.upcomingColumn}</small>
                  <p>{attribute.reason}</p>
                </article>
              ))}
            </div>
            <details className="excluded-field-details">
              <summary>Why the other workbook columns are not similarity attributes</summary>
              <div>
                {dataset.meta.attributeAudit.excludedNonComparisonFields.map((field) => (
                  <article key={field.label}>
                    <b>{field.label}</b>
                    <small>{field.historicalColumn} ↔ {field.upcomingColumn}</small>
                    <p>{field.reason}</p>
                  </article>
                ))}
              </div>
            </details>
          </section>
          <div className="method-columns">
            <article className="formula-card">
              <span className="card-label">Decision formula</span>
              <h2>Readable enough to challenge</h2>
              <div className="formula">
                <p>Match score</p>
                <strong>
                  {visualMatchingAvailable
                    ? `${attributeWeight}% × Attribute + ${visualWeight}% × Visual AI`
                    : "100% comparable workbook attributes · Visual AI pending ID mapping"}
                </strong>
              </div>
              <div className="formula">
                <p>Match confidence</p>
                <strong>Top match strength + top-3 consistency + image availability + analogue data quality</strong>
              </div>
              <div className="formula">
                <p>Historical training target</p>
                <strong>Observed unit sales, capped by the strongest supply record only when sales exceed available supply</strong>
              </div>
              <div className="formula">
                <p>Expected sales forecast</p>
                <strong>{Math.round((1 - dataset.meta.model.regressionBlend) * 100)}% analogue sales + {Math.round(dataset.meta.model.regressionBlend * 100)}% machine-learning sales forecast</strong>
              </div>
              <div className="formula">
                <p>Recommended initial order</p>
                <strong>Expected sales ÷ {targetSellThrough}% target sell-through, rounded to 25-unit packs</strong>
              </div>
              <div className="formula">
                <p>Sales uncertainty</p>
                <strong>Narrow ≤ ±20% · Moderate ≤ ±40% · Wide &gt; ±40% of expected sales</strong>
              </div>
              <div className="formula">
                <p>Forecast and order ranges</p>
                <strong>80% conformal sales interval, then sell-through conversion, 25-unit packs and 100–2,000 order guardrails</strong>
              </div>
            </article>
            <article className="readiness-card">
              <span className="card-label">Model validation</span>
              <h2>What is measured</h2>
              <ul>
                <li><b>{scorePercent(dataset.meta.model.backtest.wape)} WAPE</b> in the {dataset.meta.model.validationRows}-row temporal holdout</li>
                <li><b>{numberFormatter.format(dataset.meta.model.backtest.mae)} units</b> mean absolute error</li>
                <li><b>{scorePercent(dataset.meta.model.backtest.intervalCoverage)}</b> empirical interval coverage</li>
                <li><b>{dataset.meta.dataQuality.dispatchAboveOrder + dataset.meta.dataQuality.salesAboveDispatch}</b> order/dispatch/sales anomalies contained by guardrails</li>
              </ul>
              <div className="warning-note">{dataset.meta.model.evaluation}. The model remains a pilot because only two historical seasons are available; three to five clean seasons are still recommended for production approval.</div>
            </article>
          </div>
          <div className="upgrade-table">
            <div><span>Layer</span><b>Real-data pilot now</b><b>Future production roadmap</b></div>
            <div><span>Visual representation</span><p>{dataset.meta.imageMappingStatus}</p><p>Client-tuned visual image/text embedding service; domain tuning awaits reviewed pairs</p></div>
            <div><span>Retrieval</span><p>Attribute scoring against {dataset.meta.historicalItems} historical records; top eight retained per upcoming product</p><p>Metadata-filtered vector search followed by learned re-ranking</p></div>
            <div><span>Quantity logic</span><p>Sales forecast from a validation-tuned analogue and regularized machine-learning ensemble; buy derived from sell-through policy</p><p>Temporal P10/P50/P90 forecasting with hierarchical reconciliation</p></div>
            <div><span>Decision signals</span><p>Separate match confidence and sales uncertainty labels</p><p>Relevance calibration plus category-level temporal forecast uncertainty</p></div>
            <div><span>Uncertainty</span><p>Out-of-fold conformal expected-sales range</p><p>Temporal quantiles calibrated by category, channel and region</p></div>
            <div><span>Workflow</span><p>Browser-session planner override</p><p>Planned durable jobs, feedback capture, model registry and recommendation audit</p></div>
            <div><span>Data</span><p>{dataset.meta.historicalItems} historical / {dataset.meta.upcomingItems} upcoming real records</p><p>3–5 seasons plus inventory and markdown context</p></div>
          </div>
          <section className="scale-platform">
            <div className="scale-platform-heading">
              <div>
                <span className="eyebrow">Future production path / 200K–500K items</span>
                <h2>The database-backed scale platform is intentionally deferred</h2>
                <p>This roadmap can be implemented after the current data and model workflow is validated and the required client history is available.</p>
              </div>
              <span className="scale-badge">Future roadmap</span>
            </div>
            <div className="scale-readiness-grid">
              {[
                ["Future phase", "Vector catalogue", "Managed vector storage, hard metadata filters and nearest-neighbour retrieval."],
                ["Future phase", "Visual embedding service", "Separately deployable deep image/text inference with domain allowlisting, size limits and private-network blocking."],
                ["Needs client labels", "Learning-to-rank", "Gradient-boosted ranking uses planner relevance feedback, outcome reliability and product evidence."],
                ["Needs 3–5 seasons", "Demand forecasting", "Temporal P10/P50/P90 quantile forecasting uses clean stock-out, markdown, channel and hierarchy features."],
                ["Needs calibration", "Hierarchy + risk", "Hierarchical reconciliation keeps category, channel and region totals coherent before order constraints."],
                ["Future phase", "Operations", "MOQ, pack, budget and capacity optimisation plus durable ingestion, batch, feedback and audit records."],
              ].map(([state, title, copy]) => (
                <article key={title} className={state === "Available now" ? "ready" : "waiting"}>
                  <span>{state}</span><h3>{title}</h3><p>{copy}</p>
                </article>
              ))}
            </div>
            <div className="activation-gate">
              <b>Future production activation gate</b>
              <span>A future service should refuse startup until approved ranker and demand-model artifacts are available, preventing a baseline fallback from being presented as trained production AI.</span>
            </div>
          </section>
        </section>
      )}

      {toast && <div className="toast" role="status">✓ {toast}</div>}
    </main>
  );
}

export default App;
