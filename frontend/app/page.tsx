"use client";

import { useEffect, useMemo, useState } from "react";
import dataJson from "./generated-data.json";

type Confidence = "High" | "Medium" | "Low";
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
  qualityFlags: string[];
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
    imageMappingStatus: string;
    historicalItems: number;
    upcomingItems: number;
    historicalImageCoverage: number;
    upcomingImageCoverage: number;
    missingUpcomingImages: string[];
    confidenceCounts: Record<Confidence, number>;
    matchConfidenceCounts: Record<Confidence, number>;
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
  eligible: RankedMatch[];
  selectedMatch?: RankedMatch;
  quantity: number;
  low: number;
  high: number;
  matchConfidence: Confidence;
  noSuitableMatch: boolean;
  expectedSales: number;
  salesLow: number;
  salesHigh: number;
  analogueSales: number;
  analogueQuantity: number;
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

function packRounded(value: number) {
  return Math.max(0, Math.min(2000, Math.round(value / 25) * 25));
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
  const expectedSales = noSuitableMatch || !historical
    ? 0
    : packRounded(historical.salesTarget);
  const quantity = noSuitableMatch
    ? 0
    : packRounded(expectedSales / Math.max(targetSellThrough, 0.01));
  return {
    ranked,
    eligible,
    selectedMatch,
    quantity,
    low: quantity,
    high: quantity,
    matchConfidence,
    noSuitableMatch,
    expectedSales,
    salesLow: expectedSales,
    salesHigh: expectedSales,
    analogueSales: expectedSales,
    analogueQuantity: noSuitableMatch || !historical ? 0 : packRounded(historical.order),
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

  function chooseAnalogue(historicalId: string) {
    setFocusedHistoricalId(historicalId);
    setOverrides((current) => {
      const next = { ...current };
      delete next[selected.id];
      return next;
    });
  }

  function changeMinimumSimilarity(value: number) {
    setMinimumSimilarity(value);
    setOverrides({});
  }

  function changeTargetSellThrough(value: number) {
    setTargetSellThrough(value);
    setOverrides({});
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
        "Visual AI similarity",
        "Match confidence",
        "Matched product sales",
        "Matched product original order",
        "Planner quantity",
        "Recommendation version",
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
          top?.visualScore === null ? "" : Math.round((top?.visualScore ?? 0) * 100),
          itemDecision.matchConfidence,
          itemDecision.expectedSales,
          itemDecision.quantity,
          overrides[item.id] ?? "",
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
          <span className="sync-state"><i /> Recommendations up to date</span>
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
                    <span className="card-label">AI order quantity recommendation</span>
                  </div>
                  <div className="signal-pills">
                    {decision.noSuitableMatch
                      ? <NoMatchPill />
                      : <MatchConfidencePill confidence={decision.matchConfidence} detailed />}
                  </div>
                </div>
                <div className="quantity-hero">
                  <div className="quantity-primary">
                    <small>Recommended initial order</small>
                    <div>
                      <strong>{numberFormatter.format(finalQuantity)}</strong>
                      <span>units</span>
                    </div>
                    <p>Selected product's sales ÷ {Math.round(targetSellThrough * 100)}% target sell-through</p>
                  </div>
                  <div className="quantity-secondary">
                    <small>Matched product sales</small>
                    <strong>{numberFormatter.format(decision.expectedSales)} units</strong>
                    <span>Cleaned observed sales from the one selected historical product</span>
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
                      <>
                        This single matched product recorded <b>{numberFormatter.format(decision.expectedSales)} cleaned sales units</b>. At a <b>{Math.round(targetSellThrough * 100)}% target sell-through</b>, that implies a starting order of <b>{numberFormatter.format(decision.quantity)} units</b> — confirm with planner judgment below.
                      </>
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
                      <button className="button primary" onClick={() => setToast(`${selected.id} approved at ${numberFormatter.format(finalQuantity)} units`)}>
                        Approve order
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
                  <small className="setting-help">Recommended order = selected product sales ÷ target sell-through, rounded to packs of 25.</small>
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
                    Select one product to calculate the order quantity
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
            <article className="accent"><span>Recommended order</span><strong>{numberFormatter.format(totalOrder)}</strong><small>units across {dataset.meta.upcomingSeason}</small></article>
          </div>
          <div className="portfolio-toolbar">
            <label className="search-box wide"><span>⌕</span><input value={queueSearch} onChange={(event) => setQueueSearch(event.target.value)} placeholder="Search item, design, colour, category type or fabric" /></label>
            <select value={segment} onChange={(event) => setSegment(event.target.value)} aria-label="Filter portfolio category">
              <option>All</option>
              {productSegments.map((itemType) => <option key={itemType}>{itemType}</option>)}
            </select>
            <select value={matchConfidenceFilter} onChange={(event) => setMatchConfidenceFilter(event.target.value)} aria-label="Filter portfolio match confidence"><option value="All">All matches</option><option value="High">High match</option><option value="Medium">Medium match</option><option value="Low">No convincing match</option></select>
            <span>{queueItems.length} results</span>
          </div>
          <div className="portfolio-table-wrap">
            <table className="portfolio-table">
              <thead><tr><th>Upcoming style</th><th>Product attributes</th><th>Top historical analogue</th><th>Match score</th><th>Decision signals</th><th>Expected sales</th><th>Recommended order</th><th>Planner order</th><th><span className="sr-only">Actions</span></th></tr></thead>
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
                          <><strong className="match-score">{scorePercent(top.combinedScore)}</strong><small>100% visual</small></>
                        ) : (
                          <><strong className="no-match-table-label">Below threshold</strong><small>Candidate suppressed</small></>
                        )}
                      </td>
                      <td><div className="table-signals">{itemDecision.noSuitableMatch ? <NoMatchPill /> : <MatchConfidencePill confidence={itemDecision.matchConfidence} detailed />}</div></td>
                      <td><strong>{numberFormatter.format(itemDecision.expectedSales)}</strong><small>selected product actual</small></td>
                      <td><strong>{numberFormatter.format(itemDecision.quantity)}</strong><small>sales ÷ {Math.round(targetSellThrough * 100)}% ST</small></td>
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
            <div><span className="eyebrow">Visual evidence only</span><h1>How model v{dataset.meta.model.version} reaches a recommendation</h1><p>The top four visual analogues are shown for review; exactly one selected product supplies sales evidence for the order quantity.</p></div>
            <div className="poc-badge"><span>AI v{dataset.meta.model.version}</span><small>Real data · {dataset.meta.upcomingSeason}</small></div>
          </div>
          <div className="workflow-grid">
            {[
              ["01", "Audit inputs", "Map both workbook schemas, remove constant or non-comparable fields, link images, and quarantine inconsistent outcome values."],
              ["02", "Retrieve visual analogues", dataset.meta.visionModel.reranker
                ? `FashionSigLIP retrieves the top ${dataset.meta.visionModel.reranker.candidateCount} same-item-type candidates with ${dataset.meta.visionModel.reranker.candidateIndex?.engine ?? "exact vector search"}; dominant garment palettes use perceptual CIEDE2000 distance to reject visible colour mismatches, multi-scale DINO verifies pattern detail, and the remaining candidates are ranked using 100% visual evidence. Every OTTR visual stage uses a fixed waist-to-lower-leg trouser crop that excludes footwear, with hard pattern rejection limited to checks, prints, stripes and dobby/structure.`
                : `Create ${dataset.meta.attributeAudit.activeCount}-field structured evidence and compare mapped product images.`],
              ["03", "Select one product", `Review the top four and use only the clicked analogue when it clears the tunable visual threshold.`],
              ["04", "Use its observed outcome", "Copy cleaned sales and original order from that single historical product. If there is no accepted match, return zero and require manual review."],
            ].map(([number, title, copy]) => <article key={number}><span>{number}</span><h3>{title}</h3><p>{copy}</p></article>)}
          </div>
          <section className="attribute-audit-card">
            <div className="attribute-audit-heading">
              <div>
                <span className="card-label">Workbook attribute audit</span>
              <h2>Workbook fields are not matched</h2>
                <p>{dataset.meta.attributeAudit.policy}</p>
              </div>
              <span className="audit-range">Historical {dataset.meta.attributeAudit.historicalSourceRange}<br />Upcoming {dataset.meta.attributeAudit.upcomingSourceRange}</span>
            </div>
            <div className="active-attribute-grid">
              {dataset.meta.attributeAudit.activeAttributes.map((attribute) => (
                <article key={attribute.key}>
                  <div><b>{attribute.label}</b><strong>Audit only</strong></div>
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
              <h2>Single-analogue policy</h2>
              <div className="formula">
                <p>Match score</p>
                <strong>
                  {visualMatchingAvailable ? "100% Visual AI" : "No recommendation until an image is available"}
                </strong>
              </div>
              <div className="formula">
                <p>Match confidence</p>
                <strong>Single top visual score + image availability + historical outcome quality</strong>
              </div>
              <div className="formula">
                <p>Sales evidence</p>
                <strong>Cleaned observed sales from the single accepted historical visual analogue</strong>
              </div>
              <div className="formula">
                <p>Order evidence</p>
                <strong>Selected analogue sales ÷ target sell-through, rounded to a 25-unit pack</strong>
              </div>
              <div className="formula">
                <p>No-match behavior</p>
                <strong>Zero system quantity and mandatory planner review</strong>
              </div>
            </article>
            <article className="readiness-card">
              <span className="card-label">Policy validation</span>
              <h2>What is enforced</h2>
              <ul>
                <li><b>Top three visual analogues</b> are reviewable, but only one contributes</li>
                <li><b>No attribute score</b> is calculated or blended</li>
                <li><b>No regression model</b> predicts sales or order quantity</li>
                <li><b>{dataset.meta.dataQuality.dispatchAboveOrder + dataset.meta.dataQuality.salesAboveDispatch}</b> order/dispatch/sales anomalies contained by guardrails</li>
              </ul>
              <div className="warning-note">A visual match transfers historical evidence; it is not a statistical demand forecast. Planner approval remains required.</div>
            </article>
          </div>
          <div className="upgrade-table">
            <div><span>Layer</span><b>Real-data pilot now</b><b>Future production roadmap</b></div>
            <div><span>Visual representation</span><p>{dataset.meta.imageMappingStatus}</p><p>Client-tuned visual image/text embedding service; domain tuning awaits reviewed pairs</p></div>
            <div><span>Retrieval</span><p>Same-item-type visual search; top four products retained for selection</p><p>Metadata-filtered vector search followed by reviewed visual re-ranking</p></div>
            <div><span>Quantity logic</span><p>Selected product sales divided by tunable target sell-through</p><p>Optional demand forecasting can be reconsidered only with client approval</p></div>
            <div><span>Decision signals</span><p>Visual match confidence only</p><p>Planner feedback and reviewed match labels</p></div>
            <div><span>Uncertainty</span><p>No statistical range is claimed</p><p>Future ranges require a separately approved forecasting model</p></div>
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
