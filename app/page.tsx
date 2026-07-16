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
  sleeve: string;
  provision: string;
  pattern: string;
  range: string;
  fit: string;
  fabric: string;
  fashion: string;
  lifecycle: string;
  colour: string;
  mrp: number;
  order: number;
  dispatch: number;
  sales: number;
  sellThrough: number;
  imageUrl?: string | null;
  hasVisualFeature: boolean;
  normalizedDemand: number;
  qualityFlags: string[];
};

type Match = {
  historicalId: string;
  attributeScore: number;
  visualScore: number | null;
  hybridScore: number;
  attributeBreakdown: Record<string, number>;
};

type UpcomingItem = {
  id: string;
  itemType: string;
  style: string;
  colourCode: string;
  sleeve: string;
  provision: string;
  pattern: string;
  range: string;
  fit: string;
  fabric: string;
  fashion: string;
  lifecycle: string;
  colour: string;
  mrp: number;
  imageUrl?: string | null;
  hasVisualFeature: boolean;
  matches: Match[];
  recommendation: {
    quantity: number;
    low: number;
    high: number;
    confidence: Confidence;
    matchConfidence: Confidence;
    demandUncertainty: DemandUncertainty;
    uncertaintyRatio: number;
    analogueQuantity: number;
    regressionQuantity: number;
    intervalHalfWidth: number;
    topMatchScore: number;
    modelVersion: string;
  };
  modelFlags: string[];
};

type Dataset = {
  meta: {
    title: string;
    generatedAt: string;
    historicalItems: number;
    upcomingItems: number;
    historicalImageCoverage: number;
    upcomingImageCoverage: number;
    missingUpcomingImages: string[];
    confidenceCounts: Record<Confidence, number>;
    matchConfidenceCounts: Record<Confidence, number>;
    demandUncertaintyCounts: Record<DemandUncertainty, number>;
    visualMethod: string;
    visionModel: {
      modelId: string;
      modelRevision: string | null;
      embeddingDimension: number;
      device: string;
      historicalCoverage: number;
      upcomingCoverage: number;
    };
    model: {
      version: string;
      status: string;
      trainingRows: number;
      targetSellThrough: number;
      algorithm: string;
      demandLibrary: string;
      demandPipeline: string;
      modelSelection: string;
      attributeWeightGrid: number[];
      attributeWeight: number;
      visualWeight: number;
      topK: number;
      regressionBlend: number;
      ridgeAlpha: number;
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
  demandUncertainty: DemandUncertainty;
  uncertaintyRatio: number;
  analogueQuantity: number;
  regressionQuantity: number;
  intervalHalfWidth: number;
};

const dataset = dataJson as unknown as Dataset;
const historyById = new Map(dataset.historical.map((item) => [item.id, item]));
const numberFormatter = new Intl.NumberFormat("en-IN");
const moneyFormatter = new Intl.NumberFormat("en-IN", {
  style: "currency",
  currency: "INR",
  maximumFractionDigits: 0,
});

const attributeLabels: Record<string, string> = {
  category: "Category",
  sleeve: "Sleeve",
  provision: "Fit code",
  pattern: "Pattern",
  range: "Range",
  fit: "Collection",
  fabric: "Fabric",
  fashion: "Merch type",
  colour: "Colour",
  price: "Price band",
};

const attributeDriverPriority = [
  "pattern",
  "fabric",
  "colour",
  "range",
  "fit",
  "fashion",
  "price",
  "sleeve",
  "provision",
  "category",
];

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
    const adjusted = historical.normalizedDemand *
      (dataset.meta.model.targetSellThrough / Math.max(targetSellThrough / 100, 0.01));
    const weight = Math.max(match.combinedScore, 0.01) ** 2;
    numerator += adjusted * weight;
    denominator += weight;
  });

  const analogueQuantity = numerator / Math.max(denominator, 0.01);
  const regressionQuantity = item.recommendation.regressionQuantity *
    (dataset.meta.model.targetSellThrough / Math.max(targetSellThrough / 100, 0.01));
  const blend = dataset.meta.model.regressionBlend;
  const rawQuantity = analogueQuantity * (1 - blend) + regressionQuantity * blend;
  const quantity = clamp(roundPack(rawQuantity), 100, 2000);
  const topScore = top[0]?.combinedScore ?? 0;
  const averageTop =
    top.slice(0, 3).reduce((sum, match) => sum + match.combinedScore, 0) /
    Math.max(top.slice(0, 3).length, 1);
  const intervalHalfWidth = dataset.meta.model.conformalHalfWidth *
    (dataset.meta.model.targetSellThrough / Math.max(targetSellThrough / 100, 0.01)) *
    (1 + Math.max(0, 0.7 - topScore));
  const issueCount = top.slice(0, 3).reduce((sum, match) => {
    const historical = historyById.get(match.historicalId);
    return sum + (historical?.qualityFlags.length ?? 0);
  }, 0);
  const uncertaintyRatio = intervalHalfWidth / Math.max(quantity, 1);
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
      demandUncertainty: item.recommendation.demandUncertainty,
      uncertaintyRatio: item.recommendation.uncertaintyRatio,
      analogueQuantity: item.recommendation.analogueQuantity,
      regressionQuantity: item.recommendation.regressionQuantity,
      intervalHalfWidth: item.recommendation.intervalHalfWidth,
    };
  }

  return {
    ranked,
    quantity,
    low: clamp(roundPack(quantity - intervalHalfWidth), 100, 2000),
    high: clamp(roundPack(quantity + intervalHalfWidth), 100, 2000),
    matchConfidence,
    demandUncertainty,
    uncertaintyRatio,
    analogueQuantity: roundPack(analogueQuantity),
    regressionQuantity: roundPack(regressionQuantity),
    intervalHalfWidth: roundPack(intervalHalfWidth),
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
      onError={() => setFailedSrc(src ?? null)}
    />
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

function UncertaintyPill({
  uncertainty,
  detailed = false,
}: {
  uncertainty: DemandUncertainty;
  detailed?: boolean;
}) {
  return (
    <span className={`uncertainty-pill ${uncertainty.toLowerCase()}`}>
      {uncertainty} {detailed ? "demand uncertainty" : "uncertainty"}
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

function matchDriverSummary(match: RankedMatch) {
  const strongDrivers = Object.entries(match.attributeBreakdown)
    .filter(([, value]) => value >= 0.62)
    .sort(([leftKey, leftScore], [rightKey, rightScore]) => (
      rightScore - leftScore
      || attributeDriverPriority.indexOf(leftKey) - attributeDriverPriority.indexOf(rightKey)
    ))
    .map(([key, score]) => ({ key, score, label: attributeLabels[key] ?? key }));

  return {
    visible: strongDrivers.slice(0, 4),
    remaining: Math.max(strongDrivers.length - 4, 0),
    total: strongDrivers.length,
  };
}

function App() {
  const initialItem = dataset.upcoming.find(
    (item) => item.recommendation.matchConfidence === "High" && item.imageUrl,
  ) ?? dataset.upcoming[0];
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

  const selected = dataset.upcoming.find((item) => item.id === selectedId) ?? initialItem;
  const decision = useMemo(
    () => makeDecision(selected, attributeWeight, visualWeight, targetSellThrough, topK),
    [selected, attributeWeight, visualWeight, targetSellThrough, topK],
  );
  const selectedMatches = decision.ranked.slice(0, topK);

  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(""), 2600);
    return () => window.clearTimeout(timer);
  }, [toast]);

  const portfolio = useMemo(
    () =>
      dataset.upcoming.map((item) => ({
        item,
        decision: makeDecision(item, attributeWeight, visualWeight, targetSellThrough, topK),
      })),
    [attributeWeight, visualWeight, targetSellThrough, topK],
  );

  const queueItems = useMemo(() => {
    const query = queueSearch.trim().toUpperCase();
    return portfolio.filter(({ item, decision: itemDecision }) => {
      const searchable = [item.id, item.pattern, item.colour, item.fit, item.fabric]
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
        "Category",
        "Pattern",
        "Colour",
        "MRP",
        "Top historical match",
        "Combined similarity",
        "Attribute similarity",
        "FashionCLIP similarity",
        "Match confidence",
        "Demand uncertainty",
        "Uncertainty half-width percentage",
        "Analogue model quantity",
        "Regularized model quantity",
        "Recommended quantity",
        "Planner quantity",
        "Target sell-through",
        "Model version",
      ],
      ...portfolio.map(({ item, decision: itemDecision }) => {
        const top = itemDecision.ranked[0];
        return [
          item.id,
          item.itemType,
          item.pattern,
          item.colour,
          item.mrp,
          top?.historicalId ?? "",
          Math.round((top?.combinedScore ?? 0) * 100),
          Math.round((top?.attributeScore ?? 0) * 100),
          top?.visualScore === null ? "" : Math.round((top?.visualScore ?? 0) * 100),
          itemDecision.matchConfidence,
          itemDecision.demandUncertainty,
          Math.round(itemDecision.uncertaintyRatio * 100),
          itemDecision.analogueQuantity,
          itemDecision.regressionQuantity,
          itemDecision.quantity,
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
              <span className="season-chip">AW26</span>
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
                <option>OTSH</option>
                <option>OTTS</option>
              </select>
              <select value={matchConfidenceFilter} onChange={(event) => setMatchConfidenceFilter(event.target.value)} aria-label="Filter match confidence">
                <option value="All">All matches</option>
                <option value="High">High match</option>
                <option value="Medium">Medium match</option>
                <option value="Low">Low match</option>
              </select>
              <select value={uncertaintyFilter} onChange={(event) => setUncertaintyFilter(event.target.value)} aria-label="Filter demand uncertainty">
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
                    <small>{item.pattern} · {item.colour}</small>
                    <span className="mini-signals">
                      <span className={`mini-confidence ${itemDecision.matchConfidence.toLowerCase()}`}>
                        {itemDecision.matchConfidence} match
                      </span>
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
                  <span>{selected.pattern}</span>
                  <span>{selected.fit}</span>
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
                <div className="product-specs">
                  <div><small>MRP</small><strong>{moneyFormatter.format(selected.mrp)}</strong></div>
                  <div><small>Sleeve</small><strong>{selected.sleeve}</strong></div>
                  <div><small>Fit code</small><strong>{selected.provision}</strong></div>
                  <div><small>Lifecycle</small><strong>{selected.lifecycle}</strong></div>
                </div>
              </article>

              <article className="recommendation-card">
                <div className="recommendation-topline">
                  <div>
                    <span className="card-label">AI buy recommendation</span>
                    <p>FashionCLIP retrieval + scikit-learn demand model</p>
                  </div>
                  <div className="signal-pills">
                    <MatchConfidencePill confidence={decision.matchConfidence} detailed />
                    <UncertaintyPill uncertainty={decision.demandUncertainty} detailed />
                  </div>
                </div>
                <div className="quantity-hero">
                  <div>
                    <strong>{numberFormatter.format(finalQuantity)}</strong>
                    <span>units</span>
                  </div>
                  <p>80% expected demand range <b>{numberFormatter.format(decision.low)}–{numberFormatter.format(decision.high)}</b></p>
                </div>
                <div className="match-confidence-track" aria-label={`Top historical match score ${scorePercent(decision.ranked[0]?.combinedScore ?? 0)}`}>
                  <span style={{ width: `${Math.round((decision.ranked[0]?.combinedScore ?? 0) * 100)}%` }} />
                </div>
                <div className="rationale-box">
                  <span className="rationale-icon">✦</span>
                  <p>
                    <b>{decision.matchConfidence} match confidence</b> describes analogue relevance. <b>{decision.demandUncertainty.toLowerCase()} demand uncertainty</b> comes from the out-of-fold forecast-error range; the two signals are intentionally separate.
                  </p>
                </div>
                <div className="recommendation-metrics">
                  <div><small>Top historical analogue</small><strong>{decision.ranked[0]?.historicalId}</strong></div>
                  <div><small>Analogue-based demand</small><strong>{numberFormatter.format(decision.analogueQuantity)} units</strong></div>
                  <div><small>Ridge model baseline</small><strong>{numberFormatter.format(decision.regressionQuantity)} units</strong></div>
                  <div><small>Backtest error (WAPE)</small><strong>{scorePercent(dataset.meta.model.backtest.wape)}</strong></div>
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
                  <input type="range" min="10" max="90" value={attributeWeight} onChange={(event) => {
                    const value = Number(event.target.value);
                    setAttributeWeight(value);
                    setVisualWeight(100 - value);
                  }} />
                </label>
                <label className="range-control">
                  <span><b>Visual weight</b><strong>{visualWeight}%</strong></span>
                  <input type="range" min="10" max="90" value={visualWeight} onChange={(event) => {
                    const value = Number(event.target.value);
                    setVisualWeight(value);
                    setAttributeWeight(100 - value);
                  }} />
                </label>
                <label className="range-control">
                  <span><b>Target sell-through</b><strong>{targetSellThrough}%</strong></span>
                  <input type="range" min="50" max="90" value={targetSellThrough} onChange={(event) => setTargetSellThrough(Number(event.target.value))} />
                </label>
                <label className="select-control analogue-count-control">
                  <span>
                    <b>Products used in recommendation</b>
                    <strong>{topK === dataset.meta.model.topK ? "Validated default" : "Custom scenario"}</strong>
                  </span>
                  <select aria-label="Products used in recommendation" value={topK} onChange={(event) => {
                    setTopK(Number(event.target.value));
                    setFocusedHistoricalId(null);
                  }}>
                    <option value="3">3 closest products</option>
                    <option value="5">5 closest products</option>
                    <option value="8">8 closest products</option>
                  </select>
                  <small>Every selected product is shown below and contributes to the quantity calculation.</small>
                </label>
                <div className="method-note">
                  <span>PyTorch FashionCLIP + scikit-learn</span>
                  <p>Fashion-domain image embeddings, validation-selected weights and top-K, a fitted Ridge pipeline, and finite-sample conformal uncertainty.</p>
                </div>
              </aside>
            </div>

            <section className="matches-section">
              <div className="section-heading">
                <div>
                  <span className="eyebrow">Ranked historical analogues</span>
                  <h2>Why these styles are relevant</h2>
                  <p className="section-supporting-copy">Showing all {topK} products used in this recommendation, ranked from {decision.ranked.length} eligible historical styles.</p>
                </div>
                <div className="analogue-header-tools">
                  <span className="analogue-count-chip">{topK} used</span>
                  <div className="score-legend"><span><i className="attr" /> Attribute</span><span><i className="visual" /> FashionCLIP</span></div>
                </div>
              </div>
              <div className={`match-grid match-grid-${topK}`}>
                {selectedMatches.map((match, index) => {
                  const historical = historyById.get(match.historicalId);
                  if (!historical) return null;
                  const driverSummary = matchDriverSummary(match);
                  return (
                    <button
                      key={match.historicalId}
                      className={`match-card ${focusedMatch?.historicalId === match.historicalId ? "active" : ""}`}
                      onClick={() => setFocusedHistoricalId(match.historicalId)}
                    >
                      <span className="rank">#{index + 1}</span>
                      <ProductImage src={historical.imageUrl} alt={historical.id} className="match-image" />
                      <div className="match-copy">
                        <div className="match-title"><strong>{historical.id}</strong><span>{scorePercent(match.combinedScore)}</span></div>
                        <small>{historical.season} · {historical.pattern} · {historical.colour}</small>
                        <div className="dual-bars">
                          <span><i style={{ width: `${match.attributeScore * 100}%` }} /></span>
                          <span><i style={{ width: `${(match.visualScore ?? 0) * 100}%` }} /></span>
                        </div>
                        <small className="match-reasons-label">Top attribute drivers</small>
                        <div
                          className="match-reasons"
                          aria-label={`${driverSummary.total} strong attribute matches. Select this analogue to inspect every attribute score.`}
                        >
                          {driverSummary.visible.map(({ key, label, score }) => (
                            <span key={key} title={`${label}: ${scorePercent(score)}`}>{label} {scorePercent(score)}</span>
                          ))}
                          {driverSummary.remaining > 0 && <span className="more">+{driverSummary.remaining} more</span>}
                        </div>
                        <div className="match-performance">
                          <span><small>Order</small>{numberFormatter.format(historical.order)}</span>
                          <span><small>Sell-through</small>{scorePercent(historical.sellThrough)}</span>
                        </div>
                      </div>
                    </button>
                  );
                })}
              </div>
            </section>

            {focusedHistory && focusedMatch && (
              <section className="evidence-panel">
                <div className="evidence-intro">
                  <span className="eyebrow">Match evidence</span>
                  <h2>{selected.id} ↔ {focusedHistory.id}</h2>
                  <p>Each card summarizes four prioritized drivers; all {Object.keys(focusedMatch.attributeBreakdown).length} attribute scores are shown here. Select any of the {topK} included analogues above to inspect its complete evidence.</p>
                </div>
                <div className="evidence-scores">
                  <ScoreRing score={focusedMatch.combinedScore} label="Combined" />
                  <ScoreRing score={focusedMatch.attributeScore} label="Attributes" />
                  <ScoreRing score={focusedMatch.visualScore ?? 0} label="FashionCLIP" />
                </div>
                <div className="attribute-evidence">
                  {Object.entries(focusedMatch.attributeBreakdown).map(([key, value]) => (
                    <div key={key}>
                      <span>{attributeLabels[key] ?? key}</span>
                      <i><b style={{ width: `${value * 100}%` }} /></i>
                      <strong>{scorePercent(value)}</strong>
                    </div>
                  ))}
                </div>
              </section>
            )}
          </section>
        </div>
      )}

      {tab === "portfolio" && (
        <section className="portfolio-page page-wrap">
          <div className="page-heading">
            <div><span className="eyebrow">Upcoming assortment</span><h1>Portfolio recommendation</h1><p>Review analogue match confidence, demand uncertainty and buy quantities across the full sample.</p></div>
            <button className="button primary" onClick={exportCsv}>Export recommendation file</button>
          </div>
          <div className="kpi-grid">
            <article><span>Total styles</span><strong>{dataset.meta.upcomingItems}</strong><small>{dataset.meta.upcomingImageCoverage} with images</small></article>
            <article><span>FashionCLIP coverage</span><strong>{scorePercent(dataset.meta.upcomingImageCoverage / dataset.meta.upcomingItems)}</strong><small>{dataset.meta.missingUpcomingImages.length} linked-image exceptions</small></article>
            <article><span>LOO backtest WAPE</span><strong>{scorePercent(dataset.meta.model.backtest.wape)}</strong><small>MAE {numberFormatter.format(dataset.meta.model.backtest.mae)} units</small></article>
            <article className="accent"><span>Recommended buy</span><strong>{numberFormatter.format(totalBuy)}</strong><small>units across sample</small></article>
          </div>
          <div className="portfolio-toolbar">
            <label className="search-box wide"><span>⌕</span><input value={queueSearch} onChange={(event) => setQueueSearch(event.target.value)} placeholder="Search item, pattern, colour or fabric" /></label>
            <select value={segment} onChange={(event) => setSegment(event.target.value)} aria-label="Filter portfolio category"><option>All</option><option>OTSH</option><option>OTTS</option></select>
            <select value={matchConfidenceFilter} onChange={(event) => setMatchConfidenceFilter(event.target.value)} aria-label="Filter portfolio match confidence"><option value="All">All matches</option><option value="High">High match</option><option value="Medium">Medium match</option><option value="Low">Low match</option></select>
            <select value={uncertaintyFilter} onChange={(event) => setUncertaintyFilter(event.target.value)} aria-label="Filter portfolio demand uncertainty"><option value="All">All ranges</option><option value="Narrow">Narrow range</option><option value="Moderate">Moderate range</option><option value="Wide">Wide range</option></select>
            <span>{queueItems.length} results</span>
          </div>
          <div className="portfolio-table-wrap">
            <table className="portfolio-table">
              <thead><tr><th>Upcoming style</th><th>Product attributes</th><th>Top historical analogue</th><th>Match score</th><th>Decision signals</th><th>Recommended buy</th><th>Planner buy</th><th><span className="sr-only">Actions</span></th></tr></thead>
              <tbody>
                {queueItems.map(({ item, decision: itemDecision }) => {
                  const top = itemDecision.ranked[0];
                  const historical = top ? historyById.get(top.historicalId) : undefined;
                  return (
                    <tr key={item.id}>
                      <td><div className="table-product"><ProductImage src={item.imageUrl} alt={item.id} className="table-image" /><span><strong>{item.id}</strong><small>{item.colour} · {moneyFormatter.format(item.mrp)}</small></span></div></td>
                      <td><strong>{item.pattern}</strong><small>{item.fit} · {item.fabric}</small></td>
                      <td>{historical && <div className="table-product"><ProductImage src={historical.imageUrl} alt={historical.id} className="table-image" /><span><strong>{historical.id}</strong><small>{historical.season} · ST {scorePercent(historical.sellThrough)}</small></span></div>}</td>
                      <td><strong className="match-score">{scorePercent(top?.combinedScore ?? 0)}</strong><small>Attr {scorePercent(top?.attributeScore ?? 0)} · Visual {scorePercent(top?.visualScore ?? 0)}</small></td>
                      <td><div className="table-signals"><MatchConfidencePill confidence={itemDecision.matchConfidence} detailed /><UncertaintyPill uncertainty={itemDecision.demandUncertainty} detailed /></div></td>
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
            <div><span className="eyebrow">AI with measurable evidence</span><h1>How model v{dataset.meta.model.version} reaches a recommendation</h1><p>A PyTorch/Transformers FashionCLIP and scikit-learn demand workflow with validation, uncertainty, data guardrails, and planner control.</p></div>
            <div className="poc-badge"><span>AI v{dataset.meta.model.version}</span><small>Dedicated ML libraries</small></div>
          </div>
          <div className="workflow-grid">
            {[
              ["01", "Validate inputs", "Normalize identifiers and attributes, link images, and quarantine inconsistent order, dispatch, sales, and sell-through values."],
              ["02", "Encode products", "Create structured attribute evidence and 512-dimensional FashionCLIP embeddings from garment images."],
              ["03", "Learn retrieval", "Use scikit-learn ParameterGrid and LeaveOneOut to tune attribute/vision weights, neighbour count, blend, and regularization."],
              ["04", "Predict demand", "Ensemble similarity-weighted analogue demand with a fitted DictVectorizer, StandardScaler, and Ridge pipeline."],
              ["05", "Separate evidence and risk", "Report analogue match confidence independently from conformal demand uncertainty, then apply pack and quantity limits."],
            ].map(([number, title, copy]) => <article key={number}><span>{number}</span><h3>{title}</h3><p>{copy}</p></article>)}
          </div>
          <div className="method-columns">
            <article className="formula-card">
              <span className="card-label">Decision formula</span>
              <h2>Readable enough to challenge</h2>
              <div className="formula">
                <p>Match score</p>
                <strong>{attributeWeight}% × Attribute + {visualWeight}% × FashionCLIP</strong>
              </div>
              <div className="formula">
                <p>Match confidence</p>
                <strong>Top match strength + top-3 consistency + image availability + analogue data quality</strong>
              </div>
              <div className="formula">
                <p>Historical demand target</p>
                <strong>Sales ÷ {targetSellThrough}% sell-through, winsorized by available supply</strong>
              </div>
              <div className="formula">
                <p>Demand ensemble</p>
                <strong>{Math.round((1 - dataset.meta.model.regressionBlend) * 100)}% analogue AI + {Math.round(dataset.meta.model.regressionBlend * 100)}% scikit-learn Ridge</strong>
              </div>
              <div className="formula">
                <p>Demand uncertainty</p>
                <strong>Narrow ≤ ±20% · Moderate ≤ ±40% · Wide &gt; ±40% of recommended buy</strong>
              </div>
              <div className="formula">
                <p>Expected range</p>
                <strong>80% conformal interval + 25-unit pack + 100–2,000 unit guardrails</strong>
              </div>
            </article>
            <article className="readiness-card">
              <span className="card-label">Model validation</span>
              <h2>What is measured</h2>
              <ul>
                <li><b>{scorePercent(dataset.meta.model.backtest.wape)} WAPE</b> in leave-one-out validation</li>
                <li><b>{numberFormatter.format(dataset.meta.model.backtest.mae)} units</b> mean absolute error</li>
                <li><b>{scorePercent(dataset.meta.model.backtest.intervalCoverage)}</b> empirical interval coverage</li>
                <li><b>{dataset.meta.dataQuality.dispatchAboveOrder + dataset.meta.dataQuality.salesAboveDispatch}</b> order/dispatch/sales anomalies contained by guardrails</li>
              </ul>
              <div className="warning-note">The architecture is production-oriented; the fitted quantity model remains a pilot because only {dataset.meta.model.trainingRows} historical rows are available. Three to five clean seasons are required for a credible temporal production backtest.</div>
            </article>
          </div>
          <div className="upgrade-table">
            <div><span>Layer</span><b>Local sample now</b><b>Scale platform implementation</b></div>
            <div><span>Visual representation</span><p>{dataset.meta.visionModel.modelId} · {dataset.meta.visionModel.embeddingDimension}D cosine · revision {dataset.meta.visionModel.modelRevision?.slice(0, 8) ?? "unknown"}</p><p>Fine-tuned FashionCLIP image/text service; client-specific tuning awaits reviewed pairs</p></div>
            <div><span>Retrieval</span><p>Precomputed attribute and FashionCLIP scoring against all 33 historical styles</p><p>Metadata-filtered pgvector HNSW top-200 retrieval, then top-10 re-ranking</p></div>
            <div><span>Quantity logic</span><p>Validation-tuned analogue + {dataset.meta.model.demandPipeline} scikit-learn pipeline</p><p>P10/P50/P90 LightGBM training and inference with MinTrace hierarchy reconciliation</p></div>
            <div><span>Decision signals</span><p>Separate match confidence and demand uncertainty labels</p><p>Relevance calibration plus category-level temporal forecast uncertainty</p></div>
            <div><span>Uncertainty</span><p>Out-of-fold conformal quantity range</p><p>Temporal quantiles calibrated by category, channel and region</p></div>
            <div><span>Workflow</span><p>Browser-session planner override</p><p>Durable batch jobs, feedback capture, model registry and recommendation audit schema</p></div>
            <div><span>Data</span><p>33 historical / 167 upcoming samples</p><p>3–5 seasons plus inventory and markdown context</p></div>
          </div>
          <section className="scale-platform">
            <div className="scale-platform-heading">
              <div>
                <span className="eyebrow">Production path / 200K–500K items</span>
                <h2>The scale engine is implemented without overstating model readiness</h2>
                <p>The software path is ready to deploy. Client data is still required to fit, validate and approve the learned models.</p>
              </div>
              <span className="scale-badge">AI v3 architecture</span>
            </div>
            <div className="scale-readiness-grid">
              {[
                ["Code ready", "Vector catalogue", "512-dimensional half-vector storage, hard metadata filters and HNSW nearest-neighbour retrieval."],
                ["Code ready", "Fashion embedding service", "Containerized FashionCLIP image/text inference with domain allowlisting, size limits and private-network blocking."],
                ["Needs client labels", "Learning-to-rank", "CatBoost training and model loading use planner relevance feedback, outcome reliability and product evidence."],
                ["Needs 3–5 seasons", "Demand forecasting", "Temporal P10/P50/P90 LightGBM training uses clean stock-out, markdown, channel and hierarchy features."],
                ["Needs calibration", "Hierarchy + risk", "MinTrace reconciliation keeps category, channel and region totals coherent before order constraints."],
                ["Code ready", "Operations", "MOQ, pack, budget and capacity optimisation plus durable ingestion, batch, feedback and audit records."],
              ].map(([state, title, copy]) => (
                <article key={title} className={state === "Code ready" ? "ready" : "waiting"}>
                  <span>{state}</span><h3>{title}</h3><p>{copy}</p>
                </article>
              ))}
            </div>
            <div className="activation-gate">
              <b>Production activation gate</b>
              <span>The service can be configured to refuse startup until approved ranker and demand-model artifacts are mounted—preventing a rule-based fallback from being presented as trained AI.</span>
            </div>
          </section>
        </section>
      )}

      {toast && <div className="toast" role="status">✓ {toast}</div>}
    </main>
  );
}

export default App;
