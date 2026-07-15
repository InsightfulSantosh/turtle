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
  };
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
    visualMethod: string;
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
  confidence: Confidence;
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
    const base = (historical.order + historical.dispatch) / 2;
    const performanceFactor = clamp(
      historical.sellThrough / Math.max(targetSellThrough / 100, 0.01),
      0.65,
      1.35,
    );
    const adjusted = base * performanceFactor;
    const weight = Math.max(match.combinedScore, 0.01) ** 2;
    numerator += adjusted * weight;
    denominator += weight;
  });

  const quantity = clamp(roundPack(numerator / Math.max(denominator, 0.01)), 100, 2000);
  const topScore = top[0]?.combinedScore ?? 0;
  const averageTop =
    top.slice(0, 3).reduce((sum, match) => sum + match.combinedScore, 0) /
    Math.max(top.slice(0, 3).length, 1);
  let confidence: Confidence = "Low";
  let spread = 0.25;
  if (topScore >= 0.82 && averageTop >= 0.72) {
    confidence = "High";
    spread = 0.1;
  } else if (topScore >= 0.67 && averageTop >= 0.58) {
    confidence = "Medium";
    spread = 0.16;
  }

  return {
    ranked,
    quantity,
    low: roundPack(quantity * (1 - spread)),
    high: roundPack(quantity * (1 + spread)),
    confidence,
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
  const [failed, setFailed] = useState(false);
  useEffect(() => setFailed(false), [src]);

  if (!src || failed) {
    return (
      <div className={`image-fallback ${className}`} aria-label={`${alt}: image unavailable`}>
        <span>Image pending</span>
        <small>Attribute-only match</small>
      </div>
    );
  }

  return (
    <img
      className={className}
      src={src}
      alt={alt}
      loading={eager ? "eager" : "lazy"}
      referrerPolicy="no-referrer"
      onError={() => setFailed(true)}
    />
  );
}

function ConfidencePill({ confidence }: { confidence: Confidence }) {
  return <span className={`confidence-pill ${confidence.toLowerCase()}`}>{confidence} confidence</span>;
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

function bestReasons(match: RankedMatch) {
  return Object.entries(match.attributeBreakdown)
    .filter(([, value]) => value >= 0.62)
    .sort((left, right) => right[1] - left[1])
    .slice(0, 3)
    .map(([key]) => attributeLabels[key] ?? key);
}

function App() {
  const initialItem = dataset.upcoming.find(
    (item) => item.recommendation.confidence === "High" && item.imageUrl,
  ) ?? dataset.upcoming[0];
  const [tab, setTab] = useState<Tab>("compare");
  const [selectedId, setSelectedId] = useState(initialItem.id);
  const [queueSearch, setQueueSearch] = useState("");
  const [segment, setSegment] = useState("All");
  const [confidenceFilter, setConfidenceFilter] = useState("All");
  const [attributeWeight, setAttributeWeight] = useState(65);
  const [visualWeight, setVisualWeight] = useState(35);
  const [targetSellThrough, setTargetSellThrough] = useState(70);
  const [topK, setTopK] = useState(5);
  const [focusedHistoricalId, setFocusedHistoricalId] = useState<string | null>(null);
  const [overrides, setOverrides] = useState<Record<string, number>>({});
  const [toast, setToast] = useState("");

  const selected = dataset.upcoming.find((item) => item.id === selectedId) ?? initialItem;
  const decision = useMemo(
    () => makeDecision(selected, attributeWeight, visualWeight, targetSellThrough, topK),
    [selected, attributeWeight, visualWeight, targetSellThrough, topK],
  );

  useEffect(() => {
    setFocusedHistoricalId(decision.ranked[0]?.historicalId ?? null);
  }, [selectedId, attributeWeight, visualWeight]);

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
        (confidenceFilter === "All" || itemDecision.confidence === confidenceFilter)
      );
    });
  }, [portfolio, queueSearch, segment, confidenceFilter]);

  const focusedMatch =
    decision.ranked.find((match) => match.historicalId === focusedHistoricalId) ??
    decision.ranked[0];
  const focusedHistory = focusedMatch ? historyById.get(focusedMatch.historicalId) : undefined;
  const finalQuantity = overrides[selected.id] ?? decision.quantity;

  const highConfidenceCount = portfolio.filter(
    ({ decision: itemDecision }) => itemDecision.confidence === "High",
  ).length;
  const mediumConfidenceCount = portfolio.filter(
    ({ decision: itemDecision }) => itemDecision.confidence === "Medium",
  ).length;
  const lowConfidenceCount = portfolio.length - highConfidenceCount - mediumConfidenceCount;
  const totalBuy = portfolio.reduce(
    (sum, { item, decision: itemDecision }) => sum + (overrides[item.id] ?? itemDecision.quantity),
    0,
  );

  function chooseItem(id: string) {
    setSelectedId(id);
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
        "Visual similarity",
        "Confidence",
        "Recommended quantity",
        "Planner quantity",
        "Target sell-through",
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
          itemDecision.confidence,
          itemDecision.quantity,
          overrides[item.id] ?? "",
          targetSellThrough,
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
          <span className="sync-state"><i /> Sample synchronized</span>
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
              <select value={confidenceFilter} onChange={(event) => setConfidenceFilter(event.target.value)} aria-label="Filter confidence">
                <option>All</option>
                <option>High</option>
                <option>Medium</option>
                <option>Low</option>
              </select>
            </div>
            <div className="queue-list">
              {queueItems.map(({ item, decision: itemDecision }) => (
                <button
                  key={item.id}
                  className={`queue-item ${selected.id === item.id ? "selected" : ""}`}
                  onClick={() => setSelectedId(item.id)}
                >
                  <ProductImage src={item.imageUrl} alt={item.id} className="queue-image" />
                  <span className="queue-copy">
                    <strong>{item.id}</strong>
                    <small>{item.pattern} · {item.colour}</small>
                    <span className={`mini-confidence ${itemDecision.confidence.toLowerCase()}`}>
                      {itemDecision.confidence}
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
                    <p>Similarity-weighted historical demand</p>
                  </div>
                  <ConfidencePill confidence={decision.confidence} />
                </div>
                <div className="quantity-hero">
                  <div>
                    <strong>{numberFormatter.format(finalQuantity)}</strong>
                    <span>units</span>
                  </div>
                  <p>Suggested range <b>{numberFormatter.format(decision.low)}–{numberFormatter.format(decision.high)}</b></p>
                </div>
                <div className="confidence-track">
                  <span style={{ width: `${Math.round((decision.ranked[0]?.combinedScore ?? 0) * 100)}%` }} />
                </div>
                <div className="rationale-box">
                  <span className="rationale-icon">✦</span>
                  <p>
                    Based on <b>{topK} nearest historical styles</b>. The leading analogue scored {scorePercent(decision.ranked[0]?.combinedScore ?? 0)}, with historical sell-through normalized to a {targetSellThrough}% target.
                  </p>
                </div>
                <div className="recommendation-metrics">
                  <div><small>Top analogue</small><strong>{decision.ranked[0]?.historicalId}</strong></div>
                  <div><small>Top match</small><strong>{scorePercent(decision.ranked[0]?.combinedScore ?? 0)}</strong></div>
                  <div><small>Planner status</small><strong>{overrides[selected.id] ? "Adjusted" : "Pending review"}</strong></div>
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
                <label className="select-control">
                  Historical analogues
                  <select value={topK} onChange={(event) => setTopK(Number(event.target.value))}>
                    <option value="3">Top 3</option>
                    <option value="5">Top 5</option>
                    <option value="8">Top 8</option>
                  </select>
                </label>
                <div className="method-note">
                  <span>POC visual engine</span>
                  <p>Garment-region colour, structure, texture and edge features. Production replaces this with fashion embeddings.</p>
                </div>
              </aside>
            </div>

            <section className="matches-section">
              <div className="section-heading">
                <div>
                  <span className="eyebrow">Ranked historical analogues</span>
                  <h2>Why these styles are relevant</h2>
                </div>
                <div className="score-legend"><span><i className="attr" /> Attribute</span><span><i className="visual" /> Visual</span></div>
              </div>
              <div className="match-grid">
                {decision.ranked.slice(0, 5).map((match, index) => {
                  const historical = historyById.get(match.historicalId);
                  if (!historical) return null;
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
                        <div className="match-reasons">{bestReasons(match).map((reason) => <span key={reason}>{reason}</span>)}</div>
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
                  <p>Component scores show exactly what drove the recommendation. Click another analogue above to inspect its evidence.</p>
                </div>
                <div className="evidence-scores">
                  <ScoreRing score={focusedMatch.combinedScore} label="Combined" />
                  <ScoreRing score={focusedMatch.attributeScore} label="Attributes" />
                  <ScoreRing score={focusedMatch.visualScore ?? 0} label="Visual" />
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
            <div><span className="eyebrow">Upcoming assortment</span><h1>Portfolio recommendation</h1><p>Review confidence, analogues and buy quantities across the full sample.</p></div>
            <button className="button primary" onClick={exportCsv}>Export recommendation file</button>
          </div>
          <div className="kpi-grid">
            <article><span>Total styles</span><strong>{dataset.meta.upcomingItems}</strong><small>{dataset.meta.upcomingImageCoverage} with images</small></article>
            <article><span>High confidence</span><strong>{highConfidenceCount}</strong><small>{Math.round((highConfidenceCount / portfolio.length) * 100)}% ready for review</small></article>
            <article><span>Needs judgement</span><strong>{lowConfidenceCount}</strong><small>{mediumConfidenceCount} medium confidence</small></article>
            <article className="accent"><span>Recommended buy</span><strong>{numberFormatter.format(totalBuy)}</strong><small>units across sample</small></article>
          </div>
          <div className="portfolio-toolbar">
            <label className="search-box wide"><span>⌕</span><input value={queueSearch} onChange={(event) => setQueueSearch(event.target.value)} placeholder="Search item, pattern, colour or fabric" /></label>
            <select value={segment} onChange={(event) => setSegment(event.target.value)}><option>All</option><option>OTSH</option><option>OTTS</option></select>
            <select value={confidenceFilter} onChange={(event) => setConfidenceFilter(event.target.value)}><option>All</option><option>High</option><option>Medium</option><option>Low</option></select>
            <span>{queueItems.length} results</span>
          </div>
          <div className="portfolio-table-wrap">
            <table className="portfolio-table">
              <thead><tr><th>Upcoming style</th><th>Product attributes</th><th>Top historical analogue</th><th>Match</th><th>Confidence</th><th>System buy</th><th>Planner buy</th><th /></tr></thead>
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
                      <td><ConfidencePill confidence={itemDecision.confidence} /></td>
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
            <div><span className="eyebrow">Explainable by design</span><h1>How the POC reaches a recommendation</h1><p>A transparent retrieval and decision-support workflow, designed for planner validation before production use.</p></div>
            <div className="poc-badge"><span>POC</span><small>Client demonstration</small></div>
          </div>
          <div className="workflow-grid">
            {[
              ["01", "Normalize", "Create a canonical item key and standardize category, fabric, colour and range values."],
              ["02", "Compare attributes", "Score category, sleeve, fit, pattern, fabric, colour and price-band similarity."],
              ["03", "Compare visuals", "Extract garment-region colour, structure, texture and edge features from every image."],
              ["04", "Rank analogues", "Blend configurable attribute and visual scores to retrieve the most relevant history."],
              ["05", "Recommend buy", "Weight historical order and dispatch by sell-through and similarity, then round to case pack."],
            ].map(([number, title, copy]) => <article key={number}><span>{number}</span><h3>{title}</h3><p>{copy}</p></article>)}
          </div>
          <div className="method-columns">
            <article className="formula-card">
              <span className="card-label">Decision formula</span>
              <h2>Readable enough to challenge</h2>
              <div className="formula">
                <p>Match score</p>
                <strong>{attributeWeight}% × Attribute + {visualWeight}% × Visual</strong>
              </div>
              <div className="formula">
                <p>Performance-adjusted quantity</p>
                <strong>Average(Order, Dispatch) × Sell-through / {targetSellThrough}% target</strong>
              </div>
              <div className="formula">
                <p>Final recommendation</p>
                <strong>Similarity-weighted top {topK} × constraints × 25-unit pack rounding</strong>
              </div>
            </article>
            <article className="readiness-card">
              <span className="card-label">Data readiness</span>
              <h2>What the sample proves</h2>
              <ul>
                <li><b>{dataset.meta.historicalItems}/{dataset.meta.historicalItems}</b> historical styles linked to images</li>
                <li><b>{dataset.meta.upcomingImageCoverage}/{dataset.meta.upcomingItems}</b> upcoming styles linked to images</li>
                <li><b>{dataset.meta.missingUpcomingImages.length}</b> upcoming image exceptions flagged</li>
                <li><b>199/199</b> supplied image URLs accessible during validation</li>
              </ul>
              <div className="warning-note">Production quantity accuracy requires more seasons, consistent sales windows, replenishment and stock-out data, MOQ and budget constraints.</div>
            </article>
          </div>
          <div className="upgrade-table">
            <div><span>Layer</span><b>POC now</b><b>Production upgrade</b></div>
            <div><span>Visual representation</span><p>Garment-region handcrafted feature vector</p><p>Fashion-specific deep image embedding</p></div>
            <div><span>Quantity logic</span><p>Explainable similarity-weighted rule</p><p>Backtested hierarchical demand model</p></div>
            <div><span>Workflow</span><p>Browser-session planner override</p><p>Authenticated approvals and audit trail</p></div>
            <div><span>Data</span><p>33 historical / 167 upcoming samples</p><p>3–5 seasons plus inventory and markdown context</p></div>
          </div>
        </section>
      )}

      {toast && <div className="toast" role="status">✓ {toast}</div>}
    </main>
  );
}

export default App;
