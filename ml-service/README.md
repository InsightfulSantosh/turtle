# Turtle Season Intelligence ML service

This directory contains two related runtimes:

- a fitted v2.4.0 pilot that serves the local sample artifact; and
- a v3 scale architecture for PostgreSQL/pgvector retrieval, learned ranking,
  quantile demand forecasting, constrained ordering, jobs and feedback.

The v2.4.0 pilot is active and reproducible. The v3 software path is implemented,
but production model artifacts and a populated client catalogue are not present.

## Status at a glance

| Component | Code status | Model/data status | Active in local browser POC |
|---|---|---|---|
| FashionCLIP batch image embeddings | Implemented | Embeddings generated for 33 historical and 164 upcoming images with the base fashion-domain checkpoint | Yes, precomputed |
| Attribute similarity | Implemented | Nine audited, explainable fields; constants automatically excluded | Yes |
| Hybrid analogue retrieval | Implemented | Current default 20% attributes / 80% FashionCLIP, top 3 | Yes |
| Pilot sales ensemble | Implemented | scikit-learn Pipeline fitted on 33 sales outcomes with leave-one-out validation | Yes |
| Initial-order policy | Implemented | Expected sales divided by the target sell-through and pack-rounded | Yes |
| Conformal uncertainty and pilot buy limits | Implemented | Calibrated from out-of-fold residuals | Yes |
| v1 sample API | Implemented and tested | Loads `app/generated-data.json` | Optional; browser does not require it |
| FashionCLIP HTTP embedding service | Implemented | Public base model works for integration; client-tuned checkpoint absent | No |
| pgvector HNSW retrieval | Migration and repository implemented | Database is not populated in this repository | No |
| CatBoost learning-to-rank | Training and inference implemented | Approved trained artifact absent | No |
| LightGBM P10/P50/P90 forecasting | Training and inference implemented | Trained model bundle absent | No |
| MinTrace reconciliation | Algorithm, hierarchy builder and tests implemented | Residual covariance absent; not wired into the v2 request pipeline | No |
| Buy optimization | Implemented and wired into v2 | Uses request constraints | No |
| Batch worker, feedback and audit storage | Implemented | Requires PostgreSQL scale runtime | No |

## Active pilot model card

| Field | Value |
|---|---|
| Model version | 2.4.0 |
| Training outcomes | 33 historical items |
| Upcoming catalogue | 167 items |
| Image coverage | 33/33 historical; 164/167 upcoming |
| Image model | `patrickjohncyh/fashion-clip` |
| Checkpoint revision | `7e3ba62ce16b379a1ab479346b66f192e76f51b7` |
| Image representation | 512D unit-normalized FashionCLIP vectors |
| Visual comparison | Cosine distance with robust logistic calibration |
| Hybrid retrieval | 20% attribute + 80% visual; top 3 |
| Sales ensemble | 50% analogue + 50% scikit-learn Ridge; alpha 10 |
| Forecast target | Cleaned historical unit sales |
| Initial-order policy | Expected sales ÷ 70% target sell-through |
| Evaluation | Leave-one-out; temporal holdout unavailable |
| Sales WAPE | 44.57% |
| Sales MAE | 127.0 units |
| Sales bias | +3.82% |
| Interval | Finite-sample 80% conformal; 87.88% empirical coverage |
| Pilot order limits | 25-unit pack; 100 minimum; 2,000 maximum |

The model artifact records the exact image checkpoint revision, dimension,
execution device and coverage in `meta.visionModel`.

### Attribute similarity

The pilot keeps attributes separate from FashionCLIP so visual evidence does not
double-count structured commercial information.

| Attribute | Weight | Similarity method |
|---|---:|---|
| Category/item type | 16% | Exact match plus a strong cross-category penalty |
| Sleeve | 7% | Exact categorical match |
| Provision/fit code | 7% | Exact categorical match |
| Pattern | 17% | Exact or mapped pattern-family similarity |
| Season family | 5% | Normalized `AW`, `SS`, or `CORE` match |
| Fit/collection | 14% | Exact categorical match |
| Fabric | 14% | Exact match or token Jaccard similarity |
| Colour | 9% | Exact or mapped colour-family similarity |
| Price | 11% | Smooth log-price distance |

The current training search uses scikit-learn `ParameterGrid` and
`LeaveOneOut`. It tests attribute weights from 10% through 90%, neighbour counts
of 3, 5 and 8, Ridge penalties of 0.1, 1, 10 and 100, and regression blends of
15%, 25%, 35% and 50%. Model v2.4.0 selected 20% attributes / 80% vision,
top 3, alpha 10 and a 50/50 sales blend after the constant-field removal.
These remain pilot defaults and require nested temporal validation before
production.

The artifact builder profiles every candidate field against historical data. A
field must be comparable in both seasons and have at least two populated
historical values before it can contribute to similarity; active weights are
renormalized if another future workbook field becomes constant. In the supplied
workbooks, `CAT2` is constant after canonicalizing `CMI + VMI` and `VMI + CMI`,
while historical `CAT5` is entirely `FASHION`. Both are excluded from similarity
and from the pilot Ridge feature dictionary. Identifiers, colour variant codes
and outcome fields are retained for joins or forecasting but excluded from
product matching.

### Sales, order and risk logic

- The learned target is cleaned historical unit sales, not historical order or
  sales divided by a user-selected policy.
- The one impossible sales-above-observed-supply row is capped at the strongest
  available order/dispatch observation; valid sales outcomes remain unchanged.
- Top analogue sales are averaged with squared hybrid-similarity weights.
- A scikit-learn `DictVectorizer` → `StandardScaler` → `Ridge` Pipeline supplies
  the regularized multivariate baseline. Preprocessing and fitting occur inside
  every validation fold; there is no handwritten matrix inversion.
- The analogue and Ridge estimates are blended into expected customer sales.
- Absolute out-of-fold sales residuals produce a finite-sample conformal range.
- Recommended initial order is calculated as expected sales divided by target
  sell-through, then pack-rounded and constrained to the POC order limits.
- Changing target sell-through changes this inventory decision but leaves the AI
  expected-sales forecast unchanged.
- Match confidence describes analogue evidence only. High requires top similarity
  at least 84%, mean top-three similarity at least 72%, visual evidence and no
  analogue data-quality issues; medium requires 62% and 52%, respectively.
- Sales uncertainty is a separate range-width signal. The conformal half-width
  divided by expected sales is narrow at 20% or less, moderate through 40%,
  and wide above 40%.

This separation prevents a strong product match from being mislabeled as a weak
match simply because the demand model has limited history. In the current sample,
33 items have high match confidence, while all 167 items retain wide sales
uncertainty because the fitted conformal half-width is 250 units.

The interface can change similarity weights, target sell-through and analogue
count for scenario analysis. Similarity settings can change expected sales;
target sell-through changes only the recommended order. These controls do not
refit the model or recompute the displayed backtest score.

## Rebuild the pilot artifact

Run from the repository root after the product images and image map have been
prepared:

```bash
.venv/bin/pip install -r ml-service/requirements-fashionclip.txt
HF_HOME=.model-cache PYTHON=.venv/bin/python ./ml-service/tools/build_fashion_clip_features.sh
```

`tools/fashion_clip_embeddings.py` loads the local images, creates normalized
FashionCLIP vectors, writes historical-to-historical and
upcoming-to-historical cosine distances, and records model provenance.
`train_and_export.py` then calibrates visual distance, performs leave-one-out
model selection with scikit-learn and atomically refreshes
`app/generated-data.json`.

The batch builder is appropriate for this 200-item sample. It intentionally does
an all-pairs comparison and is not the 200,000–500,000 item retrieval path.

## Run the sample API

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r ml-service/requirements-dev.txt
TURTLE_API_KEY=change-me ENABLE_API_DOCS=true \
  .venv/bin/uvicorn --app-dir ml-service service:app --host 0.0.0.0 --port 8080
```

The v1 API loads the generated artifact at startup. Set
`TURTLE_MODEL_ARTIFACT` to use another artifact. API-key enforcement is enabled
when `TURTLE_API_KEY` is set.

### Endpoint status

| Endpoint | Purpose | Runtime requirement |
|---|---|---|
| `GET /healthz` | Sample runtime health and model version | Generated artifact |
| `GET /v1/model` | Pilot model card and coverage | Generated artifact |
| `GET /v1/recommendations/{item_id}` | Existing upcoming-item result | Generated artifact |
| `POST /v1/recommendations` | Recommendation for supplied attributes and visual similarities | Generated artifact |
| `GET /v2/health/ready` | Scale dependency readiness | Reports `sample_only` until configured |
| `POST /v2/recommendations` | Synchronous scale recommendation | PostgreSQL, embedding service and trained artifacts in production |
| `POST /v2/recommendations:batch` | Queue recommendation batch | Scale runtime and worker |
| `POST /v2/catalog/items:batch` | Queue catalogue ingestion | Scale runtime and worker |
| `GET /v2/jobs/{job_id}` | Read durable job state | Scale runtime |
| `POST /v2/feedback/similarity` | Record match relevance feedback | Scale runtime |
| `POST /v2/recommendations/{request_id}/decision` | Record approval or override | Scale runtime |

## Scale architecture for 200,000–500,000 items

The v3 path avoids an all-pairs join:

1. An isolated FashionCLIP service generates a 512D image/text embedding.
2. PostgreSQL/pgvector applies item type, gender, brand and price filters and
   uses HNSW cosine search to retrieve up to 200 candidates.
3. CatBoost re-ranks candidates using vector, attribute, price, demand
   reliability and planner-feedback features.
4. LightGBM produces P10/P50/P90 demand forecasts.
5. The buy optimizer applies pack, minimum, maximum, supplier-capacity and
   budget constraints.
6. PostgreSQL stores recommendation evidence, planner decisions and feedback;
   workers process durable catalogue and recommendation jobs.

MinTrace reconciliation is available as a tested library component and hierarchy
artifact builder. It is not yet invoked by `ScaleEngine.recommend`, and the
residual covariance needed for production reconciliation must be fitted from
rolling temporal forecast errors.

### Run the scale stack after artifacts are available

Copy and edit the environment file:

```bash
cp ml-service/.env.scale.example ml-service/.env.scale
docker compose --env-file ml-service/.env.scale -f docker-compose.scale.yml up --build
```

The production example deliberately uses `MODEL_POLICY=require_trained` and
`FASHION_MODEL_ID=/models/fashion-clip`. It will not become ready until the
approved FashionCLIP directory, CatBoost ranker, LightGBM demand bundle and
historical catalogue embeddings are mounted or ingested.

Use `allow_fallback` only for integration development. The fallback ranker and
analogue quantiles are deterministic software fallbacks, not trained production
AI, and must not be presented as such.

## Training programs and data contracts

All learned training programs require at least three seasons and reserve the
latest season as a temporal holdout.

```bash
cd ml-service
../.venv/bin/pip install -r requirements-training.txt
../.venv/bin/python -m training.fine_tune_fashion_clip pairs.csv ../models/fashion-clip
../.venv/bin/python -m training.train_ranker ranker.parquet ../models/ranker.cbm
../.venv/bin/python -m training.train_demand demand.parquet ../models/demand
../.venv/bin/python -m training.build_hierarchy series.parquet ../models/hierarchy.json
```

Required training inputs:

- FashionCLIP pairs: upcoming image path, historical image path, relevance and
  season.
- Ranker: query ID, season, relevance and all fields in `RANK_FEATURES`.
- Demand: season, observed unit sales and the configured forecast features.
- Hierarchy: one unambiguous category/channel/region path per bottom series.

## Production activation checklist

- Provide three to five clean seasons with consistent selling windows.
- Add stock-out, replenishment, markdown, price, channel, region, supplier,
  pack/MOQ, capacity and budget context.
- Collect planner-reviewed positive and negative image-match pairs.
- Train and evaluate FashionCLIP, CatBoost and LightGBM using forward temporal
  holdouts and category-level diagnostics.
- Fit hierarchy residual covariance and connect reconciliation to the forecast
  request path.
- Populate pgvector embeddings and validate filtered-retrieval recall/latency.
- Calibrate confidence and prediction intervals by category and decision use.
- Configure secrets, approved image domains, TLS, logging, metrics, model
  registry, drift monitoring, rollback and retention policies.
- Run load, failure-recovery, security and user-acceptance tests before planners
  rely on recommendations.

## Tests

From the repository root:

```bash
.venv/bin/pip install -r ml-service/requirements-dev.txt
.venv/bin/python -m pytest -q ml-service/tests
```

The current suite has 14 tests covering pilot model behavior, the scikit-learn
pipeline, artifact contract, vector validation, ranking, quantiles, optimization
and MinTrace coherence.
