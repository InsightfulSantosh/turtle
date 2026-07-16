# Turtle Season Intelligence AI

Turtle Season Intelligence is a local client POC for matching upcoming fashion
products to relevant historical styles and recommending an initial order
quantity. The repository also contains a production-oriented scale architecture
for catalogues of approximately 200,000–500,000 items.

The current POC runs locally with standard Next.js and does not require an
external hosting or authentication platform.

## Current status

| Layer | Status | Current reality |
|---|---|---|
| Planner interface | Active locally | Compare products, inspect match evidence, run scenarios, review the portfolio and export CSV |
| FashionCLIP image matching | Active in the POC | Real 512-dimensional embeddings from the supplied product images |
| Attribute similarity | Active in the POC | Weighted category, pattern, fit, fabric, colour, price and related evidence |
| Historical analogue retrieval | Active in the POC | All 33 historical products are scored; the validated default uses the top 8 |
| Order recommendation | Active as a pilot | Analogue demand blended with a scikit-learn Ridge pipeline and uncertainty guardrails |
| Sample Python API | Implemented and tested | Versioned v1 model and recommendation endpoints use the fitted POC artifact |
| Scale platform | Code implemented, not production-activated | Requires a populated pgvector catalogue and approved trained model artifacts |

## Supplied data and active model artifact

| Measure | Current value |
|---|---:|
| Historical items with outcomes | 33 |
| Upcoming items | 167 |
| Historical image coverage | 33 / 33 |
| Upcoming image coverage | 164 / 167 |
| Missing upcoming images | 3 |
| POC model version | 2.3.2 |
| FashionCLIP dimension | 512 |

The three upcoming items without a usable local image are
`OTSH-62055-1001`, `OTSH-61388V-1014`, and `OTSH-61670V-1004`. They use
attribute-only matching and are explicitly flagged in the interface.

The active image model is `patrickjohncyh/fashion-clip` at revision
`7e3ba62ce16b379a1ab479346b66f192e76f51b7`. Product images are read from the
local image directory when the model artifact is built. The browser POC then
uses the generated artifact and does not need a live embedding service.

## How the POC works

1. Validate identifiers, product attributes, order, dispatch, sales and
   sell-through fields.
2. Encode available product images with FashionCLIP and unit-normalize the
   resulting 512-dimensional vectors.
3. Convert cosine distance into a calibrated visual-similarity score.
4. Calculate explainable similarity across nine informative, comparable
   attributes for every upcoming/historical product pair.
5. Combine attributes and FashionCLIP, rank historical analogues, and use the
   selected top products as demand evidence.
6. Normalize historical sales to the target sell-through and contain anomalous
   rows relative to observed supply.
7. Fit a scikit-learn `DictVectorizer` → `StandardScaler` → `Ridge` pipeline and
   blend its prediction with similarity-weighted analogue demand.
8. Apply an out-of-fold conformal range, 25-unit pack rounding, and 100–2,000
   unit pilot limits.

### Current validated defaults

| Setting | Default | Meaning |
|---|---:|---|
| Attribute weight | 80% | Structured commercial and product evidence |
| FashionCLIP weight | 20% | Garment-image similarity |
| Historical products used | 8 | Highest-ranked analogues included in the quantity calculation |
| Analogue forecast blend | 50% | Similarity-weighted historical demand |
| Ridge forecast blend | 50% | Fitted scikit-learn multivariate demand baseline |
| Target sell-through | 70% | Planning policy used to translate sales demand into inventory |

`scikit-learn` performs the pilot model selection with `LeaveOneOut` and
`ParameterGrid`. The search tests attribute weights from 10% to 90%, top 3/5/8
analogues, Ridge penalties of 0.1/1/10/100, and regression blends of
15%/25%/35%/50%. The selected 80/20 similarity blend, top 8, 50/50 demand blend,
and Ridge alpha 10 are still pilot defaults because only 33 outcomes are
available. Production selection requires nested temporal validation and
planner-labelled relevance pairs.

The workbook audit retains item type, sleeve, provision/fit code, pattern,
lifecycle family, collection/fit, fabric, colour name and MRP. `CAT2` range is
constant after normalizing `CMI + VMI` / `VMI + CMI`, and historical `CAT5`
merch type is entirely `FASHION`; both are excluded from similarity and the
pilot Ridge feature set. Identifiers, colour variant codes and historical demand
outcomes are also kept out of similarity for semantic and leakage reasons. The
Methodology screen exposes the source-column mapping, distinct-value counts,
weights and exclusions.

The interface allows live scenario changes to the similarity weights, target
sell-through and analogue count. A custom scenario recalculates rankings and
quantities, but the displayed backtest metrics continue to describe the fitted
default model; changing a control does not retrain the model.

## Pilot validation

| Metric | Current result |
|---|---:|
| Leave-one-out WAPE | 41.47% |
| Mean absolute error | 172.3 units |
| Forecast bias | +7.37% |
| Empirical conformal interval coverage | 87.88% |
| Conformal half-width before similarity adjustment | 325 units |
| High match-confidence items | 12 |
| Medium match-confidence items | 141 |
| Low match-confidence items | 14 |
| Narrow demand-uncertainty ranges | 0 |
| Moderate demand-uncertainty ranges | 0 |
| Wide demand-uncertainty ranges | 167 |

These results are evidence for a POC, not production certification. Leave-one-out
validation is used because only 33 historical outcomes were supplied. A credible
production assessment needs at least three clean seasons and a forward temporal
holdout. Model v2.3.2 reports two deliberately separate signals: match confidence
describes the relevance and quality of the historical analogues, while demand
uncertainty describes the conformal forecast half-width relative to the proposed
buy. A product can therefore have a high-confidence match and a wide demand
range. The predominance of wide ranges reflects the small 33-outcome sample and
large out-of-fold forecast errors.

The fitted data also contains eight dispatch-above-order records, one
sales-above-dispatch record and one sell-through-above-100% record. The pipeline
flags these issues and constrains their effect rather than silently trusting them.

## User experience

- Upcoming product queue with image, pattern/colour, collection/fit, MRP,
  recommended buy, match-confidence and demand-uncertainty signals
- Consistent upcoming and historical match-attribute catalogs: Category,
  Pattern, Collection and Fabric appear first; `+5 Show all 9` reveals Colour,
  Sleeve, Fit code, Season family and Price band
- Side-by-side upcoming and historical product images
- Ranked historical analogue cards with the same expandable nine attributes,
  order and sell-through alongside the similarity evidence
- Top 3, 5 or 8 analogue scenarios; every selected analogue is displayed and
  contributes to the calculation
- Validated-default versus custom-scenario labels
- Target sell-through and similarity-weight scenarios
- Recommended quantity, expected demand range, separate decision signals and
  model components
- Planner quantity override and approval interaction
- Portfolio table, filters, totals and CSV export
- Methodology page with model provenance and production-readiness boundaries

## Run the local POC

Prerequisites:

- Node.js 22.13 or newer
- npm

From the repository root:

```bash
npm install
npm run dev
```

Open `http://localhost:3000`.

The frontend reads `app/generated-data.json`, so FashionCLIP and the Python API
do not need to run while presenting the already-built POC.

## Rebuild the data and FashionCLIP artifact

The workbook preparation script expects the reviewed workbooks in the parent
project and converted `.xlsx` copies in `../tmp/converted`. It writes the image
download map to `../tmp/vision-images-map.json` and images to
`../tmp/vision-images`.

```bash
python3 -m venv .venv
.venv/bin/pip install -r ml-service/requirements-fashionclip.txt
.venv/bin/pip install openpyxl
.venv/bin/python scripts/prepare_sample_data.py downloads
curl -L --config ../tmp/vision-images.curl.conf
.venv/bin/python scripts/prepare_sample_data.py build
HF_HOME=.model-cache PYTHON=.venv/bin/python ./ml-service/tools/build_fashion_clip_features.sh
```

The final command replaces the temporary preprocessing similarities with real
FashionCLIP distances, fits and validates the scikit-learn demand pipeline, and
atomically refreshes `app/generated-data.json`. The model cache is local and
git-ignored.

## Verification

Frontend:

```bash
npm run lint
npx tsc --noEmit
npm test
```

Python model and scale components:

```bash
.venv/bin/pip install -r ml-service/requirements-dev.txt
.venv/bin/python -m pytest -q ml-service/tests
```

The current suite contains 14 Python tests plus a local frontend/model-contract
test; `npm test` also performs a complete standard Next.js production build.

## Scale architecture status

The repository includes PostgreSQL/pgvector HNSW retrieval, a protected
FashionCLIP embedding service, CatBoost ranking training/inference, LightGBM
P10/P50/P90 training/inference, a MinTrace reconciliation component, constrained
buy optimization, durable jobs, feedback capture and recommendation audit
storage.

Those components are not equivalent to trained production models. No approved
CatBoost ranker, LightGBM demand bundle, client-tuned FashionCLIP checkpoint or
residual covariance artifact can be fitted honestly from the current 33-row
sample. The production configuration uses `MODEL_POLICY=require_trained` so the
scale service fails closed when required artifacts are missing.

See `ml-service/README.md` for the precise component status, API endpoints,
training contracts and production activation checklist.

## Project layout

```text
app/                         Local planner interface and generated model artifact
scripts/                     Workbook and image preparation
ml-service/                  POC model, APIs, scale engine, training and tests
embedding-service/           Isolated FashionCLIP HTTP service
docker-compose.scale.yml     PostgreSQL, embedding, API and worker stack
models/                      Local trained artifacts; intentionally git-ignored
```
