# Turtle Season Intelligence AI

Turtle Season Intelligence is a local client POC for matching upcoming fashion
products to relevant historical styles and recommending an initial order
quantity. The repository also contains a production-oriented scale architecture
for catalogues of approximately 200,000–500,000 items.

The current POC is local-only. No website hosting is configured, and the running
application does not call ChatGPT or the OpenAI API.

## Current status

| Layer | Status | Current reality |
|---|---|---|
| Planner interface | Active locally | Compare products, inspect match evidence, run scenarios, review the portfolio and export CSV |
| FashionCLIP image matching | Active in the POC | Real 512-dimensional embeddings from the supplied product images |
| Attribute similarity | Active in the POC | Weighted category, pattern, fit, fabric, colour, price and related evidence |
| Historical analogue retrieval | Active in the POC | All 33 historical products are scored; the validated default uses the top 8 |
| Order recommendation | Active as a pilot | Analogue demand blended with regularized regression and uncertainty guardrails |
| Sample Python API | Implemented and tested | Versioned v1 model and recommendation endpoints use the fitted POC artifact |
| Scale platform | Code implemented, not production-activated | Requires a populated pgvector catalogue and approved trained model artifacts |
| External hosting | Not configured | Run at `http://localhost:3000` |

## Supplied data and active model artifact

| Measure | Current value |
|---|---:|
| Historical items with outcomes | 33 |
| Upcoming items | 167 |
| Historical image coverage | 33 / 33 |
| Upcoming image coverage | 164 / 167 |
| Missing upcoming images | 3 |
| POC model version | 2.1.0 |
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
4. Calculate explainable attribute similarity for every upcoming/historical
   product pair.
5. Combine attributes and FashionCLIP, rank historical analogues, and use the
   selected top products as demand evidence.
6. Normalize historical sales to the target sell-through and contain anomalous
   rows relative to observed supply.
7. Blend similarity-weighted analogue demand with a ridge-regression baseline.
8. Apply an out-of-fold conformal range, 25-unit pack rounding, and 100–2,000
   unit pilot limits.

### Current validated defaults

| Setting | Default | Meaning |
|---|---:|---|
| Attribute weight | 80% | Structured commercial and product evidence |
| FashionCLIP weight | 20% | Garment-image similarity |
| Historical products used | 8 | Highest-ranked analogues included in the quantity calculation |
| Analogue forecast blend | 65% | Similarity-weighted historical demand |
| Regression blend | 35% | Regularized multivariate demand baseline |
| Target sell-through | 70% | Planning policy used to translate sales demand into inventory |

The 80/20 similarity blend is the best result inside the current coarse training
grid, which tests attribute weights from 40% to 80%. Because it selected the edge
of that grid and only 33 outcomes are available, it must not be treated as a
final production weight. Production selection requires wider, nested temporal
validation and planner-labelled relevance pairs.

The interface allows live scenario changes to the similarity weights, target
sell-through and analogue count. A custom scenario recalculates rankings and
quantities, but the displayed backtest metrics continue to describe the fitted
default model; changing a control does not retrain the model.

## Pilot validation

| Metric | Current result |
|---|---:|
| Leave-one-out WAPE | 40.59% |
| Mean absolute error | 168.7 units |
| Forecast bias | +9.45% |
| Empirical conformal interval coverage | 87.88% |
| Conformal half-width before similarity adjustment | 350 units |
| Medium-confidence upcoming items | 78 |
| Low-confidence upcoming items | 89 |
| High-confidence upcoming items | 0 |

These results are evidence for a POC, not production certification. Leave-one-out
validation is used because only 33 historical outcomes were supplied. A credible
production assessment needs at least three clean seasons and a forward temporal
holdout. The absence of high-confidence recommendations correctly reflects the
small sample and wide uncertainty.

The fitted data also contains eight dispatch-above-order records, one
sales-above-dispatch record and one sell-through-above-100% record. The pipeline
flags these issues and constrains their effect rather than silently trusting them.

## User experience

- Upcoming product queue with image, attribute and confidence filters
- Side-by-side upcoming and historical product images
- Ranked analogue cards with combined, attribute and FashionCLIP evidence
- Top 3, 5 or 8 analogue scenarios; every selected analogue is displayed and
  contributes to the calculation
- Validated-default versus custom-scenario labels
- Target sell-through and similarity-weight scenarios
- Recommended quantity, range, confidence and model components
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
FashionCLIP distances, performs leave-one-out model selection and atomically
refreshes `app/generated-data.json`. The model cache is local and git-ignored.

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

The current suite contains 10 Python tests plus a rendered-HTML regression test.

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
