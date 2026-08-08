# Turtle Season Intelligence

Turtle Season Intelligence is a decision-support workspace for planning an
upcoming fashion season. A user uploads a historical ordering and sell-through
catalogue, an upcoming catalogue, and the matching product images; the service
retrieves visual analogues, forecasts demand, and recommends an initial order
quantity for each upcoming style.

Everything the planner shows comes from a build the user uploaded. The
repository ships no catalogue, no images and no pre-generated artifact — until
the first build is activated, the workspace shows its empty state and only the
**New analysis** tab does anything.

> Forecasts are planning evidence, not production certification. Image-match
> thresholds, model revisions, and demand uncertainty should be reviewed before
> commercial decisions are made.

## What the project does

- Accepts historical and upcoming catalogue uploads as CSV, XLSX, XLSM, XLS,
  XLSB or ODS, together with their product image sets.
- Validates, cleans, and standardizes catalogue data to a common schema,
  reporting row-level issues as a downloadable validation report.
- Matches uploaded product images to catalogue identifiers by filename stem.
- Retrieves visually similar historical products using FashionSigLIP, then
  verifies colour, texture, pattern, and construction with DINOv2-based
  evidence.
- Estimates demand from accepted visual analogues, correcting for historical
  exposure and stock constraints, then recommends a pack-rounded initial buy
  for a target sell-through.
- Activates the new build only after every recommendation validates, leaving
  the previous active build untouched if anything fails.

The visual retrieval design, score semantics, and evaluation guidance are
documented in [FASHION_MATCHING.md](FASHION_MATCHING.md).

## Architecture

```text
Browser: New analysis tab
  resumable catalogue + image uploads
             |
             v
FastAPI analysis service (backend/src/analysis_service)
  validation -> cleaning -> feature engineering
             |
             +--> visual retrieval and reranking (FashionSigLIP + DINOv2)
             |
             v
  demand model + order recommendation
             |
             v
  versioned build activated under DATA/analysis-service
             |
             v
Next.js planner reads /api/active and /api/builds/{id}/artifact
```

```text
turtle/
├── backend/
│   ├── src/analysis_service/     Upload API, run lifecycle, build store
│   ├── src/core/                 Shared project paths and environment overrides
│   ├── src/data_pipeline/        Cleaning, schema standardization, features
│   ├── src/fashion_matching/     Image retrieval, encoders, appearance evidence
│   ├── src/machine_learning/     Demand forecast and order-quantity logic
│   └── tests/                    Python unit and contract tests
├── frontend/
│   ├── app/                      Next.js planner and New analysis workspace
│   ├── public/templates/         Downloadable catalogue CSV templates
│   └── tests/                    Planner contract tests
├── DATA/                         Created at runtime; ignored by Git
│   └── analysis-service/         Uploaded builds, staging, feature cache
├── .env.example                  Visual matching and service settings
├── requirements.txt              Backend dependency source of truth
├── Makefile                      Common developer commands
├── FASHION_MATCHING.md           Detailed visual-matching documentation
└── Business_Process_Document_AI_Order_Quantity_Tool.pdf
```

## Prerequisites

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/) for the Python environment
- Node.js 22.13 or newer and npm
- LibreOffice, with `soffice` available on `PATH`, only if users will upload
  `.xls`, `.xlsb` or `.ods` catalogues

On macOS with Homebrew:

```bash
brew install uv
```

```bash
brew install --cask libreoffice
```

## Quick start

From the repository root:

```bash
uv venv .venv && uv pip install -r requirements.txt && npm --prefix frontend ci
```

```bash
cp .env.example .env && set -a && source .env && set +a
```

```bash
make api-dev
```

```bash
make frontend-dev
```

Open [http://localhost:3000](http://localhost:3000). The workspace opens on
**New analysis** because no build exists yet. Upload a catalogue pair to
populate the comparison and portfolio views.

The first visual build downloads the configured encoder weights through the
normal Hugging Face cache. It can take appreciably longer than subsequent
builds; CPU builds are suitable for development but may be slow for a full
catalogue.

Both processes are required: the Next.js application proxies `/api/*` to
`TURTLE_API_URL` (default `http://127.0.0.1:8000`), so the planner has no data
without the analysis service running.

## End-user catalogue uploads

The **New analysis** workspace supports two safe replacement modes:

- **Replace historical + upcoming** uploads both catalogues and image sets,
  builds a new historical demand/visual index, and activates it only after all
  recommendations pass validation.
- **Reuse trained historical** preserves the active historical features and
  uploads only a replacement upcoming catalogue and its images.

Catalogue files may be CSV, XLSX, XLSM, XLS, XLSB, or ODS. Images may be
selected individually, as a multi-file selection, or as a folder. Image filename
stems are matched case-insensitively to `product_id`, and may be JPG, JPEG, PNG
or WebP. Missing upcoming images create manual-review records rather than
failing the entire run.

Downloadable CSV templates for both catalogues are linked from the upload
workspace and served from `frontend/public/templates/`.

The current active planner stays usable while uploads and CPU inference run.
Failed or cancelled staging runs never replace active data. After successful
activation, superseded versions are removed; a reused historical version remains
reference-protected. Failed staging uploads are retained for 24 hours for retry.

Uploads are capped at 100 MB per catalogue file and 16 MB per image.

Primary endpoints:

```text
GET  /api/health
GET  /api/historical/active
POST /api/runs
HEAD /api/runs/{id}/uploads/{catalog}/{kind}/{filename}
PUT  /api/runs/{id}/uploads/{catalog}/{kind}/{filename}
POST /api/runs/{id}/complete-upload
GET  /api/runs/{id}
GET  /api/runs/{id}/events
POST /api/runs/{id}/cancel
GET  /api/runs/{id}/validation-report
GET  /api/runs/{id}/results
GET  /api/active
GET  /api/builds/{id}/artifact
GET  /api/builds/{id}/images/{catalog}/{productId}
GET  /api/builds/{id}/exports/{format}
```

`GET /api/active` returns `404` until the first build is activated; the planner
treats that as its empty state rather than an error.

## Local data

`DATA/` is created at runtime and is ignored by Git. Everything under it is
derived from uploads:

```text
DATA/analysis-service/
├── analysis.sqlite3     Run and build records
├── staging/             In-flight uploads; removed on success
├── historical/{id}/     Activated historical images and features.npz
├── upcoming/{id}/       Activated upcoming images
├── builds/{id}/         artifact.json and validation-report.csv
└── feature-cache/       Per-image embeddings, pruned after activation
```

Point `TURTLE_DATA_ROOT` (or `TURTLE_ANALYSIS_ROOT` directly) at another
location to keep uploaded catalogues off the repository volume.

During cleaning the service removes zero-sales historical rows, removes upcoming
item types with no retained historical coverage, normalizes identifiers and
fabric families, and records data-quality counts in the validation report and
artifact metadata.

## Forecast and recommendation policy

Only visual analogues that meet the minimum visual score (default `0.50`) and
reach at least Medium confidence can produce a recommendation. Otherwise the
build reports no suitable match and returns no sales forecast or order quantity
for that product.

For accepted matches, the backend:

1. pools accepted visual analogues rather than simply copying one product's
   sales;
2. estimates a censoring-aware weekly demand rate from historical sales,
   inventory exposure, and age;
3. shrinks sparse evidence toward hierarchical category priors;
4. produces a lognormal demand distribution and uncertainty bands; and
5. solves a newsvendor order quantity for the current target sell-through,
   rounded to packs of 25.

Buy ceilings are derived per item type from the uploaded historical data. A
ceiling is a policy guardrail, not evidence that the forecast has reached the
chosen sell-through target. The UI exposes the selected analogue, forecast
range, confidence, evidence diagnostics, and capped-quantity status so a planner
can review the recommendation.

## Configuration

Copy `.env.example` to `.env` and load it into the shell before starting the
service:

```bash
cp .env.example .env && set -a && source .env && set +a
```

| Setting | Purpose |
|---|---|
| `TURTLE_DATA_ROOT` | Overrides the default `DATA/` location. |
| `TURTLE_ANALYSIS_ROOT` | Overrides where uploaded builds are stored; defaults to `{TURTLE_DATA_ROOT}/analysis-service`. |
| `TURTLE_RUN_VISION` | Set to `false` to validate uploads without loading encoders. Development only. |
| `TURTLE_ALLOWED_ORIGINS` | Comma-separated CORS allowlist for the API. |
| `TURTLE_API_URL` | Origin the Next.js app proxies `/api/*` to. |
| `FASHION_MATCHING_MODEL_ID`, `FASHION_MATCHING_MODEL_REVISION` | Primary visual encoder and revision. Pin an exact revision before production use. |
| `FASHION_MATCHING_DEVICE`, `FASHION_MATCHING_BATCH_SIZE` | Inference device (`auto`, `cuda`, `mps`, or `cpu` as supported) and memory/throughput control. |
| `FASHION_DINO_*` | DINO reranking, candidate count, item-type constraint, and model revision. |
| `FASHION_PATTERN_*`, `FASHION_COLOUR_*` | Image-based acceptance gates used by the visual reranker. |
| `FASHION_APPEARANCE_*_WEIGHT` | Neural, colour, and texture blend weights; they must sum to `1`. |

## Run and test

| Command | Result |
|---|---|
| `make api-dev` | Starts the analysis service at `http://127.0.0.1:8000`. |
| `make frontend-dev` | Starts the planner at `http://localhost:3000`. |
| `make frontend-build` | Creates an optimized Next.js production build. |
| `make backend-test` | Runs Python tests. |
| `make frontend-test` | Builds the frontend and runs its contract tests. |
| `make test` | Runs both backend and frontend test suites. |
| `make lint` | Runs ruff over the backend and ESLint over the frontend. |

Additional frontend checks, from `frontend/`:

```bash
npx tsc --noEmit
```

## Frontend behavior

The planner has comparison and portfolio views, plus the New analysis workspace.
It lets a user inspect the selected upcoming style, the ranked historical visual
evidence, uncertainty bands, sales expectation, and recommended initial order.
Changing the minimum visual similarity or target sell-through recalculates the
recommendation in the browser from the activated build's published demand-model
data; it does not retrain the backend model.

Product images are delivered by `/api/builds/{id}/images/{catalog}/{productId}`,
which only reads the image directories belonging to that build.

## Limitations and responsible use

- The visual score threshold and reranker weights are provisional until they
  are calibrated against time-separated, human-reviewed catalogue matches.
- A visual analogue is evidence, not a guarantee of demand; sparse analogue
  pools deliberately yield wider uncertainty.
- Each build is a snapshot of the catalogues it was given. Upload again after
  source catalogues, images, model configuration, or operational assumptions
  change.
- Model downloads and visual inference may require substantial local disk,
  memory, and time. Pin reviewed model revisions before production use.
- Uploaded data, model caches, and generated builds are excluded from Git to
  avoid committing commercial or large binary assets.
