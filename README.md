# Turtle Season Intelligence

The production image-first retrieval workflow, with separate image, text and
structured-attribute signals, is documented in
[FASHION_MATCHING.md](FASHION_MATCHING.md).

Turtle Season Intelligence compares upcoming fashion styles with historical
products, forecasts demand and supports an initial-order decision. The codebase
is split into an independent Next.js frontend and a modular Python backend.

The current browser artifact is built from the real historical and SS27
workbooks in `DATA/raw`. Model limitations and forecast uncertainty remain
visible in the application and should be treated as decision support rather
than production certification.

## Architecture

```text
turtle/
├── frontend/                         Next.js planner application
│   ├── app/                          Pages, styles and generated-data.json
│   ├── public/                       Static assets
│   └── tests/                        Frontend and artifact-contract tests
├── backend/                          Python data-science pipeline (no live API)
│   ├── src/
│   │   ├── core/                     Central configuration and paths
│   │   ├── data_pipeline/            Workbook validation and preparation
│   │   ├── fashion_matching/         Visual retrieval, encoding and image validation
│   │   └── machine_learning/         Analogue similarity and predictive demand models
│   └── tests/                        Backend unit and contract tests
├── requirements.txt                  Single source of truth for backend dependencies
├── pyproject.toml                    Backend package, pytest and ruff configuration
└── Makefile                          Central developer commands
```

The dependency direction is intentional:

```text
data_pipeline (CLI)
      ↓
machine learning + fashion matching
      ↓
core configuration
```

The backend is a CLI-driven pipeline, not a live service: it writes the
versioned browser artifact to `frontend/app/generated-data.json`, and the
frontend reads that static file directly. Frontend code does not import
backend internals or call a backend API at runtime; product images are
served by the frontend's own route, which reads `DATA/raw` directly.

## Central configuration

All local paths are defined in `backend/src/core/config.py`.

The defaults are:

- Raw source workbooks and product images: `DATA/raw`
- Cleaned datasets and validation report: `DATA/processed`
- Temporary workbook conversions: `tmp`
- Browser model artifact: `frontend/app/generated-data.json`

Deployments can override them with:

- `TURTLE_DATA_ROOT`
- `TURTLE_TEMP_ROOT`
- `TURTLE_MODEL_ARTIFACT`

## Setup

Prerequisites:

- Python 3.12
- [uv](https://docs.astral.sh/uv/) for virtual environment and package management
- Node.js 22.13 or newer
- npm
- LibreOffice for `.xlsb` conversion

On macOS, install LibreOffice and uv with Homebrew:

```bash
brew install --cask libreoffice
brew install uv
```

`soffice` must be on `PATH` afterward (`which soffice`) — the data pipeline
shells out to it to convert the raw `.xlsb` workbooks and fails with a clear
error if it's missing.

From the repository root, `requirements.txt` is the single source of truth
for backend Python dependencies:

```bash
uv venv .venv
uv pip install -r requirements.txt
npm --prefix frontend install
```

## Common commands

```bash
make data            # rebuild the frontend artifact from the real workbooks
make frontend-dev    # run the planner at http://localhost:3000
make backend-test    # run Python tests
make frontend-test   # build and test the Next.js application
make test            # run both test suites
```

The frontend runs by itself because it reads the generated JSON artifact
directly; there is no live backend service to run alongside it.

## Backend

The backend is a Python 3.12 src-layout data-science package with no live
API — it is a CLI-driven pipeline that writes a versioned browser artifact
consumed directly by the frontend. It keeps data preparation, visual
matching and machine learning concerns separate while sharing one
configuration.

| Location | Responsibility |
|---|---|
| `backend/src/core/` | Environment and filesystem configuration |
| `backend/src/data_pipeline/` | Real workbook ingestion, validation and artifact export |
| `backend/src/fashion_matching/` | Visual retrieval, image encoding and validation |
| `backend/src/machine_learning/` | Analogue similarity and predictive demand estimation |

### Rebuild real data

```bash
make data
```

To map local product images, generate fashion-domain image embeddings and
rebuild the browser artifact with visual similarity:

```bash
make data-vision
```

This requires LibreOffice (`soffice` on `PATH` — see Setup above) and uses
the cached models under the normal Hugging Face cache directory.

To rebuild against only a deterministic sample of the upcoming catalogue
(faster iteration, same underlying model and gating logic) instead of the
full run:

```bash
PYTHONPATH=backend/src .venv/bin/python -m data_pipeline.prepare_real_data \
  --with-vision --sample-size 200 --sample-seed 27 --verbose \
  --log-file tmp/data-vision-progress.log
```

The pooled, censoring-corrected predictive estimator is always used — there
is no flag to fall back to the legacy copy-one-analogue rule; that rule only
still applies per-item, automatically, when a product has no accepted visual
analogue to pool from. `--sample-size` selects a stratified sample balanced
across item type and category type; `--sample-seed` (default `27`) keeps the
same sample stable across reruns. Omit `--output` to write directly to
`frontend/app/generated-data.json` (the same file `make data-vision` writes),
or pass `--output` with a different path to build a side-by-side preview
without touching the live artifact.

The pipeline reads the two real `.xlsb` workbooks from `DATA/raw`, converts them
through LibreOffice when needed, validates and cleans the records, standardizes
both datasets to a shared snake_case schema, saves the cleaned datasets and
validation report under `DATA/processed`, excludes zero-sales historical rows
from training, excludes upcoming item types without retained historical
coverage, fits the current model and atomically writes
`frontend/app/generated-data.json`.

Image filenames are matched case-insensitively to catalogue product IDs from
`DATA/raw/historical_matched_images` and
`DATA/raw/upcoming_ss27_matched_images`. The browser intentionally lists only
upcoming styles with mapped images and only image-backed historical analogues.
The Next.js application serves these files through a same-origin image route,
so `make frontend-dev` is sufficient for the planner UI.

Similarity and demand features use only Item, Design, Colour, Category Type and
Fabric. Category Type is standardized to `FORMAL`, `CASUAL`, `DENIM` or
`CEREMONIAL` during preprocessing.

#### Pipeline modules

| File | Pipeline responsibility |
|---|---|
| `backend/src/data_pipeline/settings.py` | Source, temporary and artifact paths |
| `backend/src/data_pipeline/ingestion.py` | Workbook discovery, conversion and row loading |
| `backend/src/data_pipeline/validation.py` | Post-cleaning identifier and fabric quality gates |
| `backend/src/data_pipeline/preprocessing.py` | Identifier and fabric-family normalization rules |
| `backend/src/data_pipeline/schema.py` | Canonical column mappings and schema-drift checks |
| `backend/src/data_pipeline/feature_engineering.py` | Model-ready historical and upcoming records |
| `backend/src/data_pipeline/pipeline.py` | End-to-end orchestration and atomic artifact export |
| `backend/src/data_pipeline/prepare_real_data.py` | Thin command-line entry point |

### Tests and quality

```bash
make backend-test
PYTHONPATH=backend/src .venv/bin/python -m ruff check backend/src backend/tests
```

The tests cover the generated artifact contract, attribute similarity and the
analogue-pooled demand pipeline.

### Model boundaries

**Machine learning** (`backend/src/machine_learning/`) owns explainable
attribute similarity, analogue-pooled censoring-corrected predictive demand
estimation, and newsvendor-solved order quantities against a planner
sell-through target:

- `model.py` — attribute similarity, analogue pooling and match confidence.
- `demand.py` — the predictive demand estimator: a hierarchically shrunk
  rate prior (group fit blended toward its parent by row count, no hard
  cutoff), skew-corrected lognormal point estimates, and per-item-type buy
  ceilings derived from each item type's own observed order/sales history
  rather than one fixed constant.

**Fashion matching** owns protected image fetching and validation,
FashionCLIP-compatible inference and embedding generation, and candidate
retrieval with DINO reranking and colour/pattern gating — see
[FASHION_MATCHING.md](FASHION_MATCHING.md) for the full retrieval workflow.

## Frontend

The independent Next.js planner application. It reads
`frontend/app/generated-data.json` — the artifact produced by the backend
pipeline via `make data`/`make data-vision` — and does not import backend
modules or call a backend API at runtime.

```bash
npm install
npm run dev
```

Run these commands from `frontend/`, or use `make frontend-dev` from the
repository root.

Validation:

```bash
npm run lint
npx tsc --noEmit
npm test
```

The production data path is:

```text
XLSB ingestion → source validation → identifier/fabric cleaning
→ canonical snake_case schema → processed CSV export
→ feature engineering → ML training/validation → frontend artifact
```

The active AI evidence is limited to five business attributes: Item, Design,
Colour, Category Type and Fabric. Season is retained only for tracing and
temporal validation; historical outcomes remain prediction targets rather than
similarity inputs.
