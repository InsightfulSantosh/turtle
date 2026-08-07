# Turtle Season Intelligence

Turtle Season Intelligence is a decision-support workspace for planning an
upcoming fashion season. It turns historical ordering and sell-through data,
upcoming catalogue records, and matched product images into a static planner
that helps users review visual analogues, forecast demand, and choose an
initial order quantity.

It is intentionally not a live backend service. A Python pipeline produces a
versioned JSON artifact; the independent Next.js application reads that file
directly. This keeps every number shown in the browser traceable to a specific
local data build.

> Forecasts are planning evidence, not production certification. Image-match
> thresholds, model revisions, and demand uncertainty should be reviewed before
> commercial decisions are made.

## What the project does

- Ingests the supplied historical and SS27 `.xlsb` workbooks.
- Validates, cleans, and standardizes catalogue data to a common schema.
- Matches local product images to catalogue identifiers.
- Retrieves visually similar historical products using FashionSigLIP, then
  verifies colour, texture, pattern, and construction with DINOv2-based
  evidence.
- Estimates demand from accepted visual analogues, correcting for historical
  exposure and stock constraints, then recommends a pack-rounded initial buy
  for a target sell-through.
- Publishes the planner artifact consumed by the frontend—no runtime API or
  browser-to-Python connection is required.

The visual retrieval design, indexing workflow, score semantics, and evaluation
guidance are documented in [FASHION_MATCHING.md](FASHION_MATCHING.md).

## Architecture

```text
Raw workbooks + product images
             |
             v
Python data pipeline
  ingestion -> validation -> cleaning -> feature engineering
             |
             +--> optional visual retrieval and reranking
             |
             v
Demand model + order recommendation
             |
             v
frontend/app/generated-data.json
             |
             v
Next.js planner and same-origin product-image route
```

```text
turtle/
├── backend/
│   ├── src/core/                 Shared project paths and environment overrides
│   ├── src/data_pipeline/        Workbook ingestion, validation and export
│   ├── src/fashion_matching/     Image retrieval, encoders and evaluation
│   ├── src/machine_learning/     Demand forecast and order-quantity logic
│   └── tests/                    Python unit and contract tests
├── frontend/
│   ├── app/                      Next.js planner, image route and JSON artifact
│   ├── public/                   Static assets
│   └── tests/                    Browser-artifact contract tests
├── DATA/                         Local input and generated data; ignored by Git
│   ├── raw/                      Source workbooks and matched product images
│   └── processed/                Cleaned CSVs and validation report
├── .env.example                  Visual matching and optional Qdrant settings
├── requirements.txt              Backend dependency source of truth
├── Makefile                      Common developer commands
├── FASHION_MATCHING.md           Detailed visual-matching documentation
└── Business_Process_Document_AI_Order_Quantity_Tool.pdf
```

## Prerequisites

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/) for the Python environment
- Node.js 22.13 or newer and npm
- LibreOffice, with `soffice` available on `PATH`, to convert the supplied
  `.xlsb` workbooks to `.xlsx`

On macOS with Homebrew:

```bash
brew install uv
brew install --cask libreoffice
which soffice
```

`make data` and `make data-vision` fail early if `soffice` is unavailable.

## Quick start

From the repository root:

```bash
uv venv .venv
uv pip install -r requirements.txt
npm --prefix frontend ci

# Required only when rebuilding the visual artifact.
cp .env.example .env
set -a && source .env && set +a

make data-vision
make frontend-dev
```

Open [http://localhost:3000](http://localhost:3000). The frontend serves local
catalogue images itself, so a separate Python API process is not needed.

The first visual build downloads the configured encoder weights through the
normal Hugging Face cache. It can take appreciably longer than subsequent
builds; CPU builds are suitable for development but may be slow for a full
catalogue.

## Required local data

`DATA/` is deliberately ignored by Git. To run a real-data build, create this
layout (or point `TURTLE_DATA_ROOT` at an equivalent directory):

```text
DATA/
└── raw/
    ├── LAST SEASONES ORDERING & SALE THRU DATA.xlsb
    ├── SEG WISE SS27 MASTER SHEET TILL.xlsb
    ├── historical_matched_images/
    │   └── <historical-product-id>.jpg|jpeg|png|webp
    └── upcoming_ss27_matched_images/
        └── <upcoming-product-id>.jpg|jpeg|png|webp
```

Image filenames are matched case-insensitively by filename stem to their
catalogue product ID. The browser image route supports JPG, JPEG, PNG, and
WebP. Missing images do not stop the data-cleaning pipeline, but visual
retrieval needs matched images; the planner prioritizes image-backed upcoming
styles and analogues.

The pipeline reads `Sheet1` from both workbooks. It accepts the current
catalogue-oriented headers as well as the earlier compatible historical layout,
then writes these local outputs:

```text
DATA/processed/historical_cleaned.csv
DATA/processed/upcoming_cleaned.csv
DATA/processed/data_cleaning_validation.csv
tmp/real-data-converted/                    # cached workbook conversions
frontend/app/generated-data.json            # browser artifact
```

During preparation it removes zero-sales historical rows, removes upcoming item
types with no retained historical coverage, normalizes identifiers and fabric
families, and records data-quality counts in the validation report and artifact
metadata.

## Build the planner artifact

### Standard data build

```bash
make data
```

This validates and processes the workbooks, maps product images, fits the demand
model, and writes `frontend/app/generated-data.json`. It does not create image
embeddings, so it is useful for validating data inputs but is not the normal
visual-planner build.

### Visual production-style build

```bash
make data-vision
```

This additionally generates FashionSigLIP embeddings for mapped product images,
retrieves candidates from the same item type, and reranks them with image-based
colour, texture, pattern, and DINOv2 construction evidence. The model uses
visual ranking only by default; workbook attributes and descriptions are not
mixed into its score.

For an isolated preview without overwriting the active browser artifact:

```bash
PYTHONPATH=backend/src .venv/bin/python -m data_pipeline.prepare_real_data \
  --with-vision --item-type OTSH \
  --output frontend/app/generated-data-otsh-preview.json
```

For a deterministic, balanced sample of the upcoming catalogue:

```bash
PYTHONPATH=backend/src .venv/bin/python -m data_pipeline.prepare_real_data \
  --with-vision --sample-size 200 --sample-seed 27 --verbose \
  --log-file tmp/data-vision-progress.log
```

`--item-type` and `--sample-size` are mutually exclusive. A sample is balanced
first across item type and then category type; the full historical catalogue
remains available as the analogue pool. Use `--output` for any review artifact
so the live frontend file is not replaced accidentally.

## Forecast and recommendation policy

Only visual analogues that meet the minimum visual score (default `0.50`) and
reach at least Medium confidence can produce a recommendation. Otherwise the
artifact reports no suitable match and returns no sales forecast or order
quantity for that product.

For accepted matches, the backend:

1. pools accepted visual analogues rather than simply copying one product's
   sales;
2. estimates a censoring-aware weekly demand rate from historical sales,
   inventory exposure, and age;
3. shrinks sparse evidence toward hierarchical category priors;
4. produces a lognormal demand distribution and uncertainty bands; and
5. solves a newsvendor order quantity for the current target sell-through,
   rounded to packs of 25.

Buy ceilings are derived per item type from the historical data. A ceiling is a
policy guardrail, not evidence that the forecast has reached the chosen
sell-through target. The UI exposes the selected analogue, forecast range,
confidence, evidence diagnostics, and capped-quantity status so a planner can
review the recommendation.

## Configuration

Copy `.env.example` to `.env` and load it into the shell before a visual build:

```bash
cp .env.example .env
set -a && source .env && set +a
```

Important configuration groups:

| Setting | Purpose |
|---|---|
| `TURTLE_DATA_ROOT` | Overrides the default `DATA/` location for both pipeline and frontend image route. |
| `TURTLE_TEMP_ROOT` | Overrides `tmp/`, including converted-workbook cache. |
| `TURTLE_MODEL_ARTIFACT` | Overrides the generated JSON destination for Python builds. |
| `FASHION_MATCHING_MODEL_ID`, `FASHION_MATCHING_MODEL_REVISION` | Primary visual encoder and revision. Pin an exact revision before production use. |
| `FASHION_MATCHING_DEVICE`, `FASHION_MATCHING_BATCH_SIZE` | Inference device (`auto`, `cuda`, `mps`, or `cpu` as supported) and memory/throughput control. |
| `FASHION_DINO_*` | DINO reranking, candidate count, item-type constraint, and model revision. |
| `FASHION_PATTERN_*`, `FASHION_COLOUR_*` | Image-based acceptance gates used by the visual reranker. |
| `FASHION_MINIMUM_SCORE` | Minimum acceptable visual score; defaults to `0.50`. |
| `FASHION_APPEARANCE_*_WEIGHT` | Neural, colour, and texture blend weights; they must sum to `1`. |
| `QDRANT_*`, `FASHION_COLLECTION_*` | Optional standalone Qdrant indexing and matching workflow. |
| `ALLOWED_IMAGE_DOMAINS` | HTTPS allowlist for remote manifest images; local planner images do not need it. |

The Next.js image route uses `TURTLE_DATA_ROOT` too. Keep it consistent with
the Python build location when running the frontend outside the repository
root. The frontend calls no live API: it reads the generated artifact directly
and serves product images from its own route handler.

## Run and test

| Command | Result |
|---|---|
| `make frontend-dev` | Starts the planner at `http://localhost:3000`. |
| `make frontend-build` | Creates an optimized Next.js production build. |
| `make backend-test` | Runs Python tests. |
| `make frontend-test` | Builds the frontend and runs its artifact contract test. |
| `make test` | Runs both backend and frontend test suites. |
| `make data` | Rebuilds the artifact without visual embeddings. |
| `make data-vision` | Rebuilds the artifact with visual matching and verbose progress logging. |
| `make fashion-index MANIFEST=...` | Builds and activates an optional Qdrant historical-image index. |
| `make fashion-match MANIFEST=... OUTPUT=...` | Runs optional Qdrant manifest matching. |
| `make fashion-evaluate RESULTS=... LABELS=...` | Evaluates retrieved results against reviewed relevance labels. |

Additional frontend checks, from `frontend/`:

```bash
npm run lint
npx tsc --noEmit
npm test
```

Additional backend linting, from the repository root:

```bash
PYTHONPATH=backend/src .venv/bin/python -m ruff check backend/src backend/tests
```

## Frontend behavior

The planner has comparison and portfolio views. It lets a user inspect the
selected upcoming style, the ranked historical visual evidence, uncertainty
bands, sales expectation, and recommended initial order. Changing the minimum
visual similarity or target sell-through recalculates the recommendation in the
browser from the published demand-model data; it does not retrain the backend
model.

`frontend/app/generated-data.json` is a generated, versioned contract. Do not
hand-edit it: rebuild it through the pipeline and keep the frontend contract
test passing. Product images are delivered by
`/product-images/{historical|upcoming}/{product-id}`, which only reads the two
approved local image folders and sends safe image content types.

## Optional standalone fashion retrieval

The data-vision planner build uses local FAISS/NumPy retrieval and does not
need Qdrant. Qdrant supports a separate, versioned catalogue-index workflow for
experimentation or service-oriented use. It requires a Qdrant instance and a
manifest; see [FASHION_MATCHING.md](FASHION_MATCHING.md) for manifest format,
index activation, remote-image controls, evaluation labels, and rollback
guidance.

## Limitations and responsible use

- The visual score threshold and reranker weights are provisional until they
  are calibrated against time-separated, human-reviewed catalogue matches.
- A visual analogue is evidence, not a guarantee of demand; sparse analogue
  pools deliberately yield wider uncertainty.
- The current artifact is a static snapshot. Rebuild after changing source
  workbooks, images, model configuration, or operational assumptions.
- Model downloads and visual inference may require substantial local disk,
  memory, and time. Pin reviewed model revisions before production use.
- Raw data, processed data, temporary conversions, model caches, and outputs
  are excluded from Git to avoid committing commercial or large binary assets.
