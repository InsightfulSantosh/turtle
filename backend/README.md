# Turtle Season Intelligence Backend

The versioned Qdrant catalogue index and image-first retrieval workflow are
documented in [../FASHION_MATCHING.md](../FASHION_MATCHING.md).

The backend is a Python 3.12 src-layout data-science package with no live
API — it is a CLI-driven pipeline that writes a versioned browser artifact
consumed directly by the frontend. It keeps data preparation, visual
matching and machine learning concerns separate while sharing one
configuration.

## Backend map

| Location | Responsibility |
|---|---|
| `src/core/` | Environment and filesystem configuration |
| `src/data_pipeline/` | Real workbook ingestion, validation and artifact export |
| `src/fashion_matching/` | Visual retrieval, image encoding and validation |
| `src/machine_learning/` | Analogue similarity and predictive demand estimation |

## Local setup

Run from the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements-dev.txt
```

## Rebuild real data

```bash
make data
```

To map local product images, generate fashion-domain image embeddings and
rebuild the browser artifact with visual similarity:

```bash
.venv/bin/pip install -r backend/requirements-fashion-matching.txt
make data-vision
```

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

### Pipeline modules

| File | Pipeline responsibility |
|---|---|
| `src/data_pipeline/settings.py` | Source, temporary and artifact paths |
| `src/data_pipeline/ingestion.py` | Workbook discovery, conversion and row loading |
| `src/data_pipeline/validation.py` | Post-cleaning identifier and fabric quality gates |
| `src/data_pipeline/preprocessing.py` | Identifier and fabric-family normalization rules |
| `src/data_pipeline/schema.py` | Canonical column mappings and schema-drift checks |
| `src/data_pipeline/feature_engineering.py` | Model-ready historical and upcoming records |
| `src/data_pipeline/pipeline.py` | End-to-end orchestration and atomic artifact export |
| `src/data_pipeline/prepare_real_data.py` | Thin command-line entry point |

Path overrides:

```text
TURTLE_DATA_ROOT       parent directory for raw inputs and processed outputs
TURTLE_TEMP_ROOT       disposable workbook-conversion directory
TURTLE_MODEL_ARTIFACT  generated artifact destination
```

## Tests and quality

```bash
make backend-test
PYTHONPATH=backend/src .venv/bin/python -m ruff check backend/src backend/tests
```

The tests cover the generated artifact contract, attribute similarity and the
analogue-pooled demand pipeline.

## Model boundaries

### Machine learning

The classical ML layer owns:

- explainable attribute similarity;
- analogue-pooled, censoring-corrected predictive demand estimation;
- newsvendor-solved order quantities against a planner sell-through target.

### Fashion matching

The visual-matching layer owns:

- protected image fetching and validation;
- FashionCLIP-compatible inference and embedding generation;
- candidate retrieval, DINO reranking and colour/pattern gating.

See [../FASHION_MATCHING.md](../FASHION_MATCHING.md) for the full retrieval
workflow.
