# Turtle Season Intelligence Backend

The versioned Qdrant catalogue index and image-first retrieval workflow are
documented in [../FASHION_MATCHING.md](../FASHION_MATCHING.md).

The backend has a separate FastAPI application layer and a Python 3.12
src-layout data-science package. It keeps API transport, data preparation,
machine learning and deep learning concerns separate while sharing one
configuration and one set of domain contracts.

## Backend map

| Location | Responsibility |
|---|---|
| `api/` | Main and embedding FastAPI routes, validation and authentication |
| `src/ai/` | Recommendation workflow that combines all model components |
| `src/core/` | Environment and filesystem configuration |
| `src/data_pipeline/` | Real workbook ingestion, validation and artifact export |
| `src/deep_learning/` | Image/text providers and model-training workflows |
| `src/domain/` | Typed business and recommendation contracts |
| `src/machine_learning/` | Similarity, ranking, forecasting, hierarchy and optimization |

Training entry points are grouped with the model family they train:

- `machine_learning/training/` contains demand, ranker and hierarchy training.
- `deep_learning/training/` contains FashionCLIP fine-tuning.

## Local setup

Run from the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements-dev.txt
```

For the complete model-training dependency set:

```bash
.venv/bin/pip install -r backend/requirements-training.txt
```

## Run

```bash
make backend-api
```

Equivalent direct command:

```bash
PYTHONPATH=backend/src .venv/bin/python -m uvicorn \
  --app-dir backend api.main:app --host 0.0.0.0 --port 8080
```

The API reads `frontend/app/generated-data.json` by default. Override it with
`TURTLE_MODEL_ARTIFACT`.

The optional image/text embedding API is also separated under `backend/api`:

```bash
.venv/bin/pip install -r backend/requirements-deep-learning.txt
PYTHONPATH=backend/src .venv/bin/python -m uvicorn \
  --app-dir backend api.embedding_service:app --host 0.0.0.0 --port 8090
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

The tests cover the generated artifact contract, attribute similarity,
scikit-learn demand pipeline, ranking, buy constraints and hierarchy
reconciliation.

## Model boundaries

### Machine learning

The classical ML layer owns:

- explainable attribute similarity;
- CatBoost-compatible candidate ranking;
- scikit-learn and LightGBM-compatible demand forecasts;
- forecast hierarchy reconciliation;
- pack, budget, MOQ and capacity-aware order optimization.

### Deep learning

The deep-learning layer owns:

- image/text embedding generation;
- protected image fetching and validation;
- FashionCLIP-compatible inference;
- contrastive fine-tuning workflows.

### AI orchestration

`ai/recommendation_engine.py` loads the current artifact and coordinates product
matching, demand forecasting and inventory-policy calculations. It does not
contain model-training code or HTTP request schemas.

## Training entry points

Run from the repository root with `PYTHONPATH=backend/src`:

```bash
.venv/bin/python -m deep_learning.training.fine_tune_fashion_clip \
  pairs.csv models/fashion-clip
.venv/bin/python -m machine_learning.training.train_ranker \
  ranker.parquet models/ranker.cbm
.venv/bin/python -m machine_learning.training.train_demand \
  demand.parquet models/demand
.venv/bin/python -m machine_learning.training.build_hierarchy \
  series.parquet models/hierarchy.json
```

Training requires multi-season datasets with a forward temporal holdout.
Fallback algorithms are for integration and baseline analysis; they must not
be represented as approved production models. Database-backed scale
infrastructure is intentionally deferred and can be introduced in a future
phase.
