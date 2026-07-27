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
├── backend/                          Python services and data-science code
│   ├── api/                          Main and embedding FastAPI applications
│   ├── src/
│   │   ├── ai/                       End-to-end recommendation orchestration
│   │   ├── core/                     Central configuration and paths
│   │   ├── data_pipeline/            Workbook validation and preparation
│   │   ├── deep_learning/            Image/text embeddings and fine-tuning
│   │   ├── domain/                   Shared business contracts
│   │   └── machine_learning/         Ranking, demand and optimization models
│   └── tests/                        Backend unit and contract tests
├── Makefile                          Central developer commands
└── CLIENT_DEMO_GUIDE.md              Model and client-demo explanation
```

The dependency direction is intentional:

```text
backend/api
      ↓
AI orchestration
      ↓
ML + deep learning
      ↓
domain contracts + core configuration
```

Frontend code does not import backend internals. The backend writes the
versioned browser artifact to `frontend/app/generated-data.json`.

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
- Node.js 22.13 or newer
- npm
- LibreOffice for `.xlsb` conversion

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements-dev.txt
npm --prefix frontend install
```

## Common commands

```bash
make data            # rebuild the frontend artifact from the real workbooks
make frontend-dev    # run the planner at http://localhost:3000
make backend-api     # run FastAPI at http://localhost:8080
make backend-test    # run Python tests
make frontend-test   # build and test the Next.js application
make test            # run both test suites
```

The frontend can run by itself because it reads the generated JSON artifact.
The API is required only for live service integration.

## Backend responsibilities

- **Data pipeline:** validates identifiers and source columns, converts the
  workbooks and produces normalized historical/upcoming records.
- **Machine learning:** attribute matching, candidate ranking, demand
  forecasting, hierarchy reconciliation and constrained ordering.
- **Deep learning:** FashionCLIP-compatible image/text embeddings and
  fine-tuning workflows.
- **AI orchestration:** combines matching, demand forecasting and inventory
  policy into an explainable recommendation.

See [backend/README.md](backend/README.md) for backend commands and training
contracts.

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
