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
├── Makefile                          Central developer commands
└── CLIENT_DEMO_GUIDE.md              Model and client-demo explanation
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
make backend-test    # run Python tests
make frontend-test   # build and test the Next.js application
make test            # run both test suites
```

The frontend runs by itself because it reads the generated JSON artifact
directly; there is no live backend service to run alongside it.

## Backend responsibilities

- **Data pipeline:** validates identifiers and source columns, converts the
  workbooks and produces normalized historical/upcoming records.
- **Fashion matching:** FashionCLIP-compatible image encoding, DINO
  reranking and colour/pattern gated visual retrieval.
- **Machine learning:** attribute similarity, analogue-pooled predictive
  demand estimation and newsvendor-solved order quantities.

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
