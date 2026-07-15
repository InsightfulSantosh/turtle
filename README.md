# Turtle Season Intelligence AI

A client-ready local AI pilot plus a scale service for seasonal merchandise planning.
The local demo uses the supplied sample artifact. The deployable service adds
FashionCLIP embeddings, pgvector retrieval, learning-to-rank, quantile demand,
hierarchical reconciliation, operational constraints, batch jobs, and feedback.

## What the demo includes

- 167 upcoming styles and 33 historical styles from the supplied sample files
- pretrained deep-vision similarity across the supplied product images
- attribute/vision weights and neighbour count selected by validation
- top historical analogues with component-level match evidence
- analogue and regularized-regression demand ensemble
- finite-sample uncertainty range and data-quality guardrails
- planner overrides, portfolio review, and CSV export
- versioned Python API and container definition in `ml-service`
- a PostgreSQL/pgvector scale path sized for 200,000–500,000 catalogue items
- separate API, worker, FashionCLIP service, model-training jobs, and audit schema

## Run locally

```bash
npm install
npm run dev
```

The generated sample data is stored in `app/generated-data.json`. The source
workbooks remain outside this application directory.

## Refresh source data and AI model

The preprocessing script expects the reviewed workbooks in the parent project
and read-only converted XLSB copies in `../tmp/converted`.

```bash
python scripts/prepare_sample_data.py downloads
curl -L --config ../tmp/vision-images.curl.conf
PYTHON=python3 ./ml-service/tools/build_deep_features.sh
```

The model artifact is stored in `app/generated-data.json`, so the local client
demo does not require a live model server. The production API contract,
container, tests, and model documentation are in `ml-service`.

The architecture is production-oriented, but the fitted quantity model is
correctly labelled as a pilot: only 33 historical outcomes were supplied. A
production calibration requires three to five clean seasons, consistent sales
windows, stock-out and replenishment signals, markdowns, channel context, MOQ,
and budget constraints.

## Run the scalable stack

Copy `ml-service/.env.scale.example` to `ml-service/.env.scale`, set strong
secrets and the client's approved image domains, then run:

```bash
docker compose --env-file ml-service/.env.scale -f docker-compose.scale.yml up --build
```

The API is then available at `http://localhost:8080/docs`. Its readiness endpoint
is `GET /v2/health/ready`. Scale endpoints include single and batch recommendations,
catalog ingestion jobs, job status, and planner similarity feedback.

`MODEL_POLICY=require_trained` deliberately prevents production startup until
the CatBoost ranker and P10/P50/P90 LightGBM artifacts have been trained and
mounted. The embedding container likewise expects the approved fine-tuned model
at `models/fashion-clip`. Keep `allow_fallback` and the public base model only
for integration testing.
