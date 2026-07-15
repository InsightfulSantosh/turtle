# Turtle Season Intelligence ML service

This service separates model training and inference from the planner interface.
It provides a versioned recommendation API, model card, health endpoint,
request tracing, optional API-key enforcement, data guardrails, and reproducible
container deployment.

## Model

- Deep image representation: FashionCLIP 2.0 512-dimensional image embeddings for the local pilot.
- Attribute evidence: weighted categorical, family, token, and price similarity.
- Retrieval: attribute/vision weight and neighbour count selected by out-of-fold validation.
- Demand: sell-through-normalized analogue demand blended with regularized regression.
- Risk: finite-sample conformal prediction interval plus MOQ/case-pack limits.

The pilot is intentionally marked as limited-data. It has 33 historical outcomes,
so it uses leave-one-out validation. A production fit should use three to five
clean seasons and a temporal holdout, then replace the local image provider with
a client-tuned FashionCLIP embedding service after reviewed match pairs are available.

## Scalable v3 path (200,000–500,000 items)

The v3 path is implemented alongside the sample runtime:

- metadata-filtered pgvector HNSW retrieves up to 200 candidates without an all-pairs join;
- FashionCLIP produces 512-dimensional image/text vectors in an isolated service;
- CatBoost learning-to-rank re-orders candidates using visual, attribute, price,
  outcome-reliability, and planner-feedback features;
- P10/P50/P90 LightGBM models forecast demand with temporal holdouts;
- MinTrace reconciliation makes category/channel/region forecasts coherent;
- MOQ, pack, supplier-capacity, maximum-buy, and budget constraints produce the final buy;
- PostgreSQL `SKIP LOCKED` jobs handle large ingestion and recommendation batches;
- planner feedback and recommendation history create an auditable learning loop.

The code can run now, but trained v3 artifacts cannot be fitted honestly from 33
rows. In production set `MODEL_POLICY=require_trained`; startup then fails closed
if the ranker or demand artifacts are absent.

## Refresh the model artifact on macOS

```bash
../.venv/bin/pip install -r requirements-fashionclip.txt
HF_HOME=../.model-cache PYTHON=../.venv/bin/python ./tools/build_fashion_clip_features.sh
```

This reads the already downloaded source images, generates unit-normalized
FashionCLIP image vectors, computes cosine distances, tunes the ensemble using
out-of-fold predictions, and refreshes the frontend model artifact. The exported
metadata records the exact checkpoint revision, vector dimension, execution
device, and image coverage for auditability.

## Run the API

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
TURTLE_API_KEY=change-me .venv/bin/uvicorn service:app --host 0.0.0.0 --port 8080
```

Endpoints:

- `GET /healthz`
- `GET /v1/model`
- `GET /v1/recommendations/{item_id}`
- `POST /v1/recommendations`
- `GET /v2/health/ready`
- `POST /v2/recommendations`
- `POST /v2/recommendations:batch`
- `POST /v2/catalog/items:batch`
- `GET /v2/jobs/{job_id}`
- `POST /v2/feedback/similarity`
- `POST /v2/recommendations/{request_id}/decision`

For new products, the caller supplies FashionCLIP similarities keyed by historical
item ID. In production that map is produced by the image-embedding service.

## Training commands

The three training programs use the latest season as the holdout and refuse
datasets with fewer than three seasons:

```bash
python -m training.fine_tune_fashion_clip pairs.csv ../models/fashion-clip
python -m training.train_ranker ranker.parquet ../models/ranker.cbm
python -m training.train_demand demand.parquet ../models/demand
python -m training.build_hierarchy series.parquet ../models/hierarchy.json
```

Run these from `ml-service` after installing `requirements-training.txt`.
