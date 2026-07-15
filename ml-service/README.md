# Turtle Season Intelligence ML service

This service separates model training and inference from the planner interface.
It provides a versioned recommendation API, model card, health endpoint,
request tracing, optional API-key enforcement, data guardrails, and reproducible
container deployment.

## Model

- Deep image representation: Apple Vision FeaturePrint v2 for the local pilot.
- Attribute evidence: weighted categorical, family, token, and price similarity.
- Retrieval: attribute/vision weight and neighbour count selected by out-of-fold validation.
- Demand: sell-through-normalized analogue demand blended with regularized regression.
- Risk: finite-sample conformal prediction interval plus MOQ/case-pack limits.

The pilot is intentionally marked as limited-data. It has 33 historical outcomes,
so it uses leave-one-out validation. A production fit should use three to five
clean seasons and a temporal holdout, then replace the local image provider with
a containerized FashionCLIP or SigLIP embedding service.

## Refresh the model artifact on macOS

```bash
PYTHON=python3 ./tools/build_deep_features.sh
```

This reads the already downloaded source images, computes neural image distances,
tunes the ensemble using out-of-fold predictions, and refreshes the frontend model artifact.

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

For new products, the caller supplies deep-vision similarities keyed by historical
item ID. In production that map is produced by the image-embedding service.
