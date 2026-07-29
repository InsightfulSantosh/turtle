# Fashion catalogue matching

This repository includes an image-first catalogue matcher for retrieving last
season's visually similar products. Image, text and structured attributes are
kept as separate, auditable signals. Product IDs, sales and operational outcomes
never enter an embedding.

## Architecture

```text
historical manifest
  -> validation + EXIF/RGB/aspect-ratio preprocessing
  -> image embedding
  -> optional description embedding (separate named vector)
  -> versioned Qdrant collection
  -> explicit alias activation

upcoming image
  -> identical preprocessing
  -> image top-N candidates
  -> optional text top-N candidates
  -> candidate union
  -> structured-attribute scoring
  -> available-signal weight renormalization
  -> product-level deduplication
  -> threshold + top-K result
```

The default provisional encoder is `Marqo/marqo-fashionSigLIP`. The encoder is
configuration-driven so FashionCLIP can remain a benchmark and newer compatible
Hugging Face or OpenCLIP fashion encoders can be evaluated in a separate
collection. Do not activate a model merely because it wins a public benchmark;
select it with labelled, catalogue-specific evaluation.

For the real-data planner artifact, the production visual path is two-stage:

```text
FashionSigLIP retrieves the top 50 same-item-type candidates
  -> DINOv2 evaluates fine visual detail only inside that shortlist
  -> FashionSigLIP and DINOv2 each contribute 50% of visual similarity
  -> structured attributes and the calibrated visual score rank the final analogue set
```

The category constraint is relaxed only when a category has fewer than two
image-backed historical candidates. If DINOv2 is disabled, unavailable, or its
configured weight is `0`, the planner retains the FashionSigLIP
baseline; a partial reranker is never allowed to introduce a new candidate.
Set `FASHION_DINO_MODEL_REVISION` to an exact Hugging Face commit before a
production rebuild, just as for the primary encoder.

## Installation

Python 3.12 is required. Install the matching extras into the existing virtual
environment:

```bash
.venv/bin/pip install -r backend/requirements-fashion-matching.txt
cp .env.example .env
set -a
source .env
set +a
```

Before production indexing, replace `FASHION_MATCHING_MODEL_REVISION=main` with
an exact Hugging Face commit hash. Models using remote model code must be
security-reviewed and pinned before deployment.

Device configuration:

- `auto` selects CUDA, then Apple MPS, then CPU.
- `cuda` requires a compatible PyTorch/CUDA runtime.
- CPU works but is intended for small catalogues or development.
- Reduce `FASHION_MATCHING_BATCH_SIZE` if model inference exhausts memory.

Model weights are downloaded through the model library cache and are excluded
from Git. Set the normal Hugging Face cache environment variables when a
deployment needs a dedicated persistent cache.

## Manifests

Templates are provided in:

- `backend/examples/historical_catalog.template.csv`
- `backend/examples/upcoming_catalog.template.csv`
- `backend/examples/fashion_labels.template.csv`

Required image-manifest fields:

| Field | Meaning |
|---|---|
| `product_id` | Stable product identifier, used only in payload/output |
| `image_id` | Unique identifier for this image |
| `image_path` or `image_url` | Exactly one image source |
| `view` | Optional front/back/detail view |

Optional text fields are `title`, `description` and `text`. They are combined
only for the separate text vector. Optional structured fields include category,
colour, pattern, material, fabric, design, fit, gender and brand.

Relative image paths are resolved relative to the manifest. Remote URLs must use
HTTPS and match `ALLOWED_IMAGE_DOMAINS`; private, loopback, link-local, reserved
and multicast addresses are rejected.

Do not add sales, orders, dispatch, sell-through, product IDs or image IDs to
description fields. Sales outcomes belong exclusively to downstream demand
forecasting.

## Start Qdrant

```bash
docker compose -f docker-compose.qdrant.yml up -d
```

Qdrant data is stored in the named `qdrant_data` volume. For managed Qdrant,
configure `QDRANT_URL` and `QDRANT_API_KEY` instead.

## Index historical images

```bash
PYTHONPATH=backend/src .venv/bin/python -m fashion_matching.index_catalog \
  --manifest data/historical_catalog.csv \
  --activate \
  --failure-report outputs/index-failures.json
```

Indexing is idempotent. Every image stores its SHA-256 checksum, model ID, exact
resolved revision, dimension, preprocessing version and combined content
checksum. Re-running the same manifest skips unchanged points. Image or metadata
changes are updated, and failures do not stop the rest of the manifest.

Collections are versioned from model ID, resolved revision, dimension and
preprocessing version. `--activate` switches the configured alias only after an
error-free index. `--activate-with-failures` is available but should be used
only after reviewing the machine-readable failure report.

To roll back, point the alias to the previous versioned collection using
Qdrant's alias API or re-run the previous pinned configuration with
`--activate`. Old collections are not deleted automatically.

## Match upcoming images

Manifest:

```bash
PYTHONPATH=backend/src .venv/bin/python -m fashion_matching.match \
  --query-manifest data/upcoming_catalog.csv \
  --top-k 5 \
  --output outputs/matches.json
```

Directory:

```bash
PYTHONPATH=backend/src .venv/bin/python -m fashion_matching.match \
  --query-directory data/upcoming-images \
  --output outputs/matches.json
```

Single image:

```bash
PYTHONPATH=backend/src .venv/bin/python -m fashion_matching.match \
  --query-image data/upcoming/front.jpg \
  --query-product-id UPCOMING-001 \
  --query-image-id UPCOMING-001-FRONT \
  --output outputs/matches.json
```

JSON, CSV and a separate failed-query report are written. One bad query does not
stop the remaining manifest.

## Scores

Default provisional weights are:

| Signal | Weight |
|---|---:|
| Image | 0.70 |
| Structured attributes | 0.20 |
| Description text | 0.10 |

These values are configuration defaults, not validated business constants. If a
signal is missing, its weight is explicitly removed and the remaining weights
are renormalized. The output includes the individual scores and actual weights
used for every result. No fake zero score is inserted.

The provisional minimum final score is `0.62`. Results below it return
`no_suitable_match=true` with no product matches. In the browser artifact, a
candidate must also reach Medium match confidence and a calibrated visual score
of at least `0.50`; otherwise no historical product is shown and the demand
forecast uses the regression model without analogue blending.

Cosine image/text scores are mapped from `[-1, 1]` to `[0, 1]`. Structured
attribute similarity is the fraction of populated shared fields that match
after canonicalization. For production, replace provisional weight selection
and generic cosine mapping only with calibration fitted on a held-out,
time-separated relevance set.

## Evaluation

Create reviewed labels from the provided template. Relevance can be binary or
graded. A query with no acceptable historical product should set
`no_match=true`.

```bash
PYTHONPATH=backend/src .venv/bin/python -m fashion_matching.evaluation \
  --results outputs/matches.json \
  --labels data/fashion-labels.csv \
  --output outputs/evaluation.json
```

The evaluator reports Recall@1/3/5/10, MRR, NDCG@5, no-match accuracy and
mean/P50/P95 latency. Indexing summaries provide elapsed time and counts needed
to calculate indexing throughput. Matching result timings can be aggregated for
query throughput.

Compare at least:

1. `patrickjohncyh/fashion-clip` image-only baseline
2. `Marqo/marqo-fashionSigLIP`
3. a reviewed compatible challenger such as MODA
4. each winner with and without text/attribute reranking

Each model/revision gets a different Qdrant collection, so incompatible vectors
cannot be mixed.

## Tests

Normal tests use a deterministic lightweight encoder and do not download model
weights:

```bash
cd backend
PYTHONPATH=src ../.venv/bin/python -m pytest -q
```

Real model and Qdrant smoke tests should run separately in a GPU/integration
environment with pinned model revisions.

## Known limitations and next steps

- Uniform-background cropping is deterministic but is not semantic garment
  segmentation. It is disabled by default and must be evaluated before use.
- Current attribute scoring treats shared populated fields uniformly. Learn
  category-specific importance only after collecting reviewed labels.
- The default weights are provisional.
- Query embedding calls are currently isolated per failed-safe query. Catalogue
  indexing is batched; high-QPS serving should add a bounded dynamic batcher.
- Qdrant integration requires the optional dependency and a running service.
- Thresholds must be calibrated on genuine positive and no-match examples.
- The browser artifact can be rebuilt with mapped local images and visual
  similarity using `make data-vision`. Its planner workspace intentionally
  displays only upcoming products and historical analogues with mapped images.
