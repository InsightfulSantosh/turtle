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
Input product image
  -> source item type forms the strict eligibility cohort
  -> FashionSigLIP retrieves the top 50 eligible candidates through FAISS
  -> four dominant garment colours are extracted from the central foreground region
  -> CIEDE2000 perceptual distance rejects visually different palettes
  -> multi-scale DINOv2 verifies visible pattern and construction
  -> a pattern gate excludes mismatched checks, stripes, prints and structured fabrics
  -> texture evidence captures surface and print detail
  -> the visual reranker combines neural, colour-palette and texture evidence
```

Item type is the only workbook field used as a strict retrieval constraint.
Workbook colour labels do not create cohorts or influence visual ranking. Colour
matching comes from the product images themselves using dominant palettes in
CIELAB space and CIEDE2000 distance. The displayed source image is never masked
or altered; foreground suppression is used only inside the colour measurement.
OTTR uses an audited waist-to-lower-leg trouser region for FashionSigLIP,
DINO, palette and texture analysis so shirts, shoes, logos and swatches do not
dominate the measurement. For OTTR only, the hard pattern gate applies to
Checks, Prints, Stripes and Dobby/Structure; plain trousers retain DINO evidence
in ranking but are not rejected solely by the pattern-distance gate. All other
item types keep the standard full-image visual path and gate behavior.
FAISS `IndexFlatIP` is used where the runtime provides it; the local development
fallback is an exact NumPy inner-product search, so the ranking remains
deterministic and correct.
If DINOv2 is disabled, unavailable, or its configured weight is `0`, the
planner retains the FashionSigLIP baseline; a partial reranker is never allowed
to introduce a new candidate.
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

To create a smaller, non-production review artifact for a single item type,
use a separate output path. For example, this keeps the full browser artifact
unchanged while building an OTSH-only preview:

```bash
env HF_HOME=/private/tmp/turtle-hf-cache HF_HUB_OFFLINE=1 \
  .venv/bin/python -m data_pipeline.prepare_real_data \
  --with-vision --item-type OTSH \
  --output frontend/app/generated-data-otsh-preview.json
```

Device configuration:

- `auto` selects CUDA, then Apple MPS, then CPU.
- `cuda` requires a compatible PyTorch/CUDA runtime.
- CPU works but is intended for small catalogues or development.
- Reduce `FASHION_MATCHING_BATCH_SIZE` if model inference exhausts memory.

Model weights are downloaded through the model library cache and are excluded
from Git. Set the normal Hugging Face cache environment variables when a
deployment needs a dedicated persistent cache.

## Manifests

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

Retrieval is visual-only by default (`FASHION_VISUAL_ONLY_RANKING=true`):

| Signal | Weight |
|---|---:|
| Image | 1.00 |
| Structured attributes | 0.00 |
| Description text | 0.00 |

Structured-attribute and text weights exist as configuration knobs
(`FASHION_ATTRIBUTE_WEIGHT`, `FASHION_TEXT_WEIGHT`) but are zeroed by default;
if a signal is missing, its weight is explicitly removed and the remaining
weights are renormalized. The output includes the individual scores and
actual weights used for every result. No fake zero score is inserted.

Within the visual signal itself, appearance evidence is a separate weighted
blend:

| Appearance component | Weight |
|---|---:|
| Neural (FashionSigLIP) | 0.45 |
| Colour palette (CIEDE2000) | 0.45 |
| Texture | 0.10 |

configured via `FASHION_APPEARANCE_NEURAL_WEIGHT`,
`FASHION_APPEARANCE_COLOUR_WEIGHT` and `FASHION_APPEARANCE_TEXTURE_WEIGHT`.

The provisional minimum final score is `0.50`. Results below it return
`no_suitable_match=true` with no product matches. In the browser artifact, a
candidate must also reach Medium match confidence and a calibrated visual score
of at least `0.50`; otherwise no historical product is shown, no sales or
order quantity is generated, and planner review is required.

Cosine image/text scores are mapped from `[-1, 1]` to `[0, 1]`. Structured
attribute similarity is the fraction of populated shared fields that match
after canonicalization. For production, replace provisional weight selection
and generic cosine mapping only with calibration fitted on a held-out,
time-separated relevance set.

## Evaluation

Create reviewed labels as a CSV with columns `query_image_id`,
`relevant_product_id`, `relevance` and `no_match`. Relevance can be binary or
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
