# Fashion catalogue matching

The visual matcher retrieves last season's visually similar products for every
style in an uploaded upcoming catalogue. It runs inside the analysis service, on
the images the user uploads through the **New analysis** workspace. Product IDs,
sales and operational outcomes never enter an embedding.

## Architecture

The visual path is two-stage:

```text
Uploaded product image
  -> validation + EXIF/RGB/aspect-ratio preprocessing
  -> source item type forms the strict eligibility cohort
  -> FashionSigLIP retrieves the top 50 eligible candidates through FAISS
  -> four dominant garment colours are extracted from the central foreground region
  -> CIEDE2000 perceptual distance rejects visually different palettes
  -> multi-scale DINOv2 verifies visible pattern and construction
  -> a pattern gate excludes mismatched checks, stripes, prints and structured fabrics
  -> texture evidence captures surface and print detail
  -> the visual reranker combines neural, colour-palette and texture evidence
```

The default provisional encoder is `Marqo/marqo-fashionSigLIP`. The encoder is
configuration-driven so FashionCLIP can remain a benchmark and newer compatible
Hugging Face or OpenCLIP fashion encoders can be evaluated. Do not activate a
model merely because it wins a public benchmark; select it with labelled,
catalogue-specific evaluation.

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
deterministic and correct. If DINOv2 is disabled, unavailable, or its configured
weight is `0`, the planner retains the FashionSigLIP baseline; a partial
reranker is never allowed to introduce a new candidate.

Set `FASHION_DINO_MODEL_REVISION` to an exact Hugging Face commit before a
production run, just as for the primary encoder.

## Feature reuse across builds

A **Replace historical + upcoming** run encodes the uploaded historical images
once and writes the embeddings beside that historical version as `features.npz`.
A later **Reuse trained historical** run loads that index instead of re-encoding,
so only the new upcoming images go through the encoder. The saved index records
the encoder ID, resolved revision, reranker ID and revision, and preprocessing
version; a reuse run that does not match them is rejected rather than silently
mixing incompatible vectors.

Per-image features are additionally cached under
`{TURTLE_ANALYSIS_ROOT}/feature-cache`, keyed by image checksum and model
identity. After a successful activation the cache is pruned to the images the
active build still references.

## Configuration

Python 3.12 is required. All dependencies are in `requirements.txt` at the
repository root (see the root [README.md](README.md)). Copy the environment
template into the existing virtual environment:

```bash
cp .env.example .env
```

```bash
set -a && source .env && set +a
```

Before production use, replace `FASHION_MATCHING_MODEL_REVISION=main` with an
exact Hugging Face commit hash. Models using remote model code must be
security-reviewed and pinned before deployment.

Device configuration:

- `auto` selects CUDA, then Apple MPS, then CPU.
- `cuda` requires a compatible PyTorch/CUDA runtime.
- CPU works but is intended for small catalogues or development.
- Reduce `FASHION_MATCHING_BATCH_SIZE` if model inference exhausts memory.

Model weights are downloaded through the model library cache and are excluded
from Git. Set the normal Hugging Face cache environment variables when a
deployment needs a dedicated persistent cache.

Set `TURTLE_RUN_VISION=false` to validate uploads and exercise the run
lifecycle without loading encoders. Builds produced that way carry no visual
evidence and are for development only.

## Scores

Retrieval is visual-only (`FASHION_VISUAL_ONLY_RANKING=true`): workbook
attributes and description text are not mixed into the score. Within the visual
signal, appearance evidence is a weighted blend:

| Appearance component | Weight |
|---|---:|
| Neural (FashionSigLIP) | 0.45 |
| Colour palette (CIEDE2000) | 0.45 |
| Texture | 0.10 |

configured via `FASHION_APPEARANCE_NEURAL_WEIGHT`,
`FASHION_APPEARANCE_COLOUR_WEIGHT` and `FASHION_APPEARANCE_TEXTURE_WEIGHT`.
They must sum to `1`.

The provisional minimum visual score is `0.50`. A candidate must also reach
Medium match confidence; otherwise no historical product is shown, no sales or
order quantity is generated, and planner review is required.

Cosine image scores are mapped from `[-1, 1]` to `[0, 1]`. For production,
replace provisional weight selection and generic cosine mapping only with
calibration fitted on a held-out, time-separated relevance set.

## Tests

Tests use a deterministic lightweight encoder and do not download model weights:

```bash
make backend-test
```

Real-model smoke tests should run separately in a GPU/integration environment
with pinned model revisions.

## Known limitations and next steps

- The default weights and the `0.50` threshold are provisional, and must be
  calibrated on genuine positive and no-match examples from a time-separated,
  human-reviewed set.
- Compare at least `patrickjohncyh/fashion-clip` (image-only baseline),
  `Marqo/marqo-fashionSigLIP`, and a reviewed compatible challenger before
  changing the configured encoder.
- Catalogue encoding is batched; high-QPS serving would need a bounded dynamic
  batcher, which this service does not provide.
- The planner workspace intentionally displays only upcoming products and
  historical analogues that have mapped images.
