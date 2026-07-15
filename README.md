# Turtle Season Intelligence AI

A client-ready AI pilot for seasonal merchandise planning. It combines deep
neural image features, normalized product attributes, validation-tuned analogue
retrieval, regularized demand modelling, and conformal uncertainty ranges.

## What the demo includes

- 167 upcoming styles and 33 historical styles from the supplied sample files
- pretrained deep-vision similarity across the supplied product images
- attribute/vision weights and neighbour count selected by validation
- top historical analogues with component-level match evidence
- analogue and regularized-regression demand ensemble
- finite-sample uncertainty range and data-quality guardrails
- planner overrides, portfolio review, and CSV export
- versioned Python API and container definition in `ml-service`

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

The model artifact is stored in `app/generated-data.json`, so the hosted client
demo does not require a live model server. The production API contract,
container, tests, and model documentation are in `ml-service`.

The architecture is production-oriented, but the fitted quantity model is
correctly labelled as a pilot: only 33 historical outcomes were supplied. A
production calibration requires three to five clean seasons, consistent sales
windows, stock-out and replenishment signals, markdowns, channel context, MOQ,
and budget constraints.
