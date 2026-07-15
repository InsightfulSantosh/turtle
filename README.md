# Turtle Season Intelligence POC

A client-ready proof of concept for seasonal merchandise planning. It combines
normalized product attributes with garment-region visual features to rank
historical analogues and recommend an explainable order quantity.

## What the demo includes

- 167 upcoming styles and 33 historical styles from the supplied sample files
- configurable attribute and visual-similarity weights
- top historical analogues with component-level match evidence
- sell-through-adjusted order-quantity recommendation and confidence range
- planner overrides, portfolio review, and CSV export
- clear separation between the POC visual engine and the production upgrade

## Run locally

```bash
npm install
npm run dev
```

The generated sample data is stored in `app/generated-data.json`. The source
workbooks remain outside this application directory.

## Refresh sample data

The preprocessing script expects the reviewed workbooks in the parent project
and read-only converted XLSB copies in `../tmp/converted`.

```bash
python scripts/prepare_sample_data.py downloads
curl -L --config ../tmp/vision-images.curl.conf
python scripts/prepare_sample_data.py build
```

The production version should replace the POC visual feature extractor with a
fashion-specific deep embedding model and add full seasonal demand context,
workflow persistence, access controls, and an audit trail.
