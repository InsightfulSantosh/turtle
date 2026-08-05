# Turtle Season Intelligence — v5.1 Client Demo Guide

This guide describes the current 200-product SS27 validation preview. The full
5,550-product artifact has not been rebuilt yet.

## Current decision policy

| Item | v5 policy |
|---|---|
| Matching evidence | 100% product-image evidence |
| Required catalogue constraint | Same item type |
| Workbook colour/design/fabric fields | Display and data audit only |
| Minimum accepted visual score | 50% |
| Historical products shown | Top three gated visual candidates, when available |
| Historical products used | One: the candidate selected by the user |
| Expected customer sales | Cleaned sales of that one historical product |
| Recommended initial order | Selected product sales divided by target sell-through |
| Pack rounding | 25 units |
| Safety cap | 2,000 units |
| Target sell-through | Tunable from 50% to 90%; 70% default |
| Similarity criterion | Tunable from 10% to 90%; 50% default |
| Selected match below criterion | Zero system sales, zero system order, manual review |

There is no attribute similarity score, regression forecast, multi-product
average, or sell-through conversion in v5.

## How matching works

```text
Upcoming product image
  -> restrict candidates to the same item type
  -> FashionSigLIP visual retrieval
  -> DINOv2 visual detail reranking
  -> dominant-palette Lab/CIEDE2000 colour comparison
  -> visual pattern gate
  -> texture evidence
  -> highest visual score
  -> accept at 50% or send to manual review
```

Colour is measured from the image. The workbook colour name does not influence
the score or create a colour family. This avoids incorrect source labels forcing
a visually wrong match.

For OTTR, visual analysis uses a waist-to-lower-leg crop. This reduces the
effect of shirts, faces, logos, and footwear. OTTR applies the hard pattern gate
to Checks, Prints, Stripes, and Dobby/Structure; plain trousers still use DINO
visual evidence without a pattern-only rejection.

## How the two quantities are produced

The user reviews up to three visual candidates and selects one. Once that
product passes the visual threshold:

- **Expected customer sales** copies that product's cleaned historical sales.
- **Recommended initial order** divides those sales by the selected target
  sell-through.
- Both values are rounded to the nearest 25 and capped at 2,000 units.

Only the clicked analogue contributes to the calculation. The figures are not
statistical forecasts and no separate machine-learning sales model runs.

If no product passes 50%, the application deliberately shows no historical
analogue and generates no sales or order quantity. A buyer must review the item
manually.

## Current preview validation

The balanced preview contains 200 upcoming products and all seven SS27 item
types. It reuses the already-computed visual features, so changing the decision
policy did not require image re-encoding.

| Item type | Preview products | Accepted single matches | Manual review |
|---|---:|---:|---:|
| OTGL | 16 | 1 | 15 |
| OTJK | 40 | 17 | 23 |
| OTPO | 9 | 1 | 8 |
| OTSH | 40 | 25 | 15 |
| OTSU | 40 | 7 | 33 |
| OTSW | 16 | 4 | 12 |
| OTTR | 39 | 16 | 23 |
| **Total** | **200** | **71** | **129** |

The 50% threshold intentionally admits borderline examples. Client review of
those examples should determine whether 50% remains suitable before the full
5,550-product rebuild.

## Recommended client explanation

> The system first finds the visually closest historical product within the
> same product type. Visual similarity includes garment shape, construction,
> pattern, texture, and colour measured directly from the images. If the best
> product scores at least 50%, its cleaned sales and original order become the
> two recommended values. If it does not pass, the system returns no quantity
> and requests manual review. Workbook product attributes and a separate sales
> prediction model do not affect this decision.

## Important limitations

- The current artifact is a 200-product validation preview, not the full SS27
  catalogue result.
- A 50% cutoff is permissive; borderline pairs need business review.
- Historical image quality and pose can affect similarity.
- The 2,000-unit cap is a safety rule. Values above it are clipped.
- Data labels remain visible for audit even though they do not influence
  matching.

After client approval of the preview, run the full visual build for all 5,550
upcoming products and repeat the same contract checks before release.
