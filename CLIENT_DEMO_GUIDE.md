# Turtle Season Intelligence — Client Demo and Calculation Guide

This document explains, step by step, how the current local POC compares
products, finds historical analogues, forecasts sales, estimates uncertainty and
recommends an initial order quantity.

It describes the model currently running in the application:

| Item | Current POC |
|---|---:|
| Model version | 2.4.0 |
| Historical products with outcomes | 33 |
| Upcoming products | 167 |
| Historical images available | 33 of 33 |
| Upcoming images available | 164 of 167 |
| Image model | FashionCLIP |
| Image embedding size | 512 dimensions |
| Sales model | scikit-learn Ridge regression |
| Default similarity blend | 20% attributes + 80% FashionCLIP |
| Historical analogues used | Top 3 |
| Final sales blend | 50% analogue + 50% Ridge |
| Default target sell-through | 70% |
| Pack rounding | 25 units |
| POC order limits | 100–2,000 units |

> Client positioning: this is an AI-assisted buying POC with real FashionCLIP
> image features, a trained scikit-learn model and backtested uncertainty. It is
> not yet a production-certified forecasting model because only 33 historical
> sales outcomes were supplied.

## 1. The business question answered by the POC

For every upcoming product, the application answers five questions:

1. Which historical products are most similar?
2. Why are they similar?
3. Based on those analogues, how many units could the upcoming product sell?
4. What does the independent Ridge model forecast?
5. Given expected sales and the company's inventory strategy, how many units
   should be ordered initially?

The key outputs must not be confused:

| Output | Meaning |
|---|---|
| Expected customer sales | The model's point forecast of customer sales units |
| Sales forecast range | The uncertainty interval around expected sales |
| Recommended initial order | Inventory required to support expected sales at the selected target sell-through |
| Planner quantity | Optional manual override entered by the buyer |

The AI forecasts **sales**. The inventory policy converts that sales forecast
into an **order recommendation**.

## 2. Meaning of the historical outcome columns

| Field | Meaning | How it is used |
|---|---|---|
| Order | Quantity originally ordered or committed | Historical evidence and data-quality checks |
| Dispatch | Quantity supplied or dispatched | Historical evidence and data-quality checks |
| Sales | Units sold to customers | Primary model-training target |
| Sell-through | Historical sales performance relative to available inventory | Displayed as analogue performance evidence |

Order, dispatch and sales are different stages of the inventory flow. They
should not be expected to contain the same number.

Example:

| Stage | Units |
|---|---:|
| Ordered | 1,000 |
| Dispatched | 900 |
| Sold | 630 |
| Sell-through | 70% of dispatched stock |

The current model learns from cleaned **sales units**, not from order quantity.
Order and dispatch are not inserted into product similarity because doing that
would leak historical performance into a product-style comparison.

### Data-quality treatment

The supplied sample contains:

- 8 rows where dispatch is above order;
- 1 row where sales are above dispatch; and
- 1 row where sell-through is above 100%.

The model flags these records. For training, valid sales values are retained.
Only the impossible row where sales exceed both observable order and dispatch is
capped at the strongest observable supply value. This prevents one inconsistent
row from dominating a 33-row training sample.

## 3. Complete calculation flow

```text
Upcoming product
      |
      +--> Compare 9 structured attributes ------------------+
      |                                                       |
      +--> Compare image using FashionCLIP -------------------+
                                                              |
                 Combined similarity score for all 33 products
                                      |
                               Rank historical products
                                      |
                              Select top 3 analogues
                                      |
                     Similarity-weighted analogue sales
                                      |
                                      +--------------------+
                                                           |
Upcoming product attributes --> trained Ridge sales model  |
                              |                            |
                              +-------- 50/50 blend <------+
                                           |
                                  Expected customer sales
                                           |
                              Conformal sales forecast range
                                           |
                         Divide by target sell-through policy
                                           |
                          Round to 25 and apply order limits
                                           |
                              Recommended initial order
```

## 4. Step 1 — Attribute similarity

Each upcoming product is compared with every historical product using nine
informative and comparable attributes.

### Attribute weights

These weights operate inside the attribute score and total 100%:

| Attribute | Weight | Comparison method |
|---|---:|---|
| Pattern | 17% | Exact or related pattern-family match |
| Category/item type | 16% | Exact match plus a strong cross-category penalty |
| Collection/fit | 14% | Exact categorical match |
| Fabric | 14% | Exact match or fabric-token overlap |
| MRP/price | 11% | Smooth proportional price-distance calculation |
| Colour | 9% | Exact colour or related colour family |
| Sleeve | 7% | Exact categorical match |
| Provision/fit code | 7% | Exact categorical match |
| Season family | 5% | Normalized AW, SS or CORE family match |

The attribute score is:

```text
Attribute similarity = sum(attribute match x attribute weight)
```

### How individual attribute scores are assigned

#### Exact categorical attributes

Category, sleeve, fit code and collection/fit normally receive:

- `100%` when the normalized values are the same;
- `0%` when they are different.

If category/item type is different, the total attribute score also receives a
strong `0.42` multiplier. This prevents a visually similar T-shirt from becoming
the leading analogue for a shirt.

#### Pattern

- Exact pattern: `100%`
- Same mapped pattern family: `62%`
- Certain broadly related printed/check/stripe groups: `8%`
- Unrelated pattern: `0%`

This is why the Match evidence screen can show values such as 100%, 62%, 8% or
0% rather than only 100% and 0%.

#### Fabric

Fabric uses the larger of:

1. exact normalized match; or
2. Jaccard token overlap.

For example, `100% COTTON STRETCH` and `COTTON STRETCH` share most of their
meaningful tokens, so they can receive a partial score instead of being treated
as completely unrelated.

#### Colour

- Exact colour name: `100%`
- Same mapped colour family: `66%`
- Different colour family: `0%`

For example, `SKY BLUE` and `LIGHT BLUE` can be related even though their source
labels are not identical.

#### Price

Price uses a smooth proportional distance:

```text
Price similarity = exp(-abs(log(upcoming MRP / historical MRP)) / 0.30)
```

The result decreases gradually as the proportional price difference increases.
This is more useful than declaring two prices either exactly the same or
completely different.

#### Season family

Detailed season values are normalized into business families such as:

- AW
- SS
- CORE

Two AW products can therefore match even if one is labelled AW2025 and the
other AW2026.

### Why some workbook columns are excluded

| Excluded field | Reason |
|---|---|
| CAT2 range | Constant after normalizing `CMI + VMI` and `VMI + CMI`; it cannot rank products |
| CAT5 merch type | Every historical candidate is FASHION; it cannot differentiate candidates |
| Style and row identifiers | Codes identify a row but do not describe reusable product meaning |
| Colour variant code | Codes such as 1001 are not stable colour semantics |
| Order, dispatch, sales, sell-through | Outcomes would leak performance into similarity |

Only comparable fields with at least two populated historical values can
contribute to similarity.

## 5. Step 2 — FashionCLIP image similarity

FashionCLIP is a deep-learning model trained to represent fashion images and
fashion language in a shared numerical space.

For each available product image:

1. FashionCLIP processes the garment image.
2. It produces a 512-dimensional embedding.
3. The vector is normalized to unit length.
4. Cosine distance compares the upcoming image with a historical image.
5. The raw distance is calibrated against the distance distribution in the
   supplied catalogue and converted to a 0–100 similarity score.

Conceptually:

```text
Smaller cosine distance = closer image embeddings = higher visual similarity
```

The calibration uses the catalogue median and its 10th/90th percentile spread:

```text
Visual similarity = 1 / (1 + exp((distance - median distance) / calibrated scale))
```

FashionCLIP can capture visual relationships involving garment shape, colour,
pattern, texture, collar, sleeve and overall design. The score is a relative
similarity signal, not a probability that two products are identical.

If an upcoming item has no image, the application uses attribute similarity
only and explicitly marks the result as an attribute-only match.

## 6. Step 3 — Combined similarity and ranking

The currently validated POC blend is:

```text
Combined similarity
= 20% x attribute similarity
+ 80% x FashionCLIP similarity
```

Example:

| Signal | Score | Weight | Contribution |
|---|---:|---:|---:|
| Attribute similarity | 82% | 20% | 16.4% |
| FashionCLIP similarity | 85% | 80% | 68.0% |
| Combined similarity | | | **84.4%**, displayed as approximately 84–85% |

The calculation is performed for all 33 historical products. They are sorted
from the highest to the lowest combined score, and the default model selects the
top 3.

The 20%/80% and top-3 settings were not manually invented for the demo. They
were selected from the supplied sample using leave-one-out validation and a
parameter search. Because the sample contains only 33 outcomes, these remain
pilot settings rather than permanent production parameters.

## 7. Step 4 — Analogue sales forecast

The analogue forecast asks:

> How many units were sold by the historical products most similar to this
> upcoming product?

The top-three combined similarity scores are squared so that the closest match
has more influence:

```text
Analogue sales forecast
= sum(historical sales x combined similarity^2)
  / sum(combined similarity^2)
```

Illustrative example:

| Historical analogue | Similarity | Cleaned sales | Similarity squared | Weighted sales |
|---|---:|---:|---:|---:|
| A | 90% | 500 | 0.81 | 405.0 |
| B | 80% | 350 | 0.64 | 224.0 |
| C | 60% | 200 | 0.36 | 72.0 |

```text
Analogue forecast = (405 + 224 + 72) / (0.81 + 0.64 + 0.36)
                  = 387 units
```

The displayed component is rounded to the 25-unit planning increment, so this
example would display approximately 375 units.

## 8. Step 5 — Ridge sales forecast

Ridge is active in the current POC. It does **not** calculate similarity. It
produces a second, independent sales forecast.

### Ridge input features

The pipeline derives features from:

- category/item type;
- sleeve;
- provision/fit code;
- mapped pattern family;
- collection/fit;
- mapped colour family;
- AW/SS/CORE season family;
- fabric tokens; and
- logarithm of MRP.

The scikit-learn pipeline is:

```text
DictVectorizer -> StandardScaler -> Ridge regression
```

#### DictVectorizer

Converts product labels into numerical machine-learning features. For example,
`pattern=CHECKS` and `colour=BLUE` become model-readable columns.

#### StandardScaler

Scales the feature values so that the Ridge penalty behaves consistently across
features.

#### Ridge regression

Learns the relationship between product characteristics and historical sales.
Ridge adds an L2 regularization penalty to reduce unstable coefficients and
overfitting. The selected `alpha` is `10.0`.

The Ridge question is:

> Across the full historical sample, what sales level is associated with a
> product having these characteristics?

This complements the analogue question, which focuses specifically on the most
similar historical items.

## 9. Step 6 — Final expected customer sales

The current trained model gives equal weight to the two forecasts:

```text
Expected sales
= 50% x analogue sales forecast
+ 50% x Ridge sales forecast
```

Current POC example:

| Component | Forecast |
|---|---:|
| Analogue sales forecast | 375 units |
| Ridge sales forecast | 475 units |

```text
Expected sales = (375 x 50%) + (475 x 50%)
               = 425 units
```

Expected sales are rounded to the 25-unit planning increment.

## 10. Step 7 — Sales forecast range

A single point forecast cannot describe all uncertainty. The POC therefore uses
a finite-sample conformal interval derived from leave-one-out forecast errors.

### How the base interval is learned

1. Hold out one historical product.
2. Train using the other 32 products.
3. Forecast sales for the held-out product.
4. Record the absolute error.
5. Repeat until every historical product has been held out once.
6. Select the finite-sample quantile required for an 80% conformal interval.

The resulting base half-width is currently **250 sales units**.

The range can be widened for a weak top match:

```text
Adjusted half-width
= 250 x (1 + max(0, 0.70 - top-match similarity))
```

For a top-match score above 70%, there is no extra similarity penalty.

Example:

```text
Expected sales = 425
Half-width     = 250
Sales range    = 425 - 250 to 425 + 250
               = 175 to 675 units
```

The POC describes this as an **80% forecast range**. It is a data-calibrated
uncertainty interval, not a guarantee.

## 11. Step 8 — Recommended initial order

The initial order is derived from the expected-sales forecast:

```text
Raw initial order = expected sales / target sell-through
```

It is then:

1. rounded to the nearest 25-unit pack; and
2. restricted to the POC order limits of 100–2,000 units.

Example with 425 expected sales and 70% target sell-through:

```text
Raw order = 425 / 0.70
          = 607.14

Pack-rounded recommended initial order = 600 units
```

The business interpretation is:

```text
600 ordered x 70% expected sell-through = approximately 420 sales units
```

This is close to the AI forecast of 425 units.

### Effect of the inventory strategy

Expected sales remain fixed at 425 in this example:

| Target sell-through | Approximate pack-rounded order | Interpretation |
|---:|---:|---|
| 60% | 700 | More availability buffer and higher residual-stock risk |
| 70% | 600 | Current default balance |
| 80% | 525 | Leaner inventory and higher stock-out risk |
| 90% | 475 | Very lean initial commitment |

Changing target sell-through changes the recommended order. It does not retrain
the model and does not change expected customer sales.

### Recommended-order range

The lower and upper sales forecasts are separately divided by the selected
target sell-through and rounded to packs.

For the current example:

```text
Sales range       = 175 to 675
Target            = 70%
Order range       = 175/0.70 to 675/0.70
Pack-rounded      = 250 to 975 units
```

## 12. Match confidence

Match confidence answers:

> How strong and reliable is the historical analogue evidence?

It is deliberately separate from sales uncertainty.

### High match confidence

All of the following must be true:

- top combined similarity is at least 84%;
- average similarity of the top three is at least 72%;
- top match has visual evidence; and
- top-three analogues have no flagged outcome-quality issues.

### Medium match confidence

- top similarity is at least 62%; and
- average top-three similarity is at least 52%.

### Low match confidence

The medium thresholds are not met.

Current upcoming catalogue:

| Match confidence | Products |
|---|---:|
| High | 33 |
| Medium | 114 |
| Low | 20 |

## 13. Sales uncertainty label

Sales uncertainty answers:

> How wide is the forecast interval compared with expected sales?

```text
Uncertainty ratio = sales interval half-width / expected sales
```

| Label | Rule |
|---|---|
| Narrow | Ratio is 20% or less |
| Moderate | Ratio is above 20% and no more than 40% |
| Wide | Ratio is above 40% |

Example:

```text
Half-width       = 250
Expected sales   = 425
Ratio            = 250 / 425 = 58.8%
Label            = Wide sales uncertainty
```

A product can have **High match confidence** and **Wide sales uncertainty** at
the same time. This means the product match is convincing, but the limited
historical sales sample does not support a narrow forecast range.

All 167 current upcoming items have a wide range. This is an honest reflection
of the 33-row pilot sample, not a UI error.

## 14. How the model settings were selected

The POC uses scikit-learn `ParameterGrid` and `LeaveOneOut` validation.

The search tested:

| Parameter | Values tested | Selected value |
|---|---|---:|
| Attribute share of similarity | 10%, 20%, …, 90% | 20% |
| Visual share | Complement of attribute share | 80% |
| Number of analogues | 3, 5, 8 | 3 |
| Ridge alpha | 0.1, 1, 10, 100 | 10 |
| Ridge share of final forecast | 15%, 25%, 35%, 50% | 50% |
| Analogue share | Complement of Ridge share | 50% |

For each candidate configuration, the system predicted every historical
product without training on that same product and calculated WAPE. The
configuration with the lowest leave-one-out WAPE was selected.

This is genuine data-driven model selection, but 33 rows are too few for a
permanent production decision. Production selection should use forward temporal
validation across at least three clean seasons.

## 15. Validation metrics

### WAPE — 44.57%

Weighted Absolute Percentage Error measures total absolute forecast error
relative to total actual sales:

```text
WAPE = sum(abs(actual sales - forecast sales))
       / sum(actual sales)
```

A WAPE of 44.57% means the total absolute leave-one-out forecast error was about
44.6 units for every 100 units of actual historical sales.

Lower is better; zero is perfect. The current value is a pilot baseline, not a
production accuracy claim.

### MAE — 127 units

Mean Absolute Error is the average absolute difference between forecast and
actual sales:

```text
MAE = average(abs(actual sales - forecast sales))
```

The current leave-one-out MAE is 127 sales units per historical product.

### Bias — +3.82%

Bias measures whether predictions are systematically high or low in aggregate.

- Positive bias means overall over-forecasting.
- Negative bias means overall under-forecasting.

The current +3.82% indicates slight aggregate over-forecasting, although WAPE
shows that item-level errors remain much larger.

### Interval coverage — 87.88%

Coverage measures how often the actual held-out sales value fell inside the
forecast range.

Approximately 29 of the 33 leave-one-out outcomes were covered:

```text
29 / 33 = 87.88%
```

The interval targets approximately 80% coverage. Actual sample coverage is
higher because the finite-sample interval is conservative. High coverage alone
does not mean high precision: a very wide range can cover many outcomes. Coverage
and interval width must therefore be discussed together.

## 16. What is AI, what is statistical, and what is a business rule?

| Component | Type | What it contributes |
|---|---|---|
| FashionCLIP embeddings | Deep-learning AI | Learns visual fashion representations from images |
| Ridge regression | Trained machine learning | Learns product-feature relationships with unit sales |
| Validation-based model selection | Statistical machine learning | Selects weights, top K, Ridge alpha and ensemble blend |
| Analogue retrieval | Similarity algorithm | Uses the nearest historical products as evidence |
| Conformal interval | Statistical calibration | Converts out-of-fold errors into a sales range |
| Attribute matching rules | Explainable domain logic | Makes structured matches inspectable by buyers |
| Target sell-through | Business policy | Converts sales demand into inventory commitment |
| Pack size and min/max | Operational rule | Produces feasible buying quantities |
| Planner override | Human decision | Allows commercial judgement to supersede the model |

The application is not using ChatGPT to generate quantities. The local POC uses
precomputed FashionCLIP features, scikit-learn Ridge regression and deterministic
inventory constraints. The same inputs and settings produce the same output.

## 17. Decision settings in the UI

### Attribute weight and visual weight

These controls change the relative contribution of structured attributes and
FashionCLIP to similarity.

Changing them can alter:

- historical ranking;
- selected top analogues;
- analogue sales forecast; and
- final expected sales.

They do not retrain FashionCLIP or Ridge during the browser session. The UI marks
the result as a custom scenario when it differs from the validated default.

### Products used in recommendation

The user can compare top 3, 5 or 8 analogues. Every selected analogue is shown
and contributes to the analogue weighted average.

The trained default is top 3.

### Inventory strategy

This is the target sell-through used only to convert expected sales into the
recommended initial order.

It does not alter similarity, analogue ranking, Ridge output or expected sales.

### Planner quantity

The buyer can enter a manual order. This is an override for decision workflow;
it does not retrain the model in the current POC.

In production, approved recommendations and overrides should be stored as
feedback for monitoring and future retraining.

## 18. Complete real POC example for the demo

Use product `OTSH-98427-1001` when demonstrating the current local artifact.

| Stage | Current displayed result |
|---|---:|
| Attribute similarity with top analogue | 82% |
| FashionCLIP similarity with top analogue | 85% |
| Combined top-match score | Approximately 85% |
| Analogue sales forecast | 375 units |
| Ridge sales forecast | 475 units |
| Expected customer sales | 425 units |
| 80% sales range | 175–675 units |
| Target sell-through | 70% |
| Recommended initial order | 600 units |
| Recommended-order range | 250–975 units |
| Match confidence | High |
| Sales uncertainty | Wide |

Narrate it in this order:

1. “The system compares this upcoming shirt with every historical product.”
2. “The top analogue has 82% structured similarity and 85% FashionCLIP visual
   similarity.”
3. “The validation-selected combination produces an approximately 85% overall
   match.”
4. “The top three similar products produce a 375-unit analogue sales forecast.”
5. “The independent Ridge model predicts 475 units from product attributes.”
6. “The 50/50 ensemble produces 425 expected customer sales.”
7. “Historical backtest errors create a sales range of 175–675 units.”
8. “At the 70% inventory strategy, 425 divided by 70% gives about 607 units.”
9. “After 25-unit pack rounding, the recommended initial order is 600 units.”
10. “Match confidence is high, but sales uncertainty is wide because only 33
    historical outcomes are available.”

## 19. Suggested client-demo flow

### Part A — Establish the business value

Say:

> “The tool helps planners find the most relevant historical evidence for a new
> style, forecast expected customer sales, and translate that forecast into an
> initial inventory decision.”

### Part B — Select an upcoming product

Show:

- product image;
- colour, price, pattern and fabric;
- expandable nine product attributes; and
- expected sales and recommended initial order as separate outputs.

### Part C — Explain the analogue cards

Show:

- ranked historical images;
- combined, attribute and FashionCLIP scores;
- attribute-by-attribute evidence;
- historical order and sell-through; and
- which products contributed to the forecast.

### Part D — Explain the two sales models

Say:

> “The analogue model uses the actual sales of the nearest historical styles.
> Ridge provides an independent machine-learning estimate from the full
> historical feature set. The trained ensemble combines both.”

### Part E — Explain the inventory strategy

Say:

> “Expected sales are the AI forecast. Target sell-through is a commercial
> policy. Changing that policy changes the recommended inventory commitment but
> not the expected-sales forecast.”

### Part F — Explain uncertainty honestly

Say:

> “The range comes from out-of-fold historical errors. Coverage is about 88%,
> but the ranges remain wide because this POC contains only 33 historical sales
> outcomes. More seasons are needed to improve and validate precision.”

### Part G — Show portfolio and export

Show:

- expected sales;
- sales range;
- recommended order and order range;
- confidence and uncertainty labels;
- planner overrides; and
- CSV export.

## 20. Common client questions and recommended answers

### “Is ChatGPT calculating these quantities?”

No. The current local POC uses FashionCLIP deep-image embeddings, scikit-learn
Ridge regression, similarity-weighted historical sales and statistical
uncertainty calibration. ChatGPT is not called when the planner changes an item
or setting.

### “Does Ridge calculate the match score?”

No. Attribute similarity and FashionCLIP calculate the match score. Ridge is a
separate sales-forecast component.

### “Why do you use an analogue model and Ridge together?”

The analogue model captures evidence from the closest historical styles. Ridge
captures broader relationships across all historical products. Blending them
reduces dependence on a single method.

### “Why is high match confidence sometimes accompanied by a wide range?”

They measure different things. Match confidence measures relevance of the
historical products. Sales uncertainty measures the historical forecasting
error relative to expected sales.

### “Why can the user change target sell-through?”

It is a business inventory policy, not a learned customer-sales prediction.
Different risk appetites require different inventory commitments for the same
expected demand.

### “Does changing a slider retrain the model?”

No. It performs scenario analysis with the current artifact. Proper retraining
is an offline, versioned and validated process.

### “Is 44.57% WAPE production ready?”

No. It is an honest POC baseline derived from only 33 outcomes. Production
approval requires multiple complete seasons, forward temporal testing and
category-level accuracy and business-impact analysis.

### “Why are the intervals wide?”

The leave-one-out errors are large relative to the small training sample. The
model reports that uncertainty rather than hiding it behind a precise-looking
single number.

## 21. Current POC limitations

- Only 33 historical sales outcomes are available.
- Leave-one-out validation is used because a credible temporal holdout cannot be
  created from the supplied sample.
- Three upcoming items do not have usable images and use attribute-only matching.
- Stock-outs, opening inventory, replenishment, markdown timing and sales-window
  length are not available.
- Size-level demand, store/channel, region, promotions, margin, supplier lead
  time, MOQ, capacity and budget are not modelled in the fitted POC.
- All 167 current sales ranges are classified as wide.
- The selected 20/80 similarity blend and top-three configuration need validation
  on a much larger multi-season dataset.

## 22. Production approach for 200,000–500,000 items

For a large multi-season catalogue, the all-pairs POC approach should be
replaced by:

1. batch FashionCLIP embedding generation;
2. PostgreSQL/pgvector HNSW approximate nearest-neighbour retrieval;
3. metadata filters such as category, gender, brand and price band;
4. CatBoost learning-to-rank trained from planner relevance feedback;
5. LightGBM P10/P50/P90 temporal demand forecasting;
6. hierarchical reconciliation across category/channel/region;
7. inventory optimization using pack, MOQ, capacity and budget constraints;
8. temporal backtesting, drift monitoring, model registry and rollback; and
9. persistent planner approvals and overrides for learning and audit.

The repository contains the software path for these components, but the current
33-row sample cannot honestly train and approve the production CatBoost,
LightGBM or fine-tuned FashionCLIP artifacts.

## 23. Minimum data requested for production validation

- At least three complete seasons; five is preferable.
- Consistent selling-window dates.
- Opening stock, receipts, transfers, returns and stock-out days.
- Original price, markdown events and promotion exposure.
- Channel, store/region and online/offline context.
- Size-level sales and availability where relevant.
- Product hierarchy, brand, gender and occasion.
- Supplier lead time, pack size, MOQ and capacity.
- Planner-approved relevant and irrelevant analogue pairs.
- Approved orders, overrides and final actual outcomes.

## 24. One-minute executive explanation

> “For every upcoming product, the platform compares nine commercial attributes
> and a 512-dimensional FashionCLIP image representation against historical
> products. It ranks the closest analogues and calculates a similarity-weighted
> sales estimate. In parallel, a trained Ridge model predicts sales from the
> complete product feature set. The validation-selected ensemble combines both
> forecasts. Historical out-of-fold errors produce a transparent sales range.
> Finally, the selected target sell-through converts expected sales into a
> pack-rounded initial order. The current POC proves the workflow, traceability
> and AI integration; additional seasons are required to reach production-grade
> forecast accuracy and narrower uncertainty.”

## 25. Final demo checklist

- Start the application with `npm run dev`.
- Open `http://localhost:3000`.
- Select `OTSH-98427-1001` for the worked example.
- Explain expected sales before recommended order.
- Open at least one historical analogue and show Match evidence.
- Point out attribute and FashionCLIP scores separately.
- Explain the analogue and Ridge forecasts.
- Explain the sales range and the difference between confidence and uncertainty.
- Change inventory strategy and emphasize that expected sales remain unchanged.
- Open Portfolio and show both Expected sales and Recommended buy columns.
- Export CSV if the client wants the operational output.
- State the 33-row limitation and production data requirements clearly.
