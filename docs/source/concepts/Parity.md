# Parity

## What is it

{term}`Parity` is a metric used to assess **statistical independence** between
metadata factors and class labels in a dataset. Specifically, this function
calculates **statistical** or **demographic parity** under the assumption that
all metadata factors are equally distributed within the data.

By measuring the association between a specific factor (e.g., location, time of
day, demographic group) and the target labels, Parity helps users identify
potential sources of {term}`bias<Bias>` or spurious correlations *before* a
model is trained.

> **Important Note:** Parity measures **data distribution**, not **fairness**.
> While high parity is a component of many fairness definitions, achieving
statistical parity does not guarantee a model is "fair." A dataset can be
statistically balanced yet still produce unfair outcomes due to other
historical or systemic factors.

## When to use it

Parity is best used during the **exploratory data analysis (EDA)** phase or
dataset development. It is critical for understanding the underlying
correlational relationships in your training data, especially when:

* **Data collection is sparse:** Data collected in limited operational
conditions may lack target diversity.
* **Investigating "Shortcut Learning":** A model trained on unbalanced data may
learn to use secondary metadata (e.g., background scenery) to predict the class
label rather than the object features itself.

A T\&E engineer or model developer should use Parity *a priori* to design tests
for model generalization or data augmentation strategies that mitigate sampling
imbalances.

**Implementation Note:**
In order to use {term}`parity<Parity>`, the user must supply their metadata in
a DataEval specific format. Because of this requirement, DataEval has a
`Metadata` class that will take in user [metadata](Metadata.md) and format it
into DataEval's format. The `parity` function takes in the `Metadata` class for
its analysis.

## Why use parity over other statistical methods?

{term}`Parity` allows for **data-centric** bias mitigation rather than
**model-centric** mitigation.

Many common fairness metrics—such as error rate balance, equalized odds, or
positive/negative class {term}`balance<Balance>`—rely on model predictions and
probabilities. This means they can only be calculated *after* the model is
trained. Parity allows you to detect potential issues in the pipeline early,
saving time on training models that are destined to overfit to spurious
correlations.

## What can be done with the parity information?

* **If factors are independent:** If metadata factors show high parity
(independence) regarding the labels, the model is less likely to overfit to
spurious correlations.
* **If factors are dependent:** If a metadata factor is strongly correlated
with class labels, the model may exhibit intended {term}`bias<Bias>`.

Recommended actions for low parity include:

1. **Data Augmentation:** Collecting or generating additional training data to
ensure consistent label distributions across all values of the metadata factor.
2. **Latent Space Adjustment:** Identifying how the correlation manifests in
the model's {term}`embeddings<Embeddings>` and subtracting the bias in the
{term}`latent space<Latent Space>`.
3. **Loss Weighting:** Assigning weights to the loss function to de-emphasize
samples that exhibit strong spurious correlations.

## How it works

The parity function calculates statistical parity assuming an
**equal distribution** of metadata factors.

Internally, the function first constructs a **contingency matrix** (crosstab)
to tally the frequency of each metadata factor against class labels.

It then uses the **G-test (Log-Likelihood Ratio)** to calculate association.
While similar to the Chi-Squared test, the G-test is often more robust for
analyzing categorical data.

Finally, it converts this statistic into **Bias-Corrected Cramér's V**. The
bias correction (based on Bergsma, 2013) provides a more accurate estimate of
association strength than standard Cramér's V, which often overestimates
relationships in finite samples or large contingency tables.

**Interpretation guidelines:**

Cramér's V ranges from 0 to 1:

* **0:** No association (Perfect Parity/Independence)
* **1:** Perfect association (Complete Dependence)

| Bias-Corrected Cramér's V Score | Interpretation |
| :--- | :--- |
| **0.0 - 0.1** | Negligible association |
| **0.1 - 0.3** | Weak association |
| **0.3 - 0.5** | Moderate association |
| **\> 0.5** | Strong association |

Scores closer to 1 (with low p-values) suggest a strong correlation between a
metadata factor and class labels, indicating a risk that the model will use
this factor as a "shortcut" for prediction.

## Data Sufficiency and Warnings

Statistical tests for parity rely on having enough samples to make valid
inferences. The `parity` function automatically detects **insufficient data**,
defined as specific category-class combinations with a frequency count of
fewer than 5.

If the function detects insufficient data, it will:

1. Still calculate the score (removing rows that are entirely zero).
2. Return a dictionary detailing exactly which factors and classes lack
sufficient representation.

**Note:** If you receive high p-values or warnings about insufficient data, the
resulting Parity score may not be statistically significant, and you should
collect more data before making decisions based on that factor.

## See Also

* [Fairness In Machine Learning: A Survey](https://arxiv.org/abs/2010.04053)
* [G-test (Log-Likelihood Ratio)](https://en.wikipedia.org/wiki/G-test)
* [Bias Correction in Cramér's V (Bergsma, 2013)](https://www.google.com/search?q=https://www.researchgate.net/publication/261982746_A_bias-correction_for_Cramer%27s_V_and_Tschuprow%27s_T)
