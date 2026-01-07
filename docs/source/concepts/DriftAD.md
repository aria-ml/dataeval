# _Anderson-Darling (AD)_

The {term}`Anderson-Darling test<Anderson-Darling Test>` is a modification of
the Cramér-von Mises test that gives more weight to the tails of the
distribution:

$$
A^2 = -n - \sum_{i=1}^{n} \frac{2i-1}{n} \left[ \ln F(z_i) + \ln(1 - F(z_{n+1-i})) \right]
$$

**Key characteristics:**

- **Tail sensitivity**: Emphasizes differences in the tails through weighted
  integration
- **Distribution testing**: Particularly effective for testing specific
  distributional forms
- **Power**: More powerful than KS for tail-heavy distributions
- **Weighting**: Uses $\frac{1}{F(1-F)}$ weighting that emphasizes extremes

**When to use:**

- **Rare object class distributions** (imbalanced detection scenarios)
- **Extreme lighting/weather conditions** (nighttime, fog, snow)
- When drift is expected in the tails of distributions
- For heavy-tailed distributions (rare events in video streams)
- Detecting outlier shifts or changes in extreme values
- When tail behavior is critical for model performance

**Limitations:**

Can be "hyper-sensitive." It may trigger alerts for tail-end noise that doesn't actually
impact the model's predictive performance, leading to high alert volume.
