# _Mann-Whitney U (MWU)_

The {term}`Mann-Whitney U test<Mann-Whitney U Test>` (also known as the
Wilcoxon rank-sum test) is a non-parametric test that compares whether two
samples come from the same distribution by analyzing their ranks.

$$
U = n_1 n_2 + \frac{n_1(n_1+1)}{2} - R_1
$$

where $n_1$ and $n_2$ are the sample sizes, and $R_1$ is the sum of ranks for
the first sample.

**Key characteristics:**

- **Rank-based**: Uses ranks instead of raw values, making it robust to outliers
- **Sensitivity**: Most sensitive to shifts in the median (location shifts)
- **Robustness**: Highly robust to extreme values and heavy-tailed distributions
- **Assumptions**: Only requires ordinal data (rankable values)

**When to use:**

- **Image quality metrics with noise** (blur scores, sharpness with outliers)
- **Outdoor vision systems** (weather-induced variance, illumination extremes)
- When data contains outliers or extreme values
- For detecting median shifts rather than mean shifts
- With heavy-tailed or skewed distributions
- In scenarios where robustness is more important than power

**Limitations:**

High risk of false negatives. It completely ignores changes in variance or shape.
If your data's spread doubles but the median stays the same, MWU will report "no
drift," which could be catastrophic for your model.
