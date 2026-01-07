# _Baumgartner-Weiss-Schindler (BWS)_

The {term}`Baumgartner-Weiss-Schindler test<Baumgartner-Weiss-Schindler Test>`
is a modern non-parametric test that combines advantages of several classical
tests, with particular sensitivity to tail differences:

$$
B = \frac{1}{n_1 n_2} \sum_{i,j} \psi(F(x_i), F(y_j))
$$

where $\psi$ is a weighting function that emphasizes tail regions.

**Key characteristics:**

- **Modern design**: Developed to address limitations of classical tests
- **High power**: Generally higher statistical power across various scenarios
- **Tail sensitivity**: Strong sensitivity to tail differences like Anderson-Darling
- **Versatility**: Performs well across different types of distributional shifts

**When to use:**

- **High-stakes computer vision** (medical imaging, autonomous vehicles)
- **Production vision systems** where missing drift is costly
- When you need high statistical power across diverse drift scenarios
- For detecting both location and scale shifts simultaneously
- When computational cost is not a primary constraint
- As a robust alternative when other tests give ambiguous results

**Limitations:**

Complexity and overhead. It is a more modern, complex statistic that is harder
to find in standard libraries and may require more compute time for high-velocity
data streams.
