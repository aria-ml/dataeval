# _Maximum Mean Discrepancy_

{term}`Maximum Mean Discrepancy (MMD) Drift Detection` is a kernel-based method
for comparing two distributions by calculating the distance between their mean
{term}`embeddings<Embeddings>` in a reproducing kernel Hilbert space (RKHS).
The MMD test statistic is defined as:

$$
\textrm{MMD}(F, p, q) = || \mu_{p} - \mu_{q} ||^2_{F}
$$

where $\mu_{p}$ and $\mu_{q}$ are the mean embeddings of distributions _p_ and
_q_ in the RKHS. The MMD test is particularly useful for detecting complex,
multivariate distributional differences. Unbiased estimates of $\textrm{MMD}^2$ can be
obtained using the [kernel trick], and a permutation test is used to obtain the
{term}`p-value<P-Value>`.

A common choice for the kernel is the [radial basis function] (RBF) kernel,
though other kernels can be used depending on the application.

**Key characteristics:**

- **Kernel trick**: Projects data into high-dimensional feature space using [kernel trick]
- **Multivariate**: Naturally handles multiple features and their dependencies
- **Universal**: With universal kernels (e.g., RBF), can detect any distributional difference
- **Non-parametric**: No assumptions about distribution shapes
- **Interpretability**: Lower than univariate tests; doesn't identify which features drifted

**Common kernels:**

1. **Radial Basis Function (RBF) / Gaussian kernel:**
   $$
   k(x, y) = \exp\left(-\frac{\|x-y\|^2}{2\sigma^2}\right)
   $$
   - Most common choice; universal kernel
   - Bandwidth $\sigma$ controls sensitivity to local vs. global differences

2. **Polynomial kernel:**
   $$
   k(x, y) = (x^T y + c)^d
   $$
   - Captures polynomial interactions up to degree $d$

**Statistical testing:**

A permutation test is used to obtain the {term}`p-value<P-Value>`:

1. Pool reference and test samples
2. Randomly permute and split into two groups multiple times
3. Compute MMD for each permutation
4. P-value = proportion of permutations with MMD ≥ observed MMD

**When to use:**

- **Image/video embeddings** (ResNet, CLIP, ViT, etc.) - primary use case
- High-dimensional data where feature interactions matter
- When drift involves changes in correlations between features
- Deep learning computer vision applications
- Cross-domain shifts (e.g., synthetic → real, indoor → outdoor)
- When univariate tests fail to detect known drift

**Limitations:**

- Computationally expensive for large datasets (quadratic in sample size)
- Kernel selection and hyperparameter tuning required
- Limited interpretability (doesn't indicate which features drifted)
- Requires sufficient samples for reliable permutation testing

[kernel trick]: https://en.wikipedia.org/wiki/Kernel_method
[radial basis function]:
  https://en.wikipedia.org/wiki/Radial_basis_function_kernel
