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

[kernel trick]: https://en.wikipedia.org/wiki/Kernel_method
[radial basis function]:
  https://en.wikipedia.org/wiki/Radial_basis_function_kernel
