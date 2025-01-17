# Divergence

## What is it

HP {term}`divergence<Divergence>` is a nonparametric divergence metric which
gives the distance between two datasets. A divergence of 0 means that the two
datasets are approximately identically distributed. A divergence of 1 means the
two datasets are completely separable. Unlike {term}`drift<Drift>` metrics, HP
divergence is a quantitative measure of how far apart two datasets are, rather
than a formal test of drift.

## When to use it

The `divergence` metric should be used when you would like to know how far two
datasets are diverged for one another. For example, if you have incoming data
and would like to know if the new data is similar to existing data, but are not
yet interested in formally testing {term}`drift<Drift>` on a population level.

## Theory behind it

HP {term}`diverence<Divergence>` is defined by the following:

$$
D_p(f_0,f_1) = \frac{1}{4pq}\int\frac{(pf_0(x)-qf_1(x))^2}{pf_0(x)+qf_1(x)}dx-(p-q)^2
$$

with $0\leq p\leq 1, q = (1-p)$.

There are a few things to note about this quantity. Firstly, it is $0$ if
$f_0=f_1$ and $1$ if the domains of $f_0$ and $f_1$ are disjoint, confirming
the qualitative description from the first section of this page. Furthermore,
in the case where $p=q=0.5$, this quantity simplifies to

$$
D_{0.5}(f_0,f_1) = \frac{1}{2}\int\frac{(f_0(x)-f_1(x))^2}{f_0(x)+f_1(x)}dx
$$

a quite intuitive diverence metric.

There are two methods of estimating this divergence in DataEval. The first is
the MST-based estimator derived in
[Learning to Bound the Multi-class Bayes Error][ber]. The second is a kNN-based
estimator from [Nearest neighbor pattern classification][knn]. Some
[empirical results](https://arxiv.org/abs/2108.13034) suggest that there is not
a decisive preference between the two in terms of performance. kNN does tend to
be less of a computational burden however.

## References

[1] [T. Cover and P. Hart, "Nearest neighbor pattern classification," in IEEE
Transactions on Information Theory, vol. 13, no. 1, pp. 21-27, January 1967,
doi: 10.1109/TIT.1967.1053964.](https://ieeexplore.ieee.org/document/1053964)

[2] [Renggli, C., Rimanic, L., Hollenstein, N., & Zhang, C. (2021). Evaluating
Bayes Error Estimators on Read-World Datasets with FeeBee. arXiv preprint
arXiv:2108.13034.](https://arxiv.org/abs/2108.13034)

[3] [S. Y. Sekeh, B. Oselio and A. O. Hero, "Learning to Bound the Multi-Class
Bayes Error," in IEEE Transactions on Signal Processing, vol. 68, pp.
3793-3807, 2020, doi:
10.1109/TSP.2020.2994807.](https://ieeexplore.ieee.org/document/9093984)

[ber]: https://arxiv.org/abs/1811.06419
[knn]: https://ieeexplore.ieee.org/document/1053964
