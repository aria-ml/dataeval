<!-- markdownlint-disable MD051 -->

# Performance Limits

Before investing in model architecture, hyperparameter tuning, or additional
data collection, a T&E engineer needs to answer a more fundamental question:
_what is this dataset capable of producing?_ Performance Limits analysis
addresses two distinct ceilings that constrain any model trained on a given
dataset — the irreducible error imposed by the data itself, and the saturation
point beyond which additional samples stop producing meaningful improvement.

Understanding these limits has direct program consequences. If the irreducible
error of a dataset exceeds the allowance of an operational accuracy requirement,
no model trained on that data can meet the requirement — the problem must change,
not the model. If the learning curve has already saturated, budget directed at
data collection will yield diminishing returns, and engineering effort is
better spent elsewhere. Both findings are as valuable as a positive result:
they prevent programs from pursuing improvements that are mathematically
impossible given the current data.

```{important}
The Bayes Error Rate (BER) metric is applicable to **classification tasks only**.
BER operates on class embeddings. The Upper-bound Average Precision (UAP) metric
reduces an object detection problem to a classification problem over bounding
box crops. See the UAP section below for details.
```

## What they are

### Bayes Error Rate — the irreducible floor

The {term}`Bayes Error Rate (BER)` is the lowest possible error rate
achievable by _any_ classifier on a given probability distribution. It is not
a property of any particular model — it is a property of the data. No matter
how large the model, how long it trains, or how well it is tuned, it cannot
achieve error below the BER on that distribution.

BER arises from **class overlap in the feature space**. When two classes share
regions of the input space — when the same image features are legitimately
associated with more than one class — no deterministic classifier can resolve
the ambiguity. The probability of misclassification in those regions is
irreducible.

In computer vision, class overlap has several common sources. **Label
ambiguity** occurs when human annotators genuinely disagree about the correct
class for an image. **Sensor noise and resolution limits** cause images that
are semantically distinct to become perceptually identical at the pixel level.
**Incomplete feature sets** occur when the data simply does not contain
enough information to distinguish the classes — trying to predict vehicle type
from low-resolution thumbnails where distinguishing features are not
resolvable. In all cases, the result is the same: overlapping class
distributions in embedding space, and a non-zero floor on error.

In operational contexts, BER is a pre-development diagnostic. If BER estimation on
the current dataset returns an upper bound that exceeds the program's
operational accuracy requirement, the program needs higher-resolution sensors,
revised labeling procedures, additional discriminating features, or a revised
requirement — before any further model development is warranted.

### Upper-bound Average Precision — feasibility for object detection

{term}`Upper-bound Average Precision (UAP)` extends the feasibility concept
to object detection tasks, where the primary metric is mean average precision
(mAP) rather than classification accuracy.

Object detection combines two subtasks: **localization** (where is the
object?) and **classification** (what is it?). UAP isolates the
classification component by treating the ground-truth bounding box crops as a
classification dataset — removing localization error from the equation
entirely. The mAP of the resulting classification problem is an upper bound on
the mAP achievable by any object detector trained on the same data, since a
real detector must also solve localization.

If the UAP on a dataset is below the program's mAP requirement, that
requirement cannot be met regardless of detector architecture. The data —
specifically the class discriminability of the bounding box crops — is the
binding constraint.

UAP is marked experimental and may change in future releases.

### Sufficiency — the diminishing returns curve

While BER and UAP identify the performance ceiling, {class}`.Sufficiency`
characterizes the path to that ceiling. It answers the question: _given the
data we have, how much of our budget should go toward collecting more?_

Sufficiency trains a model on progressively larger subsets of the training
data — at 10%, 25%, 50%, 75%, and 100% of available samples, for example —
evaluates it at each step, and fits a power law curve to the resulting
learning trajectory. That curve can then be used to project performance at
larger dataset sizes and to invert the projection: given a target accuracy,
how many samples are required to reach it?

## Theory

### BER estimation: MST and KNN bounds

The formal definition of BER is the expected misclassification rate of the
optimal classifier:

$$\text{BER} = E_X\!\left[P\!\left(Y \neq \underset{i}{\arg\max}\; P(Y=i \mid X=x)\right)\right]$$

This expectation cannot be computed exactly in practice — it requires knowing
the true conditional class distribution $P(Y \mid X)$, which is what the model
is trying to learn. DataEval instead computes **bounds** on BER using two
graph-theoretic estimators that operate on the {term}`embedding
<Embeddings>` space of the dataset.

Both estimators measure the rate of class label mismatches in the local
neighborhood structure of the embedding space. The intuition is that samples
near the class boundary — where the same image features are associated with
multiple classes — will have neighbors with different labels. The frequency
of these mismatches in the neighborhood graph estimates the confusion in the
feature space.

**MST-based estimation** (`ber_mst`) constructs a minimum spanning tree over
the full set of embeddings. Edges in the MST connect each sample to its
globally nearest neighbor under the constraint that the tree spans all samples.
The number of MST edges crossing class boundaries (after subtracting the
$m - 1$ forced inter-class connections required to span $m$ classes) is used
to estimate the error rate. For a dataset of $n$ samples with $m$ classes, the
upper bound is:

$$\hat{e}_\text{upper} = \frac{2 \cdot \text{mismatches}}{n}$$

The lower bound is derived from the multi-class extension of the
Hellman-Devroye inequality (see Tumer & Ghosh, 1996; Th. 3 and Th. 4 in
Noshad et al., 2019):

$$\hat{e}_\text{lower} = \frac{m-1}{m}\left(1 - \sqrt{\max\!\left(0,\; 1 - \frac{2m}{m-1}\hat{e}_\text{upper}\right)}\right)$$

**KNN-based estimation** (`ber_knn`) constructs a $k$-nearest neighbor graph
instead. For each sample, the modal class among its $k$ nearest neighbors is
computed. The fraction of samples where the modal neighbor class differs from
the true label is the upper bound estimate:

$$\hat{e}_\text{upper} = \frac{\text{misclassified}}{n}$$

The lower bound uses a $k$-dependent correction. For the binary case ($m = 2$),
the correction tightens as $k$ grows:

$$\hat{e}_\text{lower} = \frac{\hat{e}_\text{upper}}{1 + a_k}$$

where $a_k$ follows the Devroye (1981) formula for $k > 5$, simplifies to
$1/\sqrt{k}$ for $k \in \{3, 4, 5\}$, and to $1/2$ for $k = 2$. For
multiclass problems, the same inequality as the MST case applies.

The two estimators are complementary. The MST estimator uses global structure
and is more stable on smaller datasets. The KNN estimator is sensitive to the
choice of $k$ but provides tighter bounds when $k$ is well chosen. The
recommended default is $k = 5$. Adaptive choices such as $k = \sqrt{n}$ or
$k = c_0 n^{4/(4+d)}$ (where $d$ is the embedding dimension) have theoretical
support but are less practical for most users.

Both methods return an `upper_bound` and a `lower_bound`. The upper bound is
the more actionable value for program decision-making: if the upper bound
exceeds an operational requirement, that requirement is at risk regardless of
which bound is tighter. The lower bound establishes that some irreducible
error genuinely exists — it is not all estimation noise.

### UAP estimation

UAP is computed directly as the weighted average precision score over the
bounding-box classification problem:

$$\text{UAP} = \sum_c w_c \cdot \text{AP}_c$$

where $w_c$ is the class weight (proportional to class frequency) and
$\text{AP}_c$ is the area under the precision-recall curve for class $c$.
This is scikit-learn's [`average_precision_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
with `average="weighted"`.

The reduction from object detection to classification is what makes this an
_upper_ bound. A real detector must predict bounding box coordinates as well
as class labels; any localization error reduces its mAP below the UAP. A
program whose UAP is 0.6 cannot achieve a mAP of 0.7 from any detector
trained on that data, because UAP already removes the localization burden.

### Power law curve fitting for sufficiency

The learning curves produced by {class}`.Sufficiency` are modeled with a
three-parameter **inverse power law**:

$$f(n) = c \cdot n^{-m} + c_0$$

where $n$ is the training set size, $c$ is the scale coefficient, $m > 0$ is
the decay exponent, and $c_0$ is the asymptote. DataEval works internally with
the _error_ form of this curve (higher is worse), then converts to the
_performance_ form for output:

$$\text{performance}(n) = 1 - f(n) = 1 - c \cdot n^{-m} - c_0$$

The asymptote $c_0$ is what makes this model useful for program decisions. As
$n \to \infty$, $f(n) \to c_0$, meaning performance converges to $1 - c_0$.
If $c_0 > 0$, performance will never reach 1.0 regardless of dataset size —
reflecting the same irreducible error that BER measures directly, now
estimated from the shape of the learning curve rather than from the embedding
space geometry.

The three parameters are fitted by minimizing the sum of squared residuals:

$$\min_{c, m, c_0} \sum_i \left(f(n_i) - p_i\right)^2$$

where $(n_i, p_i)$ are the observed (sample size, performance) pairs. Because
this loss surface is non-convex and has multiple local minima, DataEval uses
**basin-hopping** (Wales & Doye, 1997) with L-BFGS-B as the local minimizer.
Initial parameters are set by log-linear regression on the observed points,
and the search runs for up to 1,000 iterations by default (early stopping at
200 iterations without improvement).

For unit-interval metrics (accuracy, precision, recall), the asymptote $c_0$
is constrained to $[0, 1]$. For unbounded metrics (loss), no constraint is
applied. The `unit_interval` parameter controls this.

**Forward projection** evaluates $\text{performance}(n)$ at arbitrary $n$
values using the fitted parameters. This answers: _if we had $N$ samples, what
accuracy would we expect?_

**Inverse projection** solves $\text{performance}(n) = t$ for $n$ given a
target $t$:

$$n = \left(\frac{t - (1 - c_0)}{-c}\right)^{-1/m}$$

This answers: _how many samples do we need to reach accuracy $t$?_ When the
target exceeds $1 - c_0$ (the asymptotic ceiling), the inverse projection
returns $-1$, indicating the target is unachievable with more data of the same
type. This is the sufficiency equivalent of the BER bound — the learning
curve has told you that this requirement cannot be met by scaling data alone.

Multiple independent runs (controlled by `runs`) average the learning curves
before fitting, reducing the variance introduced by random subset sampling.
The default schedule uses geometrically spaced evaluation points
(`np.geomspace`) across the full training set size, which provides more
resolution at small $n$ where the curve changes fastest.

## When to use it

**BER** should be run during dataset readiness assessment, before any
significant model development investment. The most important use case is
comparing the BER upper bound against the program's operational accuracy
requirement. If the upper bound is below the requirement, the data is
feasible; if it exceeds the requirement, the constraint is in the data, not
the model.

BER is also useful as a comparative metric across dataset versions. If a new
data collection campaign was intended to reduce class overlap, a lower BER
upper bound after the collection is quantitative evidence that it succeeded.

**UAP** should be used in object detection programs where mAP is the
acceptance criterion, and where you want to verify feasibility before
investing in detector training. Because it requires running a classifier on
bounding box crops rather than a full detector, it is substantially cheaper
to compute than training and evaluating an object detector.

**Sufficiency** is most useful at two points: early in a program, when
deciding how much data to collect before beginning training; and after an
initial training run, when deciding whether more data will help. The inverse
projection (`inv_project`) is the primary tool for the first case — it
translates a target accuracy directly into a required sample count. The
forward projection (`project`) is the primary tool for the second case — it
shows where the learning curve is headed and whether the current trajectory
will reach the operational requirement.

When the inverse projection returns $-1$ for a target's sample count, the interpretation
depends on context. If BER has already been run and returns a high upper
bound, the two findings are consistent — the data has a fundamental
separability problem. If BER is low but sufficiency projects $-1$, the issue
is more likely the model architecture or training procedure rather than the
data itself.

To better understand what to do after assessing performance limits, review the
[Performance Limits section in the Acting on Results explanation page](ActingOnResults.md#performance-limits-findings).

## Limitations

BER and UAP estimation both depend on embedding quality. The estimated class
overlap in embedding space is only as meaningful as the embedding's ability to
represent the semantic distinctions relevant to the classification task. An
embedding model that is poorly suited to the target domain may project
separable classes into overlapping regions, producing an artificially high BER
estimate, or may separate classes that are genuinely confusable at the pixel
level, producing an artificially low estimate. Always pair BER results with a
qualitative check that the embedding structure is meaningful for the task (see
the [Embeddings](Embeddings.md) concept page).

The MST and KNN estimators produce bounds, not point estimates. The gap
between upper and lower bounds can be wide, particularly for small datasets or
high-dimensional embeddings. A wide gap means the true BER could be anywhere
in a large range — the result is informative but not precise. Increasing
sample count narrows the bounds.

BER is a classification-only metric and UAP is a object detection metric that
reduces the problem to a classification metric. They do not assess feasibility
for regression tasks, generative tasks, or object detection beyond the
UAP reduction. For object detection, UAP is a necessary but not sufficient
feasibility check: a program can pass UAP and still face localization
challenges that prevent meeting its mAP requirement.

Sufficiency curve fitting can fail or produce unreliable parameters when the
observed learning curve is non-monotonic (due to randomness in small-subset
training), when too few evaluation points are available, or when the data is
far from power-law behavior. The default of 5 substeps is a minimum; more
substeps and multiple runs produce more reliable curve fits. When fitting
produces an asymptote that seems implausibly high or low, inspect the raw
learning curve data before trusting the projection.

The power law model assumes that performance will continue improving
monotonically with more data of the same type. If the limiting factor is
something other than sample count — label noise, model capacity, distribution
mismatch between train and test — the projection will be optimistic.

## Related concept pages

- [Embeddings](Embeddings.md) — the representation space that BER and UAP
  operate in; embedding quality directly affects the reliability of both
  estimates
- [Data Integrity](DataIntegrity.md) — label errors and sensor corruption
  raise BER by introducing noise into the class boundary; fix integrity
  problems before interpreting BER results
- [Dataset Bias and Coverage](DatasetBias.md) — poor coverage raises
  effective BER on held-out data even when the training distribution appears
  separable; bias causes BER on the operational distribution to exceed BER
  measured on the training set
- [Acting on Results](ActingOnResults.md) — what to do when BER is high or
  sufficiency projection falls short of the operational requirement

## See this in practice

### How-to guides

- [How to measure classification feasibility (BER)](../notebooks/h2_measure_ic_feasibility.py)
- [How to measure data sufficiency](../notebooks/h2_measure_ic_sufficiency.py)
- [How to encode with ONNX](../notebooks/h2_encode_with_onnx.py)

## References

1. Devroye, L. (1981). On the inequality of Cover and Hart in nearest neighbor
   discrimination. _IEEE Transactions on Information Theory_, 27(1), 68–70. [paper](https://ieeexplore.ieee.org/document/4767052)

2. Noshad, M., Xu, L., & Hero, A. O. (2019). Learning to bound the multi-class
   Bayes error. _IEEE Transactions on Signal Processing_, 67(14), 3657–3669. [paper](https://arxiv.org/abs/1811.06419)  
   (Theorems 3 and 4 provide the MST and KNN bound derivations used in DataEval.)

3. Borji, A., & Iranmanesh, S. M. (2019). Empirical upper bound in object
   detection and more. _arXiv preprint arXiv:1911.12451._ [paper](https://arxiv.org/abs/1911.12451)

4. Hestness, J., Narang, S., Ardalani, N., Diamos, G., Jun, H., Kianinejad,
   H., Patwary, M. A., Yang, Y., & Zhou, Y. (2017). Deep learning scaling is
   predictable, empirically. _arXiv preprint arXiv:1712.00409._ [paper](https://arxiv.org/abs/1712.00409)

5. Wales, D. J., & Doye, J. P. K. (1997). Global optimization by basin-hopping
   and the lowest energy structures of Lennard-Jones clusters containing up to
   110 atoms. _Journal of Physical Chemistry A_, 101(28), 5111–5116. [paper](https://arxiv.org/abs/cond-mat/9803344)  
   (Basin-hopping algorithm used for power law curve fitting.)
