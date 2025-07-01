# Dataset prioritization

This page describes dataset prioritization,  including the theory and
application of the tools, how to use them, and why to use them.

## What is dataset prioritization?

Dataset prioritization&mdash;sometimes referred to as pruning&mdash;is the
process of ranking instances in a dataset to focus training on a balanced
representative subset of data, select new data for training that is least
redundant with existing data and, under the right circumstances, to improve
model performance by training on a reduced dataset.  Dataset prioritization is
most important in problems where the dataset is either too large to manage or in
problems with high natural redundancy such as full motion video data or overhead
imagery with abundant and redundant background samples.  DataEval's approach to
prioritization consolidates multiple state-of-the-art approaches for scoring the
difficulty, or conversely the representativeness, of individual instances and
multiple policies for prioritizing the instance scores.

## Why should you prune your dataset?

Prioritizing your dataset can lead to faster development cycles, better
performing models, and better resource efficiency.  Most real-world datasets
intrinsically contain redundancy&mdash;e.g. targets measured on similar
backgrounds or many redundant looks at a single target or class&mdash;and models
trained with this natural redundancy may require more computational resources
and ultimately underperform models trained with a more balanced dataset.
Prioritization mitigates redundancy and balances datasets for model development.

Prioritized subsets of data can benefit model developers and engineers in the
following ways:

1. **Faster Development Cycles**
Training on a prototypical subset of data enables model developers and T&E
engineers to iterate more quickly, saving time and both physical and human
resources.

2. **Better Performing Models**
Operational datasets are not naturally balanced, and when models are trained
with imbalanced data, models tend to underperform, particularly in new domains
or locations.  A prioritized subset of data will retain essential (prototypical)
samples but include samples that are challenging for a model while reducing
amount of data needed to train and iterate during model development.  Sample
selection policies can target an even distribution of classes, while respecting
priority, or balance across sample difficulty.  Pruning a highly-redundant
dataset can in some cases improve model performance on a held-out test set
compared to a model trained on a larger but imbalanced dataset.

3. **Resource Efficiency**
DataEval deprioritizes redundant data samples.  Overly-redundant instances in a
dataset may implicitly emphasize certain target types or presentations, leading
to a biased model.  Furthermore, training on data that provides negligible
additional information content wastes computational and storage resources as
well as developer time.

Prioritization provides a mechanism to understand semantic similarity of new
data compared an existing dataset.  When faced with newly collected unlabeled
data, subject matter experts can use prioritization to identify which
samples will contribute most to the existing dataset prior to labeling, saving
valuable time and human resources needed for manual labeling.

## When should you prune and prioritize your dataset?

Typical times to prioritize and prune datasets are

1. **During initial dataset development**:
Prioritizing data during initial dataset development is used to form a core
set of data for training or Test and Evaluation (T&E).  When it is
appropriate to subsample a dataset, prioritization identifies a
representative subset that is used for rapid iterative model development.
Prioritization is conducted during [Exploratory Data Analysis
(EDA)](../notebooks/tt_clean_dataset.ipynb) stages after initial
[cleaning](DataCleaning.md) and {class}`.Outliers`.

2. **After collecting new data**:
Prioritization tools should be used when incorporating new data into an existing
dataset, using prioritization scores to identify which of the unlabeled data
contribute most to the existing dataset.

3. **When datasets contain known redundancy**:
Operational datasets often contain obvious redundancy and imbalance&mdash;e.g.
oversampled secondary/background classes, undersampled high-value targets, etc.
When this is the case, dataset prioritization is often beneficial and may
improve model performance by mitigating redundancy-induced dataset imbalance.

## Theory behind dataset prioritization

DataEval's approach to prioritization and pruning involves several stages
including dataset embedding (dimensionality reduction), sample scoring and
prioritization according to a selection policy.  Below we discuss each stage and
permutations of options that are available to customize the procedure to new
datasets.

First, however, it is important to contextualize some of the discussion about
dataset pruning in general.  We have found the literature to be occasionally
contradictory, where claims reported by one author cannot be reproduced by
another (or us!).  The point of this disclaimer is to emphasize that
_performance of dataset prioritization strongly depends on the dataset itself_,
and no single algorithm is universally optimal for all datasets.  Pruning with
image datasets is also further complicated by a mismatch between the space in
which we prune and the space over which the resulting model operates.  Training
on a subset of data induces a different manifold from what the data were pruned
on. Or, in other words, we prune with one projection and perform inference with
another.  Below we try to provide some intuition for using the prioritization
tools successfully.

### Embedding image data

When conducting dataset analysis, a meaningful representation of the data is
often critical for generating meaningful results.  For dataset pruning, we are
interested in measuring redundancy, so as long as the embedding approach
captures meaningful information&mdash;e.g. texture information, background
properties, semantic information, etc. as is relevant to the
problem&mdash;pruning algorithms will in turn provide meaningful results.

Choosing a model and task for training the embedding model are important to
consider before pruning.  For reasonably balanced datasets, self-supervised
embeddings are likely to be adequate and do not require labels for training.
However, if the initial dataset is highly imbalanced&mdash;measured by class
frequencies or target size relative to image size&mdash;self-supervised methods
may not be appropriate.  They are likely to encode spatial or frequency biases
into the projection.

Supervised models will better embed datasets than unsupervised models if target
classes of interest are either rare in frequency or small in size.  In this
case, supervised training will guide the model to focus on the salient
properties of the image&mdash;e.g. small-target signatures whose contribution
would otherwise be overwhelmed by abundant background.  The challenge, at least
conceptually, with using a supervised model to embed data is that the problem
becomes circular.  The initial dataset is used to train a model, the dataset is
prioritized using the trained embedder, and then another model (potentially the
same architecture) is trained on a subset of the initial dataset.  It is
perfectly reasonable to iteratively prioritize a dataset in this way given
appropriate resources, and the initial effort of doing so could enable and
benefit future iteration with compact datasets.

Object detection datasets can present embedding challenges, especially if
objects are small relative to chip size or image size.  This is a case of
spatial imbalance, discussed above, where the salient content of the image is
spatially underrepresented.  Using object detection labels to form or weight
embeddings can mitigate spatial imbalance; however, a straightforward
unsupervised embedder is likely to still capture redundancy in background types
or other dominant features.

Finally, it is reasonable to use pre-trained models to embed datasets
depending on the domain of the data and availability of models.  For instance, a
model trained on Imagenet is appropriate for other natural image datasets,
but it might not be appropriate for sonar data.  A key factor when deciding
whether to use a pre-trained model is whether the target dataset is
[out-of-distribution](OOD.md) with respect to the data used to train the model.
It is difficult to predict how a model will embed out-of-distribution data, so
exercise caution when embedding with pre-trained models.

### Sample scoring

Before selecting or ranking instances of a dataset, prioritization algorithms
begin by scoring samples using some measure of prototypicality or task
difficulty.  There are many scoring strategies available, and DataEval focuses
on those that only require access to the dataset and optionally to labels.
Other methods in recent literature require access to models, gradients, or
scores throughout the training pipeline requiring deeper integration with the
model training pipeline than is typically allocated for dataset analysis.  Also
note that none of the core scoring algorithms used in DataEval require labels or
annotations, making them suitable for prioritizing unlabeled data as well.

#### K-nearest neighbor scoring

Our approach to scoring samples in latent space according to k-NN distances is
related to its role in {func}`.coverage`.  The score assigned to each instance
is the k-NN distance for fixed $k$, e.g. $k =50$.  In densely sampled regions of
latent space, k-NN distances will be small, indicating a degree of redundancy.
Samples with large k-NN distances correspond to regions that are less densely
sampled and likely to be less prototypical. Put another way, challenging samples
(with respect to the discrimination task) are more likely to occur in
undercovered regions of the latent space manifold, while relatively 'easy'
samples are likely to occur in densely sampled regions of the manifold.
DataEval applies ranking policies directly to the k-NN distance statistic.

#### K-means scoring

Another approach to sample scoring commonly used in the literature&mdash;e.g.
[[1]](https://proceedings.neurips.cc/paper_files/paper/2022/file/7b75da9b61eda40fa35453ee5d077df6-Paper-Conference.pdf),
[[2]](https://arxiv.org/pdf/2401.04578)&mdash;is to cluster the data and use
inter- and intra-cluster distances to compute scores.  The simplest approach is
to score each sample according to its distance to cluster centroid.  Not unlike
k-NN scoring, prototypical samples will live close to the cluster centroid,
regardless of what labels would be assigned to the sample or cluster, and
challenging samples near the decision boundary are likely to be far from the
cluster centroid.  In DataEval, the `kmeans_distance` scoring approach directly
uses distance to cluster centroid as the prioritization statistic.

The `kmeans_complexity` is based on [[2]](https://arxiv.org/pdf/2401.04578) and
uses the product of average inter-cluster distance and intra-cluster distance of
the nearest 20 clusters.  The product of these two average distances is a
complexity score assigned to each cluster.  Sampling from clusters in proportion
to cluster complexity scores, we sort the dataset.  This approach begins to blur
the lines between sample scoring and re-ranking, but it aims to address
exacerbation of class imbalance without explicitly using class labels.
Optional use of class labels in prioritization policies can more directly
address issues of class imbalance.

### Prioritization policies

With a set of scored samples, nominally sorted by prototypicality, we may use
prioritization policies to produce a balanced and effective dataset for model
development and T&E.  The most straightforward policies, articulated and
discussed in
[[1]](https://proceedings.neurips.cc/paper_files/paper/2022/file/7b75da9b61eda40fa35453ee5d077df6-Paper-Conference.pdf),
are `easy_first` and `hard_first`.  While the names are somewhat reductive,
these polices lead to samples sorted by score in increasing and decreasing
order, respectively.  The student-teacher models in [1] may inform policy
selection, but the core algorithm favored by the authors on real datasets is
k-means clustering with `hard_first` prioritization.  The `easy_first` policy is
generally preferred for very small datasets where a model needs to learn the
essential concepts, while the `hard_first` policy tends to include data needed
to sample the decision boundary for higher-performing models.  For moderate
sized datasets and anything short of very aggressive pruning, we have found that
`hard_first` leads to better performing models than `easy_first`.  However,
with image datasets, we found that these two policies often did not lead to
improved performance over random pruning.

DataEval also includes a `stratified` re-ranking policy, described in
[[3]](https://arxiv.org/pdf/2210.15809). This approach bins the range of
scores and draws samples uniformly from each bin, leading to a mixture of
challenging and prototypical samples.  This approach avoids oversampling dense
regions of score-space&mdash;e.g. many samples with very small scores and high
redundancy.  In our testing, we found that this approach leads to downstream
model performance improvement when compared with random decimation, something we
could not consistently demonstrate with `hard_first` and `easy_first`.

Finally, DataEval provides a `class_balance` policy that samples evenly from
each class and by priority within each class, requiring that the user provide
class labels.  This approach was used in
[[1]](https://proceedings.neurips.cc/paper_files/paper/2022/file/7b75da9b61eda40fa35453ee5d077df6-Paper-Conference.pdf)
after the authors found that pruning exacerbates class imbalance.  Especially
when the initial dataset is extremely unbalanced, this level of direct control
over class balance can be beneficial.

## Default settings

There are many permutations of options available, but we recommend starting with
the `stratified` policy and either `knn` or `kmeans_distance` scoring.  In our
testing, over several datasets, neither consistently outperformed the other, and
they both tended to outperform random decimation.  On a tabular dataset derived
from satellite imagery, `knn` scoring with `hard_first` led to significant
performance improvements over the unpruned dataset; however, on relatively clean
image datasets (CIFAR, Imagenet) stratified re-ranking was necessary to see
improvement over random decimation.

## Related concepts

Dataset pruning is closely related to various forms of data cleaning in DataEval
but aims to shape and balance the distribution of samples in a subset of data.
Low-level [data cleaning](DataCleaning.md) identifies and measures invalid
or unexpected low-level image or label properties, while prioritization operates
in a latent (feature) space, focusing on semantic redundancy.  The
{func}`.clusterer` measures and identifies near duplicates and outlier samples
in the dataset, similarly to prioritization, but is not focused on sampling the
dataset.  {func}`.coverage` is closely related to prioritization and
characterizes regions of the latent space (notionally, concepts) that are
undersampled, while prioritization provides a recommended ranking of samples to
include in a representative subset or to incorporate into an existing dataset.
Prioritization tools aim to provide recommendations for both initial and ongoing
dataset curation.

More broadly, in AI/ML, prioritization is related to

- Coreset selection, dataset pruning
- Curriculum development in active learning
- Loss weighting strategies that emphasize challenging samples, e.g. hard
  negative mining, focal loss, etc.

## References and related work

1. Sorscher, B., Geirhos, R., Shekhar, S., Ganguli, S., & Morcos, A. (2022).
   Beyond neural scaling laws: beating power law scaling via data pruning.
   _Advances in Neural Information Processing Systems_, 35, 19523-19536.

    This paper discusses k-means pruning and compares to  many other pruning
    approaches, some of which require access to the model-training pipeline.  We
    attempted to reproduce the results of Fig. 5c but were unable to outperform
    random pruning on CIFAR with pretrained embeddings.  See Fig. 4 in Ref. [4]
    for corroboration of this assertion about k-means scoring by itself.

    Sorscher et al. also introduce a theoretical framework for understanding the
    optimal tradeoff between selecting prototypical samples and more challenging
    samples.

2. Abbas, A., Rusak, E., Tirumala, K., Brendel, W., Chaudhuri, K., & Morcos, A.
   S. (2024). Effective pruning of web-scale datasets based on complexity of
   concept clusters. arXiv preprint arXiv:2401.04578.

    This paper introduces the idea of cluster complexity (augments k-means
    scoring) to optimize coverage over the dataset and indirectly mitigate class
    imbalance.

3. Zheng, H., Liu, R., Lai, F., & Prakash, A. (2022). Coverage-centric coreset
   selection for high pruning rates. _arXiv preprint arXiv:2210.15809_.

    This paper introduces stratified resampling (as it is called in DataEval).
    Stratified re-ranking led to improvement even on well-balanced datasets,
    e.g. CIFAR10.  However, there is some difficulty in interpreting and
    directly comparing results to other papers, e.g. [1], because Zheng et al.
    pre-prune by a variable amount that is treated as a hyperparameter.  This
    may be a reasonable approach if test performance can be repeatedly
    estimated, but oftentimes it is not.

4. Maharana, A., Yadav, P., & Bansal, M. (2023). D2 pruning: Message passing for
   balancing diversity and difficulty in data pruning. _arXiv preprint
   arXiv:2310.07931_.
