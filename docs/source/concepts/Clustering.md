<!-- markdownlint-disable MD051 -->

# Clustering

Clustering is not an evaluator in DataEval — it is the infrastructure that
several evaluators depend on. Understanding how clustering works, and what its
outputs mean, helps you interpret results from {class}`.Duplicates` (cluster
mode), {class}`.Outliers` (cluster mode), {func}`.label_errors`, and all five
methods of {class}`.Prioritize`.

## What is it

Clustering partitions a set of embeddings into groups (clusters) such that
samples within a group are more similar to each other than to samples in other
groups. In DataEval, clustering always operates on embeddings rather than raw
images: pixel-space distance is a poor proxy for semantic similarity, and
clustering in raw pixel space would identify images with similar colors or
brightness as neighbors regardless of content.

The {func}`.cluster` function is the entry point. It accepts embeddings and
returns a {class}`.ClusterResult` containing cluster assignments, the minimum
spanning tree (MST), linkage tree, membership strengths, and k-nearest neighbor
indices and distances. Two algorithms are available: [HDBSCAN](#ref1) (default) and
[KMeans](#ref2).

## The two algorithms

### HDBSCAN

HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with
Noise) discovers clusters from density structure rather than requiring a
predetermined number of clusters. The algorithm proceeds in stages:

1. **MST construction**: compute the k-nearest neighbor graph (up to $k=25$
   neighbors per point) and extract the minimum spanning tree over the full
   dataset. The MST captures the global connectivity structure of the embedding
   space in a compact form — cutting an edge in the MST corresponds to
   separating the dataset into two connected components.

2. **Condensed tree**: convert the MST into a hierarchical linkage tree via
   single-linkage clustering, then condense it by merging nodes with fewer than
   `min_cluster_size` samples into their parent. The condensed tree represents
   the multi-scale cluster structure of the data.

3. **Cluster extraction**: apply excess-of-mass (EOM) selection to extract the
   most persistent clusters from the condensed tree. Clusters that persist
   across a wide range of density thresholds are preferred over clusters that
   only appear briefly.

4. **Outlier assignment**: HDBSCAN assigns label -1 to samples that do not
   belong to any dense cluster. DataEval then reassigns these outliers to their
   nearest cluster center, so the final `clusters` array contains no -1 values.
   The `membership_strengths` array retains the original HDBSCAN confidence
   scores — a low membership strength on a sample assigned to a cluster means
   HDBSCAN originally considered it borderline.

**`min_cluster_size`** is the key hyperparameter. DataEval sets it adaptively:

- if `n_clusters` is provided, `min_cluster_size = max(5, n_samples // (n_clusters * 3))`
  — smaller clusters are allowed when more clusters are expected;
- if `n_clusters` is _not_ provided, the default is 5% of the dataset, capped at 100 —
  `min_cluster_size = min(100, max(5, (n_samples * 0.05)))`

HDBSCAN is the default because it handles irregular cluster shapes, varying
cluster densities, and datasets where the number of clusters is unknown. Its
results are deterministic given the same input data in the same order.

### KMeans

KMeans partitions the embeddings into exactly `n_clusters` spherical clusters
by iteratively assigning each point to its nearest centroid and recomputing
centroids. DataEval uses k-means++ initialization to reduce sensitivity to the
starting configuration.

When `n_clusters` is not specified, the default is $\lfloor\sqrt{n}\rfloor$
where $n$ is the number of samples. KMeans always assigns every sample to a
cluster — there are no outliers. Membership strength is computed from normalized
distance to the cluster center: points close to their centroid have high
membership strength; points far from any centroid have low membership strength.

KMeans is appropriate when you have strong prior knowledge about the number of
clusters (for example, when it equals the number of classes in a classification
problem) or when cluster shape is expected to be roughly spherical.

**KMeans results are not exactly reproducible** without fixing the global random
seed via `dataeval.config.set_seed()`. The k-means++ initialization draws
random starting centroids; different runs will produce different cluster
assignments unless the seed is fixed. HDBSCAN does not have this property.

## What the ClusterResult contains

| Field                  | Type               | Description                                                  |
| ---------------------- | ------------------ | ------------------------------------------------------------ |
| `clusters`             | `NDArray[int64]`   | Cluster assignment for each sample (0-indexed, no -1 values) |
| `mst`                  | `NDArray[float32]` | MST edge list as (source, target, weight) triples            |
| `linkage_tree`         | `NDArray[float32]` | Hierarchical linkage array from the MST                      |
| `membership_strengths` | `NDArray[float32]` | Per-sample cluster membership confidence [0, 1]              |
| `k_neighbors`          | `NDArray[int64]`   | Indices of up to 25 nearest neighbors per sample             |
| `k_distances`          | `NDArray[float32]` | Distances to those neighbors                                 |

The `membership_strengths` field is the most useful for downstream use. Samples
with low membership strength are at the boundary of their cluster — they are
the candidates that {class}`.Duplicates` cluster mode considers semantically
ambiguous, and they are the samples that {func}`.label_errors` is most likely to
flag as mislabeled if they are near a different class's cluster.

## Relationship to other evaluators

**{class}`.Duplicates` cluster mode** runs HDBSCAN on embeddings and groups
samples that land in the same cluster. Each cluster's mean and standard
deviation are determined and used to tag samples whose nearest neighbor
is less than one deviation from the mean (sample*dist < mean - std).
Unlike hash-based deduplication (which detects visually similar images),
cluster-based deduplication detects \_semantically* redundant images — images
that are different in appearance but represent the same concept in the embedding space.

**{class}`.Outliers` cluster mode** runs clustering and flags samples whose
distance to their nearest cluster center exceeds a threshold derived from that
cluster's distance distribution (mean + n·std, default n=3). This identifies
samples that are semantically isolated — not just statistically unusual in image
statistics, but genuinely unlike the rest of the cluster they were assigned to.

**{func}`.label_errors`** does not call `cluster()` directly, but uses the same
MST infrastructure (`compute_neighbor_distances`) to find intra-class and
extra-class nearest neighbors. The geometry it relies on — whether a sample's
nearest neighbors come from its own class or a different one — is the same
geometric structure that clustering makes visible.

**{class}`.Prioritize`** with `kmeans_distance`, `kmeans_complexity`,
`hdbscan_distance`, or `hdbscan_complexity` methods runs the respective
clustering algorithm internally and uses cluster structure to score samples for
dataset pruning. See [Prioritization](ActingOnResults.md#prioritization-acting-on-redundancy-and-coverage-together)
for details on how cluster distance and complexity scores translate to sample ranking.

## Practical considerations

**Embedding dimension should be below 500.** Clustering in very high-dimensional
spaces suffers from the curse of dimensionality — distances become increasingly
uniform, making density estimation unreliable. If your embedding model produces
high-dimensional representations, apply PCA or another dimensionality reduction
step before clustering.

**HDBSCAN's `min_cluster_size` controls granularity.** A large value produces
fewer, larger clusters; a small value produces many small clusters with more
samples classified as boundary points (low membership strength). For datasets
with many fine-grained categories, setting `n_clusters` to approximately the
number of classes is a reasonable starting hint.

**KMeans requires fixing the seed for reproducibility.** If reproducible
cluster assignments matter for your workflow (for example, if you are using
cluster assignments as input to a downstream analysis), call
`dataeval.config.set_seed(your_seed)` before running KMeans. See the
[Configuring the seed](../notebooks/h2_configure_hardware_settings.py#configuring-the-global-seed)
how-to for an example.

**Clustering is embedding-dependent.** The clusters discovered by either
algorithm reflect the structure of the embedding space, not ground truth
semantic categories. A weak embedding that fails to separate classes will
produce clusters that do not align with class boundaries, making all downstream
uses of cluster structure — outlier detection, duplicate detection, label error
detection — less reliable.

## When to use it directly

Most users will interact with clustering indirectly through {class}`.Duplicates`,
{class}`.Outliers`, {func}`.label_errors`, or {class}`.Prioritize`. Calling
{func}`.cluster` directly is appropriate when:

- You want to visualize the cluster structure of your dataset (using the `mst`
  or `linkage_tree` outputs with a dendrogram or network visualization)
- You want to understand how many natural groupings exist in your embeddings
  before configuring `n_clusters` for KMeans methods in Prioritize
- You want to examine `membership_strengths` to identify boundary samples before
  label review

## Limitations

**HDBSCAN is sensitive to the density structure of the embedding.** Very
uniform density distributions (common in high-dimensional spaces) may cause
HDBSCAN to return a single large cluster. If HDBSCAN produces one cluster for
your entire dataset, consider reducing embedding dimensionality or switching to
KMeans with an explicit `n_clusters`.

**KMeans assumes spherical clusters of roughly equal size.** Datasets with
elongated, nested, or strongly variable-density clusters will be partitioned
suboptimally by KMeans. In these cases, HDBSCAN is more appropriate.

**Non-repeatable KMeans results** can complicate auditing and reproducibility
requirements. Document the seed used for any cluster-based analysis that feeds
into a test report.

## Related concept pages

- [Data Integrity](DataIntegrity.md) — how clustering feeds into Duplicates,
  Outliers, and label_errors
- [Embeddings](Embeddings.md) — the representation clustering operates on
- [Acting on Results](ActingOnResults.md) — how to interpret and respond to
  cluster-based findings

## See this in practice

### How-to guides

- [How to perform cluster analysis](../notebooks/h2_cluster_analysis.py)
- [How to configure global hardware configuration defaults in DataEval](../notebooks/h2_configure_hardware_settings.py)

### Tutorials

- [Assessing the data space tutorial](../notebooks/tt_assess_data_space.py) —
  coverage gaps and embedding-space decisions

## References

1. [[HDBSCAN documentation](https://hdbscan.readthedocs.io/en/latest/)]{#ref1}

2. [[KMeans documentation](https://scikit-learn.org/stable/modules/clustering.html#k-means)]{#ref2}
