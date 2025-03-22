import numpy as np
from sklearn.cluster import KMeans
# from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import pairwise_distances
# from sklearn.model_selection import train_test_split
import warnings

from numpy.typing import NDArray

def get_pruned_dataset(ns, X, y, sorter, policy="keep_hard"):
    """
    Wrapper for selection policy -- not used below
        We may want something like this in the future if we use more complex
        selection policies
    Parameters
    ----------
        ns : int
            number of samples
        X : NDArray
            embedded dataset
        y : NDArray
            labels for embedded dataset
        policy : str
            selection policy: "keep_hard", "keep_easy"
        sorter:
            Sorter object
    """
    if policy=="keep_hard":
        keep_inds = sorter.srt_inds[-ns:]
    elif policy=="keep_easy":
        keep_inds = sorter.srt_inds[:ns]
    else:
        # e.g. random
        keep_inds = sorter.srt_inds[:ns]

    # assert(len(keep_inds)==ns)
    return X[keep_inds, :],y[keep_inds]

def kmeans_prio(ref_embeds, cand_embeds, num_clusters=-1):
    """
    return indices 'easiest' or most similar to reference embeddings first
    """
    validate_embedding_inputs(ref_embeds)
    validate_embedding_inputs(cand_embeds)
    num_clusters = validate_num_clusters(num_clusters, ref_embeds)
    _,num_dim = ref_embeds.shape
    _,num_dim_cand = cand_embeds.shape

    if num_dim != num_dim_cand:
        raise ValueError("Reference and candidate embeddings should have the same dimensionality")
    est = KMeans(n_clusters=num_clusters, init="k-means++", n_init=100)
    est.fit(ref_embeds)
    return np.argsort(dist2center(est, cand_embeds))

def validate_embedding_inputs(emb):
    num_samp,num_dim = emb.shape
    if emb.ndim != 2:
        raise ValueError(f"Number of dimensions should be 2; dataset should be embedded to N values per instance")
    if num_samp < 1:
        raise ValueError(f"Number of samples less than 1; please pass non-empty dataset")
    if num_dim < 1:
        raise ValueError(f"Number of dimensions less than 1; please pass non-empty dataset")

def validate_k_knn(k: int, emb : NDArray):
    num_samp,_ = emb.shape
    if k <= 0:
        # default value from size of X
        k = int(np.sqrt(num_samp))
        warnings.warn(f"Setting the value of k a default value of {k}")
    if k >= num_samp/10 and k > np.sqrt(num_samp):
        if k >= num_samp:
            raise ValueError(f"k = {k} should be smaller than size of the dataset; choose a value less than {num_samp-1}")
        else:
            warnings.warn(f"Variable k = {k} is large with respect to dataset size but valid; a nominal recommendation is sqrt(n) = {int(np.sqrt(num_samp))} for this dataset")
    return k

def validate_num_clusters(nc0, emb):
    num_samp = emb.shape[0]
    if nc0 > num_samp:
        raise ValueError(f"num_clusters should be less than dataset size ({num_samp})")
    if nc0 < 1:
        nc = int(np.sqrt(num_samp))
        warnings.warn(f"Setting the value of num_clusters to a default value of {nc}")
    else:
        nc = nc0
    return nc

def prioritize(ref_embeds, cand_embeds, method = 'knn', strategy = 'keep_easy', k = 0):
    """
    kNN prioritization: given a reference embedded dataset, identify candidate
    data that are least similar to the reference.

    Parameters
    ----------
    ref_embeds : NDArray
        reference embedded dataset
    cand_embeds : NDArray
        candidate embedded dataset to be compared to reference
    method : str
        'knn' prioritization is currently implemented
    strategy : str
        'keep_easy' or 'keep_hard'
    k : int
        nearest neighbor index in kNN graph for computing prioritization statistic
    """
    validate_embedding_inputs(ref_embeds)
    validate_embedding_inputs(cand_embeds)
    k = validate_k_knn(k, ref_embeds)
    _,num_dim = ref_embeds.shape
    _,num_dim_cand = cand_embeds.shape

    if num_dim != num_dim_cand:
        raise ValueError("Reference and candidate embeddings should have the same dimensionality")
    if method == 'knn':
        dist = np.argsort(np.sort(pairwise_distances(cand_embeds,ref_embeds), axis=1)[:,k])
    else:
        raise TypeError("Invalid method")

    if strategy == 'keep_easy':
        inds = dist
    elif strategy == 'keep_hard':
        inds = np.flip(dist)
    else:
        raise TypeError("Invalid strategy")
    return inds


class Sorter(object):
    """
    Parent class for pruning/prioritization sorting classes

    Attributes
    ----------
    num_samples : int
        number of instances or samples in the dataset
    ndim : int
        embedding dimension per sample
    srt_inds : NDArray
        list of indices sorted with 'easiest' or most prototypical samples first

    """
    def __init__(self, X):
        # future could accommodate an Embedding object
        if X.ndim != 2:
            raise ValueError(f"Number of dimensions should be 2; dataset should be embedded to N values per instance")
        validate_embedding_inputs(X)
        self.num_samples = X.shape[0]
        self.ndim = X.shape[1]
        self.srt_inds = []
        if self.num_samples < 1:
            raise ValueError(f"Number of samples less than 1; please pass non-empty dataset")
        if self.ndim < 1:
            raise ValueError(f"Number of dimensions less than 1; please pass non-empty dataset")

class KNNSorter(Sorter):
    """
    Sorting class for k-Nearest Neighbor (kNN) pruning and prioritization

    Attributes
    ----------
    num_samples : int
        number of instances or samples in the dataset
    ndim : int
        embedding dimension per sample
    srt_inds : NDArray
        list of indices sorted with 'easiest' or most prototypical samples first
    k : int
        nearest neighbor index used for ranking samples
    name : str
        sorter name for convenience
    """
    def __init__(self, X, k=-1):
        super().__init__(X)
        k = validate_k_knn(k, X)
        self.k = k
        self.name = "knn" # don't necessarily need this
        self.srt_inds = self.sort_easy_first(X)

    def sort_easy_first(self,X):
        dists = pairwise_distances(X, X)
        sort_dists = np.sort(dists, axis=0)
        self.scores = sort_dists[self.k+1, :]
        inds = np.argsort(self.scores)
        return inds

class KMeansSorter(Sorter):
    """
    Sorting class for k-Means pruning and prioritization

    Attributes
    ----------
    num_samples : int
        number of instances or samples in the dataset
    ndim : int
        embedding dimension per sample
    srt_inds : NDArray
        list of indices sorted with 'easiest' or most prototypical samples first
    num_clusters : int
        number of clusters for the kmeans.  sqrt(num_samples) gives on average
        sqrt(n) clusters each with sqrt(n) samples per cluster
    name : str
        sorter name for convenience
    """
    def __init__(self, X, num_clusters=-1):
        super().__init__(X)
        num_clusters = validate_num_clusters(num_clusters,X)
        self.num_clusters = num_clusters
        self.name="kmeans"
        self.srt_inds = self.sort_easy_first(X)

    def sort_easy_first(self,X):
        """
        sort samples by increasing difficulty/score
        """
        est = KMeans(n_clusters=self.num_clusters, init="k-means++", n_init=10)
        est.fit(X)
        dist = dist2center(est, X)
        self.scores = dist
        inds = np.argsort(dist)
        return inds

def dist2center(clst, X):
    """
    helper function for KMeansSorter, ClusterSorter
        distance from each sample to its cluster center
    """
    labels = clst.labels_
    dist = np.zeros(labels.shape)
    for lab in np.unique(labels):
        dist[labels == lab] = np.linalg.norm(X[labels==lab, :]- clst.cluster_centers_[lab, :], axis=1)
    return dist

class ClusterSorter(Sorter):
    """
    Sorting class for cluster complexity pruning and prioritization.  The
    pruning/prioritization statistic is

    Attributes
    ----------
    num_samples : int
        number of instances or samples in the dataset
    ndim : int
        embedding dimension per sample
    srt_inds : NDArray
        list of indices sorted with 'easiest' or most prototypical samples first
    num_clusters : int
        number of clusters for the kmeans.  sqrt(num_samples) gives on average
        sqrt(n) clusters each with sqrt(n) samples per cluster
    name : str
        sorter name for convenience
    """
    def __init__(self, X, num_clusters=-1):
        super().__init__(X)
        num_clusters = validate_num_clusters(num_clusters, X)
        self.num_clusters = num_clusters
        self.name="cluster_complexity"
        self.srt_inds = self.sort_easy_first(X)

    def sort_easy_first(self,X):
        clst = KMeans(n_clusters=self.num_clusters, init="k-means++", n_init=10)
        clst.fit(X)
        pr = complexity_cluster_weights(clst, X)
        inds = sort_by_cluster_weights(pr, clst, X)
        return inds

def sample_from_clusters(clst, X, npc):
    """
    Helper function for ClusterSorter
    """
    ulab = np.unique(clst.labels_)
    inds = []
    for lab,n in zip(ulab, npc):
        lab_inds = np.nonzero(clst.labels_ == lab)[0]
        dist = np.linalg.norm(X[lab_inds, :]- clst.cluster_centers_[lab, :], axis=1).squeeze()
        inds.append(lab_inds[np.argsort(dist)[-n:]])
    return np.hstack(inds)

def all_clusters_empty(c):
    """
    Helper function for ClusterSorter
    """
    return np.all([arr.size==0 for arr in c])

def sort_by_cluster_weights(pr, clst,X):
    """
    Helper function for ClusterSorter
    """
    d2c = dist2center(clst, X)
    ulab = np.unique(clst.labels_)
    inds_per_clst = []
    for lab in zip(ulab):
        inds = np.nonzero(clst.labels_==lab)[0]
        # 'hardest' first
        srt_inds = np.argsort(d2c[inds])[::-1]
        inds_per_clst.append(inds[srt_inds])
    glob_inds = []
    while not all_clusters_empty(inds_per_clst):
        clst_ind = np.random.choice(ulab, 1, p=pr)[0]
        if inds_per_clst[clst_ind].size>0:
            glob_inds.append(inds_per_clst[clst_ind][0])
        else:
            continue
        inds_per_clst[clst_ind] = inds_per_clst[clst_ind][1:]
    # sorted hardest first; reverse for consistency
    return np.array(glob_inds[::-1])

def complexity_cluster_weights(clst, X):
    """
    helper function for cluster sorter
    """
    labels = clst.labels_
    ulab = np.unique(labels)
    num_clst_intra = np.maximum(np.minimum(int(ulab.shape[0]/5), 20), 1)
    d_intra = np.zeros(ulab.shape)
    d_inter = np.zeros(ulab.shape)
    for cdx,lab in enumerate(ulab):
        d_intra[cdx] = np.mean(np.linalg.norm(X[labels==lab, :]- clst.cluster_centers_[cdx, :], axis=1))
        d_inter[cdx] = np.mean(np.linalg.norm(clst.cluster_centers_- clst.cluster_centers_[cdx, :], axis=1)[:num_clst_intra])
    cj = d_intra * d_inter
    tau = 0.1
    prob = np.exp(cj/tau) / np.sum(np.exp(cj/tau))
    return prob

class RandomSorter(Sorter):
    """
    Baseline reference class pruning and prioritization equivalent to random
    subset selection or random decimation.  This only provides a homogeneous
    interface for testing.

    Attributes
    ----------
    num_samples : int
        number of instances or samples in the dataset
    ndim : int
        embedding dimension per sample
    srt_inds : NDArray
        list of indices sorted with 'easiest' or most prototypical samples first
    name : str
        sorter name for convenience
        """
    def __init__(self, X, num_clusters=500):
        super().__init__(X)
        validate_embedding_inputs(X)
        inds = np.arange(self.num_samples, dtype=int)
        self.name = "random"
        np.random.shuffle(inds) # in place
        self.srt_inds = inds
        self.scores = np.random.rand(inds.shape[0])
