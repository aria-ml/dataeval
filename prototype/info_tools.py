import numpy as np

# from intrinsic_factors import (box_location_features, compute_hwa,
#                                get_image_sizes)
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


def infer_categorical(X, threshold: float = 0.25):
    """
    Compute fraction of feature values that are unique --- intended to be used
    for inferring whether variables are categorical.

    Notes:
        - assume all features
    """
    if X.ndim == 1:
        X = np.expand_dims(X, axis=1)
    num_samples = X.shape[0]
    pct_unique = np.empty(X.shape[0])
    for col in X.shape[1]:
        uvals = np.unique(X[:, col], axis=0)
        pct_unique[col] = len(uvals) / num_samples
    return pct_unique < threshold


def compute_mi_class(props, class_var, cat_vars, con_vars, return_sorted=True):
    """
    Mutual information (MI) with respect to a class label

    Notes:
        - we use mutual_info_classif from sklearn since class label is categorical
        - mutual_info_classif outputs are consistent up to O(1e-4) and depend on
          a random seed
        - MI is computed differently for categorical and continuous variables,
          so we have to specify.

    Inputs:
        - props: dict of arraylike variables including class labels, intrinsic
          factors, and extrinsic factors
        - class_var: str dictionary key corresponding to class label
        - cat_vars: List[str] list of categorical or discrete variable names
        - con_vars: List[str] list of continuous variable names
        - return_sorted: bool if True, return sorted MI and corresponding
          variable names
    """
    tgt = props[class_var]
    feat_cat = np.array([props[var] for var in cat_vars]).T
    feat_con = np.array([props[var].astype(float) for var in con_vars]).T
    all_feat = np.concatenate([feat_cat, feat_con], axis=1)
    var_names = list(cat_vars) + list(con_vars)
    cat_mask = [True for _ in range(len(cat_vars))] + [False for _ in range(len(con_vars))]
    # units: nat
    mi = mutual_info_classif(all_feat, tgt, discrete_features=cat_mask)

    ent_class = entropy_(props[class_var], [True])
    ent_all = entropy_(all_feat, cat_mask)
    nmi = mi / np.sqrt(ent_class[0] * ent_all)

    if return_sorted:
        srt_inds = np.argsort(mi)
        var_names = [var_names[i] for i in srt_inds]
        nmi = nmi[srt_inds]

    return nmi, var_names


def compute_mutual_information(factors, is_categorical, num_neighbors=5, return_sorted=False):
    # filter categorical/discrete and continuous variables
    var_names = list(factors.keys())
    cat_vars = [v for v in var_names if is_categorical[v]]
    con_vars = [v for v in var_names if not is_categorical[v]]

    num_cat, num_con = len(cat_vars), len(con_vars)
    num_feat = num_cat + num_con
    mi = np.empty((num_feat, num_feat))
    mi[:] = np.nan
    feat_cat = np.array([factors[var] for var in cat_vars]).T
    feat_con = np.array([factors[var].astype(float) for var in con_vars]).T
    all_feat = np.concatenate([feat_cat, feat_con], axis=1)
    cat_mask = [True for _ in range(len(cat_vars))] + [False for _ in range(len(con_vars))]
    # classification MI for discrete/categorical features
    for idx, tgt_var in enumerate(cat_vars):
        tgt = factors[tgt_var]
        # units: nat
        mi[idx, :] = mutual_info_classif(all_feat, tgt, discrete_features=cat_mask, n_neighbors=num_neighbors)

    # regression MI for continuous features
    for idx, tgt_var in enumerate(con_vars):
        tgt = factors[tgt_var]
        # units: nat
        mi[idx + num_cat, :] = mutual_info_regression(
            all_feat, tgt, discrete_features=cat_mask, n_neighbors=num_neighbors
        )

    ent_all = entropy_(all_feat, cat_mask)
    norm_factor = np.sqrt(np.outer(ent_all, ent_all))
    norm_factor = 0.5 * np.sum.outer(ent_all, ent_all) + 1e-6
    nmi = 0.5 * (mi + mi.T) / norm_factor

    return nmi, var_names


def compute_mutual_information_class(factors, is_categorical, class_var="class", num_neighbors=5, return_sorted=False):
    """
    Compute MI with respect to class for one class against the rest rather than
    all classes together.  Look for correlation of one class with factors rather
    than all classes with factors.
    """

    # filter categorical/discrete and continuous variables
    var_names = list(factors.keys())
    cat_vars = [v for v in var_names if is_categorical[v]]
    con_vars = [v for v in var_names if not is_categorical[v]]
    cat_vars.remove(class_var)

    u_cls = np.unique(factors[class_var])
    num_classes = len(u_cls)

    num_cat, num_con = len(cat_vars), len(con_vars)
    num_feat = num_cat + num_con
    mi = np.empty((num_classes, num_feat))
    mi[:] = np.nan
    feat_cat = np.array([factors[var] for var in cat_vars]).T
    feat_con = np.array([factors[var].astype(float) for var in con_vars]).T
    all_feat = np.concatenate([feat_cat, feat_con], axis=1)
    cat_mask = [True for _ in range(len(cat_vars))] + [False for _ in range(len(con_vars))]
    # classification MI for discrete/categorical features
    for idx, cls in enumerate(u_cls):
        tgt = factors[class_var] == cls
        # units: nat
        mi[idx, :] = mutual_info_classif(all_feat, tgt, discrete_features=cat_mask, n_neighbors=num_neighbors)

    ent_all = entropy_(all_feat, cat_mask)
    ent_tgt = entropy_(factors[class_var], [True])
    norm_factor = 0.5 * np.sum.outer(ent_tgt, ent_all) + 1e-6
    nmi = mi / norm_factor

    return nmi, var_names


def compute_corr_matrix(props, cat_vars, con_vars, num_neighbors=7):
    num_cat, num_con = len(cat_vars), len(con_vars)
    num_feat = num_cat + num_con
    mi = np.empty((num_feat, num_feat))
    mi[:] = np.nan
    feat_cat = np.array([props[var] for var in cat_vars]).T
    feat_con = np.array([props[var].astype(float) for var in con_vars]).T
    all_feat = np.concatenate([feat_cat, feat_con], axis=1)
    cat_mask = [True for _ in range(len(cat_vars))] + [False for _ in range(len(con_vars))]
    var_names = list(cat_vars) + list(con_vars)
    # classification MI for discrete/categorical features
    for idx, tgt_var in enumerate(cat_vars):
        tgt = props[tgt_var]
        # units: nat
        mi[idx, :] = mutual_info_classif(all_feat, tgt, discrete_features=cat_mask, n_neighbors=num_neighbors)

    # regression MI for continuous features
    for idx, tgt_var in enumerate(con_vars):
        tgt = props[tgt_var]
        # units: nat
        mi[idx + num_cat, :] = mutual_info_regression(
            all_feat, tgt, discrete_features=cat_mask, n_neighbors=num_neighbors
        )

    ent_all = entropy_(all_feat, cat_mask)
    norm_factor = 0.5 * np.sum.outer(ent_all, ent_all) + 1e-6
    nmi = 0.5 * (mi + mi.T) / norm_factor

    return nmi, var_names


def entropy_(X, discrete_features):
    """
    Compute entropy for discrete/categorical variables and, through standard
    histogram binning, for continuous variables.
    """
    if X.ndim == 1:
        X = np.expand_dims(X, axis=1)
    num_vars = X.shape[1]
    ent = np.empty(num_vars)
    # loop over columns for convenience
    for col in range(num_vars):
        if discrete_features[col]:
            # if discrete, use unique values as bins
            _, counts = np.unique(X[:, col], return_counts=True)
        else:
            counts, _ = np.histogram(X[:, col], bins="auto", density=True)

        # entropy in nats, normalizes counts
        ent[col] = entropy(counts)

    return ent


def validate_dict(d):
    # assert that length of all arrays are equal -- could expand to other properties
    lengths = []
    for arr in d.values():
        lengths.append(arr.shape)
    assert lengths[1:] == lengths[:-1]


def cat_to_int(x):
    """
    map categorical variables to numbers that mutual_information can accommodate
    """
    ux, mapped_vals = np.unique(x, return_inverse=True)
    return np.array(mapped_vals), ux


def str2int(d):
    for key, val in d.items():
        # if not numeric
        if not np.issubdtype(val.dtype, np.number):
            _, mapped_vals = np.unique(val, return_inverse=True)
            d[key] = mapped_vals
    return d
