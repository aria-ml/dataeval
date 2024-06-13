import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fmow_utils import get_image_sizes
from intrinsic_factors import box_location_features, compute_hwa_xyxy
from matplotlib.patches import Rectangle
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
    for col in X.shape[1]:  # type: ignore
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
    _vars = list(cat_vars) + list(con_vars)
    cat_mask = [True for _ in range(len(cat_vars))] + [False for _ in range(len(con_vars))]
    # units: nat
    mi = mutual_info_classif(all_feat, tgt, discrete_features=cat_mask)

    ent_class = entropy_(props[class_var], [True])
    ent_all = entropy_(all_feat, cat_mask)
    nmi = mi / np.sqrt(ent_class[0] * ent_all)

    if return_sorted:
        srt_inds = np.argsort(mi)
        _vars = [_vars[i] for i in srt_inds]  # vars[srt_inds]
        nmi = nmi[srt_inds]

    return nmi, _vars


def compute_mutual_information(factors, is_categorical, num_neighbors=5, return_sorted=False):
    # filter categorical/discrete and continuous variables
    _vars = list(factors.keys())
    cat_vars = [v for v in _vars if is_categorical[v]]
    con_vars = [v for v in _vars if not is_categorical[v]]

    num_cat, num_con = len(cat_vars), len(con_vars)
    num_feat = num_cat + num_con
    mi = np.empty((num_feat, num_feat))
    mi[:] = np.nan
    feat_cat = np.array([factors[var] for var in cat_vars]).T
    feat_con = np.array([factors[var].astype(float) for var in con_vars]).T
    all_feat = np.concatenate([feat_cat, feat_con], axis=1)
    cat_mask = [True for _ in range(len(cat_vars))] + [False for _ in range(len(con_vars))]
    _vars = list(cat_vars) + list(con_vars)
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
    nmi = 0.5 * (mi + mi.T) / norm_factor

    return nmi, _vars


def compute_mutual_information_class(factors, is_categorical, class_var="class", num_neighbors=5, return_sorted=False):
    """
    Compute MI with respect to class for one class against the rest rather than
    all classes together.  Look for correlation of one class with factors rather
    than all classes with factors.
    """

    # filter categorical/discrete and continuous variables
    _vars = list(factors.keys())
    cat_vars = [v for v in _vars if is_categorical[v]]
    con_vars = [v for v in _vars if not is_categorical[v]]
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
    _vars = list(cat_vars) + list(con_vars)
    # classification MI for discrete/categorical features
    for idx, cls in enumerate(u_cls):
        tgt = factors[class_var] == cls
        # units: nat
        mi[idx, :] = mutual_info_classif(all_feat, tgt, discrete_features=cat_mask, n_neighbors=num_neighbors)

    ent_all = entropy_(all_feat, cat_mask)
    ent_tgt = entropy_(factors[class_var], [True])
    norm_factor = np.sqrt(np.outer(ent_tgt, ent_all))
    nmi = mi / norm_factor

    return nmi, _vars


def compute_corr_matrix(props, cat_vars, con_vars, num_neighbors=7):
    num_cat, num_con = len(cat_vars), len(con_vars)
    num_feat = num_cat + num_con
    mi = np.empty((num_feat, num_feat))
    mi[:] = np.nan
    feat_cat = np.array([props[var] for var in cat_vars]).T
    feat_con = np.array([props[var].astype(float) for var in con_vars]).T
    all_feat = np.concatenate([feat_cat, feat_con], axis=1)
    cat_mask = [True for _ in range(len(cat_vars))] + [False for _ in range(len(con_vars))]
    _vars = list(cat_vars) + list(con_vars)
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
    # norm_factor = np.sqrt(np.outer(ent_all, ent_all))+1e-6
    norm_factor = 0.5 * np.sum.outer(ent_all, ent_all) + 1e-6
    nmi = 0.5 * (mi + mi.T) / norm_factor

    # nmi = mi / np.sqrt(ent_class[0]*ent_all)
    # primary bias metric --- MI with target class
    # vars = list(cat_vars)+list(con_vars)
    # mi = np.concatenate([mi_cat, mi_con])
    # srt_inds = np.sort()
    # vars = vars[srt_inds]
    # mi = mi[srt_inds]

    return nmi, _vars


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
    # assert that length of all arrays are equal
    lengths = []
    for arr in d.values():
        lengths.append(arr.shape)
    assert lengths[1:] == lengths[:-1]


def cat_to_int(x):
    """
    map categorical variables to numbers that mutual_information can accommodate
    """
    ux, mapped_vals = np.unique(x, return_inverse=True)
    # rand_classes = random.sample(list(np.arange(len(ux))), len(ux))
    # rand_mapped = [rand_classes[c] for c in mapped_vals]
    return np.array(mapped_vals), ux


def str2int(d):
    for key, val in d.items():
        if not np.issubdtype(val.dtype, np.number):
            _, mapped_vals = np.unique(val, return_inverse=True)
            d[key] = mapped_vals
    return d


if __name__ == "__main__":
    ldf = pd.read_csv("/mnt/nas_device_0/xview/labels/class_labels.csv", keep_default_na=False)
    md = pd.read_csv("/mnt/nas_device_0/xview/labels/inverse_meta.csv", keep_default_na=False)

    # boxes are xyxy
    df = md.merge(ldf, left_on="obj_id", right_on="id")
    boxes = df[["x_min", "y_min", "x_max", "y_max"]].to_numpy()

    # image sizes
    sz = get_image_sizes("/mnt/nas_device_0/xview/data/train_images")
    # exploded -- none values indicate labels in the dataframe with no
    #   corresponding image
    sz_exp = np.array([sz.get(_id, [None, None]) for _id in df.fn])
    missing_imgs = sz_exp[:, 0] is None
    df = df[~missing_imgs]
    sz_exp = sz_exp[~missing_imgs, :]
    boxes = boxes[~missing_imgs, :]

    prop = {}
    prop["height"], prop["width"], prop["box_size"], prop["aspect_ratio"] = compute_hwa_xyxy(boxes)
    prop["dist_to_center"], prop["dist_to_edge"] = box_location_features(boxes, sz_exp)
    # convert to int
    prop["size_cat"], size_map = cat_to_int(df["size"])
    #  = df["size"].to_numpy()
    prop["rarity"], rarity_map = cat_to_int(df["rarity"])
    prop["class_id"] = df["id"].to_numpy()
    validate_dict(prop)

    categorical_vars = ["size_cat", "rarity"]
    continuous_vars = ["height", "width", "box_size", "aspect_ratio", "dist_to_center", "dist_to_edge"]
    class_var = "class_id"

    mi_class, mi_class_vars = compute_mi_class(prop, class_var, categorical_vars, continuous_vars)
    mi, mi_vars = compute_corr_matrix(prop, categorical_vars, continuous_vars)
    jkl = 23

    f, ax = plt.subplots(figsize=(8, 6))
    mask = np.zeros_like(mi, dtype=np.bool_)
    mask[np.tril_indices_from(mask)] = True

    # Want diagonal elements as well
    mask[np.diag_indices_from(mask)] = True
    # Generate a custom diverging co lormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    all_mi = np.concatenate([np.expand_dims(mi_class, 0), mi], 0)
    all_mask = np.concatenate([np.zeros((1, mi.shape[0]), dtype=bool), mask], 0)

    # Draw the heatmap with the mask and correct aspect ratio
    sns_plot = sns.heatmap(
        all_mi,
        mask=all_mask,
        cmap=cmap,
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5, "label": "Mutual Information [nat]"},
        xticklabels=mi_vars,
        yticklabels=["class"] + mi_vars,
        annot=True,
    )
    ax.add_patch(Rectangle((0, 0), mi.shape[0], 1, fill=False, edgecolor="k", lw=4))
    # save to file
    plt.tight_layout()
    fig = sns_plot.get_figure()
    fig.savefig("figs/nmi_map.png")
