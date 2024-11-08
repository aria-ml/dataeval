import numpy as np
import numba

@numba.njit(parallel=True, locals={"i": numba.types.int32})
def _group_mst_by_clusters(mst, clusters):
    cluster_grouping = np.full(mst.shape[0], -1, dtype=np.int16)
    for i in numba.prange(mst.shape[0]):
        cluster_id = clusters[np.int32(mst[i,0])]
        if cluster_id == clusters[np.int32(mst[i,1])]:
            cluster_grouping[i] = np.int16(cluster_id)

    return cluster_grouping

@numba.njit()
def _cluster_variance(clusters, mst_clusters, mst):
    cluster_ids = np.unique(clusters)
    cluster_std = np.zeros_like(cluster_ids, dtype=np.float32)

    for i in range(cluster_ids.size):
        cluster_links = np.nonzero(mst_clusters==cluster_ids[i])[0]
        cluster_std[i] = mst[cluster_links, 2].std()

    return cluster_std

@numba.njit(parallel=True, locals={"i": numba.types.int32})
def _compare_links_to_std(cluster_std, mst_clusters, mst):
    overall_mean = mst.T[2].mean()
    order_mag = np.floor(np.log10(overall_mean))
    compare_mag = -3 if order_mag >= 0 else order_mag - 3
    
    cluster_ids = np.unique(mst_clusters)
    exact_dup = np.full((mst.shape[0],2), -1, dtype=np.int32)
    near_dup = np.full((mst.shape[0],2), -1, dtype=np.int32)

    for i in numba.prange(mst.shape[0]):
        cluster_id = mst_clusters[i]
        std_loc = np.nonzero(cluster_ids==cluster_id)[0]
        std = cluster_std[std_loc]

        if mst[i,2] < 10**compare_mag:
            exact_dup[i] = (np.int32(mst[i,0]), np.int32(mst[i,1]))
        elif mst[i,2] < std:
            near_dup[i] = (np.int32(mst[i,0]), np.int32(mst[i,1]))
    
    exact_idx = np.nonzero(exact_dup.T[0]!=-1)[0]
    near_idx = np.nonzero(near_dup.T[0]!=-1)[0]

    return exact_dup[exact_idx], near_dup[near_idx]
         