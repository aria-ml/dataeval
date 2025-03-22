

import pytest
import numpy as np

from pruning_utils import KNNSorter,KMeansSorter,ClusterSorter,prioritize,kmeans_prio


class TestSorterInputs:
    """Tests valid combinations of inputs"""
    @pytest.mark.parametrize(
        "sorter, emb, arg",
        [
            pytest.param(KNNSorter,np.ones((4,3)),5),
            pytest.param(KMeansSorter,np.ones((4,3)),5),
            pytest.param(ClusterSorter,np.ones((4,3)),5),
            pytest.param(KNNSorter,np.ones((4,3)),0),
            pytest.param(KMeansSorter,np.ones((4,3)),0),
            pytest.param(ClusterSorter,np.ones((4,3)),0),
        ]
    )
    def test_embedding_shapes(self, sorter,emb,arg):
        if arg > emb.shape[0]:
            with pytest.raises(ValueError):
                srt = sorter(emb,arg)
        elif arg < 1:
            with pytest.warns(UserWarning, match="value of"):
                srt = sorter(emb,arg)

    def test_prioritization_mismatched_input_dims(self):
        emb_ref = np.ones((5,3))
        emb_cnd = np.ones((3,4))
        with pytest.raises(ValueError):
            prioritize(emb_ref, emb_cnd, method="knn", strategy="keep_easy", k=2)

    def test_prioritization_k_gt_n(self):
        emb_ref = np.ones((5,3))
        emb_cnd = np.ones((3,3))
        with pytest.raises(ValueError):
            prioritize(emb_ref, emb_cnd, method="knn", strategy="keep_easy", k=6)

    def test_prioritization_default_k(self):
        emb_ref = np.ones((5,3))
        emb_cnd = np.ones((3,3))
        with pytest.warns(UserWarning, match="value of"):
            prioritize(emb_ref, emb_cnd, method="knn", strategy="keep_easy", k=0)