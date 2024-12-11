import numpy as np
from numpy.typing import NDArray, ArrayLike

import torch
from torch.utils.data import DataLoader
from metadata_utils import collate_fn_2 as collate_fn

from scipy.stats import ks_2samp, iqr
from scipy.stats import wasserstein_distance as emd

from sklearn.feature_selection import mutual_info_classif

from typing import Dict, List, Union, Mapping

from dataeval.detectors.ood.metadata_ood_mi import get_metadata_ood_mi
from dataeval.detectors.ood.metadata_ks_compare import meta_distribution_compare
from dataeval.detectors.ood.metadata_least_likely import get_least_likely_features

import pytest


#<a name="predict_ood_mi"></a>
def predict_ood_mi(refdl, newdl, ood_detector, **kwargs):

    images, metadata, corrimages, corrmetadata = torch.empty(0), [], torch.empty(0), [] # tired of warnings from VSCode
    
    for images, _, metadata in refdl:
        break

    for corrimages, _, corrmetadata in newdl:
          break

    allimages = np.concatenate((images, corrimages), axis=0)
    # input_shape = (*allimages[0].shape, 1)
    # bbshape = (*allimages.shape,1)
    # allimages = allimages.reshape(bbshape) 

    metadata.extend(corrmetadata)

    mdict = {}
    for k in metadata[0]:
        mdict.update({k: []})

    for k in metadata[0]:
        mdict[k].extend([d[k] for d in metadata])

    for k in mdict:
        mdict[k] = np.array(mdict[k])

    is_ood = ood_detector.predict(allimages).is_ood
    
    MI_dict = get_metadata_ood_mi(mdict, is_ood, discrete_features=kwargs.get('discrete_features')) # type: ignore
    
    MI = np.array([MI_dict[k] for k in MI_dict])
    iord = np.argsort(MI)[::-1] # decreasing order
    names = [k for k in MI_dict]
    maxlen = max([len(name) for name in names])

    hdr = 'feature'
    print(f'{hdr:{maxlen}} |  MI (bits)')
    print('='*(maxlen+14))
    for i in iord:
         print(f'{names[i] :{maxlen}}:    {MI[i]:.3f}')

    return MI_dict


@pytest.fixture
def mock_md_ood():
    rng = np.random.default_rng(20241022)
    nsamp, nfeatures = 100, 5
    x = rng.normal(size=(nsamp, nfeatures))
    # features 0, 1, and 2 should have some MI, but 3 and 4 will not
    is_ood = np.abs(x[:, 0]) > np.abs(x[:,1]) + np.abs(x[:,2])

    md = {}
    for i in range(nfeatures):
        md.update({'feature_'+str(i): x[:,i]})

    discrete_features = False

    mi_dict = get_metadata_ood_mi(md, is_ood, discrete_features=discrete_features)

    return mi_dict

class TestGetMetadataOodMi():
    def test_type(self, mock_md_ood):
        assert isinstance(mock_md_ood, dict)

    def test_mi_values(self, mock_md_ood):
        mi_dict = mock_md_ood
        assert np.allclose([v for v in mi_dict.values()], [0.17562533, 0.18743624, 0.15217528, 0.        , 0.        ])


# <a name="ks_compare"></a>
def ks_compare(dl0, dl1, k_stop=None):
    # if every ks stat changes by less than k_stop, then stop adding data to the sample
    k_stop = 5e-3 if k_stop is None else k_stop
    
    dol0, dol1, results, ks_prev, del_ks = {}, {}, {}, {}, {}
    first = True
    for (_, _, md0), (_, _, md1) in zip(dl0, dl1):
        if first:
            for k in md0[0]:
                  ks_prev[k] = -1
                  dol0.update({k: []})
                  dol1.update({k: []})
            first = False

        for k in md0[0]:
            dol0[k].extend([d[k] for d in md0])
            dol1[k].extend([d[k] for d in md1])

        stable = True # start True, then do logical and with every KS stat change < k_stop
        results = meta_distribution_compare(dol0, dol1).mdc
        for k in dol0:
            del_ks[k] = np.abs(results[k].statistic - ks_prev[k])
            stable = stable and (del_ks[k] < k_stop)  # *all* quantities must be stable before we quit.  
            ks_prev[k] = results[k].statistic

        arg_max, maxdk = max(list(enumerate([del_ks[k] for k in dol0])), key=lambda x: x[1])
        maxkey = [k for k in dol0.keys()][arg_max]

        if stable:
            break
    else:
        pass

    pvals = [v.pvalue for v in results.values()]
    shifts = [v.shift_magnitude for v in results.values()]
    iord = np.argsort(pvals)
    names = [k for k in results]
    maxlen = max([len(name) for name in names])

    hdr = 'feature'
    print(f'{hdr:{maxlen}}| p-value | shift/IQR')
    print('='*(maxlen+21))
    for i in iord:
         print(f'{names[i]:{maxlen}}:  {pvals[i]:.3f}  :   {shifts[i]:.3f}')

    return results



@pytest.fixture
def mock_mdc():
    md0 = {'time': [1.2, 3.4, 5.6], 'altitude': [235, 6789, 101112]}
    md1 = {'time': [7.8, 9.10, 11.12], 'altitude': [532, 9876, 211101]}
    mdc = meta_distribution_compare(md0, md1)
    return mdc

class TestMetadataCompare():
    # test conditions, as many as you can think of
    def test_type(self, mock_mdc):
        mdc = mock_mdc
        assert isinstance(mdc, dict) 
    
    def test_shifts(self, mock_mdc):
        mdc = mock_mdc
        assert mdc['time'].shift_magnitude == 2.7 and  np.isclose(mdc['altitude'].shift_magnitude,  0.7492490855199898)

    def test_pvalue(self, mock_mdc):
        mdc = mock_mdc
        assert np.isclose(mdc['time'].pvalue, 0.0) and np.isclose(mdc['altitude'].pvalue, 0.9444444444444444)


# <a name="least_likely_features"></a>
def least_likely_features(refds, testds, ood_detector):

    test_images = testds.images

    is_ood = ood_detector.predict(test_images).is_ood # is_ood is bool
    
    big_batch_size = len(refds)

    refbb = DataLoader(refds, collate_fn=collate_fn, batch_size=big_batch_size)

    for _, _, metadata in refbb:
        break
    metadata = _lod2dol(metadata) # type: ignore
    
    testbb = DataLoader(testds, collate_fn=collate_fn, batch_size=big_batch_size)

    for _, _, corrmetadata in testbb:
        break
    corrmetadata  = _lod2dol(corrmetadata) # type: ignore

    unlikely_features = get_least_likely_features(metadata, corrmetadata, is_ood)

    uvals, freq = np.unique([uftp[0] for uftp in unlikely_features], return_counts=True)

    iord = np.argsort(freq,)[::-1] # decreasing order
    names = [k for k in uvals]
    maxlen = max([len(name) for name in names])

    hdr = 'feature'
    print(f'{hdr:{maxlen}}|  occurences')
    print('='*(maxlen+14))
    for i in iord:
            print(f'{names[i]:{maxlen}}:    {freq[i]}')
    
    return unlikely_features
    


def _lod2dol(lod, to_numpy=None):
    to_numpy = False if to_numpy is None else to_numpy

    dol = {}
    for k in lod[0]:
        dol.update({k: []})

    for k, v in dol.items():
        dol[k].extend([d[k] for d in lod])

    if to_numpy:
        for v in dol.values():
            v = np.array(v)

    return dol

@pytest.fixture
def mock_llf():
    md0 = {'time': [1.2, 3.4, 5.6], 'altitude': [235, 6789, 101112]}
    md1 = {'time': [7.8, 9.10, 11.12], 'altitude': [532, 9876, 211101]}

    is_ood = np.array([False, False, True])

    llf = get_least_likely_features(md0, md1, is_ood)
    return llf

class TestGetLeastLikelyFeatures():
    def test_nothing(self, mock_llf):
        features = mock_llf
        assert True

    def test_llf(self, mock_llf):
        llf = mock_llf
        assert llf[0] == 'time'

    def test_llf_type(self, mock_llf):
        crap = mock_llf
        assert isinstance(crap, np.ndarray)

# if __name__ == "__main__":
#     import doctest
#     doctest.testmod()