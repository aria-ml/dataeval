import torch
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from torch.utils.data import DataLoader


from scipy.stats import ks_2samp, iqr
from scipy.stats import wasserstein_distance as emd
import numpy as np

from typing import Dict, List, Union, Mapping
from numpy.typing import NDArray, ArrayLike

from metadata_utils import collate_fn_2 as collate_fn

from dataeval._internal.metrics.metadata_ood_mi import get_metadata_ood_mi
from dataeval._internal.metrics.metadata_ks_compare import meta_distribution_compare

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
    
    MI_dict = get_metadata_ood_mi(mdict, is_ood, discrete_features=kwargs.get('discrete_features'))
    
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
def ks_compare(dl0, dl1, k_stop=None, debug=None):
    # if every ks stat changes by less than k_stop, then stop adding data to the sample
    k_stop = 5e-3 if k_stop is None else k_stop

    debug = False if debug is None else debug
    
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
        results = meta_distribution_compare(dol0, dol1)
        for k in dol0:
            x0, x1 = dol0[k], dol1[k]
            allx = x0 + x1 # x0 and x1 are lists, so + concatenates them. 
            xmin = min(allx) 
            xmax = max(allx)

            if xmax > xmin:
                results[k].statistic_location = (results[k].statistic_location - xmin)/(xmax - xmin)
                  
            del_ks[k] = np.abs(results[k].statistic - ks_prev[k])
            stable = stable and (del_ks[k] < k_stop)  # *all* quantities must be stable before we quit.  
            # ks_prev[k] = res.statistic
            ks_prev[k] = results[k].statistic

        arg_max, maxdk = max(list(enumerate([del_ks[k] for k in dol0])), key=lambda x: x[1])
        maxkey = [k for k in dol0.keys()][arg_max]
        if debug:
             print(f'{len(dol0[k])}: {maxkey} {maxdk:.3f}: {results[maxkey].statistic_location:.3f}')       

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
    bbshape = (*test_images.shape, 1)

    is_ood = ood_detector.predict(test_images.reshape(bbshape)).is_ood # is_ood is bool
    
    test_images = test_images.reshape(bbshape)

    big_batch_size = len(refds)

    refbb = DataLoader(refds, collate_fn=collate_fn, batch_size=big_batch_size)

    for _, _, metadata in refbb:
        break
    metadata = _lod2dol(metadata)
    
    testbb = DataLoader(testds, collate_fn=collate_fn, batch_size=big_batch_size)

    for _, _, corrmetadata in testbb:
        break
    corrmetadata  = _lod2dol(corrmetadata)

    unlikely_features = get_least_likely_features(metadata, corrmetadata, is_ood)
    uvals, freq = np.unique(unlikely_features, return_counts=True)

    iord = np.argsort(freq,)[::-1] # decreasing order
    names = [k for k in uvals]
    maxlen = max([len(name) for name in names])

    hdr = 'feature'
    print(f'{hdr:{maxlen}}|  occurences')
    print('='*(maxlen+14))
    for i in iord:
            print(f'{names[i]:{maxlen}}:    {freq[i]}')
    
    return unlikely_features
    
def get_least_likely_features(metadata, corrmetadata, is_ood)-> NDArray[str]:
    r"""Computes which metadata feature is most :term:`out of distribution<Out-of-distribution (OOD)>` relative to a reference metadata set. 
    
    Given a reference metadata dictionary `metadata` (where each key maps to one scalar metadata feature), a second 
    metadata dictionary, and a corresponding boolean flag `is_ood` indicating whether each example falls 
    out-of-distribution (OOD) relative to the reference, this function finds which metadata feature is the most OOD,
    for each OOD example. 

    Parameters
    ----------
    metadata:
        A reference set of arrays of values, indexed by metadata feature names, with one value per data example per feature. 
    corrmetadata:
        A second metedata set, to be tested against the reference metadata. It is ok if the two meta data objects hold different 
        numbers of examples.  
    is_ood:
        A boolean array, with one value per corrmetadata example, that indicates which examples are OOD. 
    
    Returns
    -------
    NDArray[str]
        An array of names of the features of each OOD corrmetadata example that were the most OOD. 

    Examples
    --------
    Imagine we have 3 data examples, and that the corresponding metadata contains 2 features called time and altitude.

>>> from metadata_tools import get_least_likely_features
>>> import numpy
>>> rng = numpy.random.default_rng(123)
>>> metadata = {'time': [1.2, 3.4, 5.6], 'altitude': [235, 6789, 101112]}
>>> corrmetadata = {'time': [7.8, 9.10, 11.12], 'altitude': [532, 9876, 211101]}
>>> is_ood = rng.choice(a=[False, True], size=len(metadata['time']))
>>> get_least_likely_features(metadata, corrmetadata, is_ood)
array(['time', 'time'], dtype='<U4')

"""   
    norm_dict = {}
    for k,v in metadata.items():
        loc = np.median(v)
        dev = v - loc
        posdev, negdev = dev[dev > 0], dev[dev < 0]
        pos_scale = np.median(posdev)
        neg_scale = np.abs(np.median(negdev))

        norm_dict.update({k: {'loc': loc, 'pos_scale': pos_scale, 'neg_scale': neg_scale}})

    mkeys = [k for k in norm_dict.keys()]

    maxpdev = np.array([-1e30 for _ in is_ood])
    maxndev = -1.0 * maxpdev

    deviation = np.zeros(is_ood.shape)
    ikmax = np.zeros(is_ood.shape, dtype=np.int32)
    for ik, k in enumerate(mkeys):
        if k == 'random': # exclude cases where random happens to be out on tails, not interesting. 
            continue 
        ndk = norm_dict[k]
        x, x0, dxp, dxn = corrmetadata[k], ndk['loc'], ndk['pos_scale'], ndk['neg_scale']
        dxp = dxp if dxp > 0 else 1.0
        dxn = dxn if dxn > 0 else 1.0

        xdev = x - x0
        pos, neg = xdev >= 0, xdev < 0

        X = np.zeros_like(x)
        X[pos], X[neg] = xdev[pos]/dxp, xdev[neg]/dxn

        pbig, nbig, abig = X > maxpdev, X < maxndev, np.abs(X) > deviation

        update_mpdev, update_mndev = np.logical_and(pbig, abig), np.logical_and(nbig, abig)
        maxpdev[update_mpdev], maxndev[update_mndev] = X[update_mpdev], X[update_mndev]

        update_k = np.logical_or(update_mpdev, update_mndev)
        ikmax[update_k] = ik
        deviation[update_k] = np.abs(X[update_k])

    unlikely_features = np.array([mkeys[ik] for ik in ikmax])[is_ood]
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