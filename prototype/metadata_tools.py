import torch
import pandas as pd
import statsmodels.api as sm
from sklearn.feature_selection import mutual_info_classif

from torch.utils.data import DataLoader


from scipy.stats import ks_2samp, iqr
from scipy.stats import wasserstein_distance as emd
import numpy as np

#<a name="predict_ood_mi"></a>
def predict_ood_mi(refdl, newdl, ood_detector, discrete_features=None):
    nats2bits = 1.442695

    discrete_features = False if discrete_features is None else discrete_features

    images, metadata, corrimages, corrmetadata = torch.empty(0), [], torch.empty(0), [] # tired of warnings from VSCode
    
    for images, _, metadata in refdl:
        break

    for corrimages, _, corrmetadata in newdl:
          break

    allimages = np.concatenate((images, corrimages), axis=0)
    input_shape = (*allimages[0].shape, 1)
    bbshape = (*allimages.shape,1)
    allimages = allimages.reshape(bbshape) 

    metadata.extend(corrmetadata)

    mdict = {}
    for k in metadata[0]:
        mdict.update({k: []})

    for k in metadata[0]:
        mdict[k].extend([d[k] for d in metadata])

    for k in mdict:
        mdict[k] = np.array(mdict[k])

    df = pd.DataFrame.from_dict(mdict)

    is_ood = ood_detector.predict(allimages).is_ood
    y = is_ood

    X = df[[k for k in mdict.keys()]].to_numpy() # convenient way to make columnwise features. 
    
    X0, dX = np.mean(X, axis=0), np.std(X, axis=0, ddof=1)
    Xscl = (X - X0)/dX

    MI = mutual_info_classif(Xscl, y, discrete_features=discrete_features) * nats2bits
    
    iord = np.argsort(MI,)[::-1] # decreasing order
    names = [k for k in mdict]
    maxlen = max([len(name) for name in names])

    hdr = 'feature'
    print(f'{hdr:{maxlen}} |  MI (bits)')
    print('='*(maxlen+14))
    for i in iord:
         print(f'{names[i] :{maxlen}}:    {MI[i]:.3f}')


# <a name="ks_compare"></a>
def ks_compare(dl0, dl1, k_stop=None, debug=None):

    two_samp = ks_2samp

    # if every ks stat changes by less than k_stop, stop adding data
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
        for k in dol0:
            x0, x1 = dol0[k], dol1[k]
            allx = x0 + x1 # x0 and x1 are lists, so + concatenates them. 
            xmin = min(allx) 
            xmax = max(allx)

            res = two_samp(x0, x1, method='asymp')
            results.update({k: res})
            if xmax > xmin:
                  results[k].statistic_location = (results[k].statistic_location - xmin)/(xmax - xmin)
                  
            del_ks[k] = np.abs(res.statistic - ks_prev[k])
            stable = stable and (del_ks[k] < k_stop)  # *all* quantities must be stable before we quit.  
            ks_prev[k] = res.statistic

        arg_max, maxdk = max(list(enumerate([del_ks[k] for k in dol0])), key=lambda x: x[1])
        maxkey = [k for k in dol0.keys()][arg_max]
        if debug:
             print(f'{len(dol0[k])}: {maxkey} {maxdk:.3f}: {results[maxkey].statistic_location:.3f}')       

        if stable:
            break
    else:
        pass

    for k in dol0:
        x0, x1 = dol0[k], dol1[k]
        dX = iqr(x0)
        if dX == 0: 
             dX = (max(x0) - min(x0))/2.0
             dX = 1.0 if dX==0 else dX

        dX = 1.0 if dX == 0 else dX
        drift = emd(x0, x1)/dX
        results[k].shift_magnitude = drift

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


from scipy.stats import percentileofscore
from metadata_utils import collate_fn_2 as collate_fn

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