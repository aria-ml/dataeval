"""
Source code derived from NannyML 0.11.0
https://github.com/NannyML/nannyml/blob/main/nannyml/drift/multivariate/domain_classifier/calculator.py

Licensed under Apache Software License (Apache 2.0)
"""

import nannyml as nml 
from numpy.typing import ArrayLike
from typing import List, Optional, Tuple 
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

class MVDC():
    """ Multivariant Domain Classifier
    
        Parameters
        ----------
        x_trn : ArrayLike
            Training (reference) data with dim[n_samples, n_features].
        x_tst : ArrayLike
            Test (analysis) data with dim[n_samples, n_features].
        n_folds : Optional[int], optional
            Number of cross-validation (CV) folds. 
            The default is 5.
        chunk_sz : Optional[int], optional
            Number of samples in a chunk/batch used in CV, will get one metric & prediction per chunk/batch. 
            The default is None.
        threshold : Tuple[float,float], optional
            Upper, Lower metric bounds on roc_auc for identifying :term:`drift<Drift>`. 
            The default is (0.45, 0.65).
    """
    def __init__(
        self,
        x_trn: ArrayLike,
        x_tst: ArrayLike,
        n_folds: Optional[int] = 5,
        chunk_sz: Optional[int] = None,
        threshold: Tuple[float,float] = (0.45, 0.65),
    ) -> None: 

        self._x_trn = np.atleast_2d(x_trn)  # for 1D input, assume that is 1 sample: dim[1,n_features]
        self._x_tst = np.atleast_2d(x_tst)
        self._n_trn_samples = x_trn.shape[0]
        self._n_tst_samples = x_tst.shape[0]
        self._n_features()
        self._assign_col_labels()
        self.trndf = self._convert_to_df('trn')
        self.tstdf = self._convert_to_df('tst')
        self._n_folds = n_folds
        self._chunk_sz = chunk_sz
        self.thr = np.clip(threshold, 0, 1)
        self._thr = nml.thresholds.ConstantThreshold(lower=np.min(threshold), upper=np.max(threshold))

    def _n_features(self) -> int:
        """
        Return the number of features in the train/test sets.

        Raises
        ------
        ValueError
            Error raised when the number of features across train/test data differ.

        Returns
        -------
        int
            Returns the number of features.

        """
        if self._x_trn.shape[-1] != self._x_tst.shape[-1]:
            raise ValueError('Reference and test embeddings have different number of features')
        else:
            self.n_feats = self._x_trn.shape[-1]
    
    def _assign_col_labels(self) -> List[str]:
        """
        Create column labels based on the number of features in the train/test data

        Returns
        -------
        List[str]
            List of column numbers.

        """
        self._col_lbl =  [str(n) for n in range(self.n_feats)]
        return self._col_lbl 
    
    def _convert_to_df(self, mysplit: str) -> pd.DataFrame:
        """
        Convert :term:`NumPy` array into NML preferred dataframe

        Parameters
        ----------
        mysplit : str
            String delimiter used for converted train/test arrays to dataframes

        Raises
        ------
        ValueError
            Only accepts 'trn' or 'tst' identifiers, used internally.

        Returns
        -------
        mydf : pandas dataframe
            DESCRIPTION.

        """
        if mysplit == 'trn':
            mydf = pd.DataFrame(data=self._x_trn , 
                                index=np.arange(0, self._n_trn_samples),
                                columns=self._col_lbl)
            return mydf
        elif mysplit == 'tst':
            mydf = pd.DataFrame(data=self._x_tst , 
                                index=np.arange(0, self._n_tst_samples),
                                columns=self._col_lbl)
            return mydf
        else:
            raise ValueError('Unrecognized split assignment: {mysplit}')
            
    def fit(self):
        """
        Fit the domain classifier on the training dataframe

        Returns
        -------
        None.

        """
        self.calc = nml.DomainClassifierCalculator(feature_column_names = self._col_lbl,
                                                   cv_folds_num = self._n_folds,
                                                   chunk_size = self._chunk_sz,
                                                   threshold = self._thr)
        self.calc.fit(self.trndf)  
        
    def predict(self) -> pd.DataFrame:
        """
        Perform :term:`inference<Inference>` on the test dataframe

        Returns
        -------
        None.

        """
        results = self.calc.calculate(self.tstdf)
        self.resdf = results.to_df()
        
    def plot(self, showme=True, savedir=None):
        """
        Display the roc_auc metric over the train/test data in relation to the bounds.

        Parameters
        ----------
        showme : bool, optional
            If True, displays the figure. The default is True.
        savedir : str path, optional
            Directory of where to save the image. The default is None.

        Returns
        -------
        None.

        """
        fig, ax = plt.subplots(dpi=300)
        resdf = self.resdf
        xticks = np.arange(resdf.shape[0])
        trndf = resdf[resdf['chunk']['period']=='reference']
        tstdf = resdf[resdf['chunk']['period']=='analysis']
        # Get local indices for drift markers
        driftx = np.where(resdf['domain_classifier_auroc']['alert'].values)
        if np.size(driftx) > 2:
            ax.plot(resdf.index, resdf['domain_classifier_auroc']['upper_threshold'], 
                    'r--', label = 'thr_up')
            ax.plot(resdf.index, resdf['domain_classifier_auroc']['lower_threshold'], 
                    'r--', label = 'thr_low')
            ax.plot(trndf.index, trndf['domain_classifier_auroc']['value'], 
                    'b', label = 'train')
            ax.plot(tstdf.index, tstdf['domain_classifier_auroc']['value'], 
                    'g', label = 'test')  
            ax.plot(resdf.index.values[driftx], resdf['domain_classifier_auroc']['value'].values[driftx], 
                    'dm', markersize=3, label = 'drift')
            ax.set_xticks(xticks)
            ax.tick_params(axis='x', labelsize=6)
            ax.tick_params(axis='y', labelsize=6)
            ax.legend(loc='lower left', fontsize=6)
            ax.set_title('Domain Classifier, Drift Detection', fontsize=8)
            ax.set_ylabel('ROC AUC', fontsize=7)
            ax.set_xlabel('Chunk Index', fontsize=7)
            ax.set_ylim([0,1.1])
            self.ax = ax
            self.fig = fig
            if savedir is not None:
                fp = os.path.join(savedir,'DomainClassification.png')
                plt.savefig(fp)
            if showme: 
                plt.show()
            