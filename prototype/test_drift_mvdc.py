from domain_classifier import MVDC
import pytest
import numpy as np
import numpy.testing as npt
import os

@pytest.fixture
def tst_data():
    """Zeros as test data, just needs to be really different from the Gaussian training data"""
    n_samples, n_features = 100, 4
    tstData = np.zeros((n_samples, n_features))
    return tstData

@pytest.fixture
def trn_data():
    """Gaussian distribution, 0 mean, unit variance training data"""
    n_samples, n_features, mean, std_dev = 100, 4, 0, 1
    size = n_samples*n_features
    x = np.linspace(-3, 3, size)
    # Calculate the Gaussian distribution values
    trnData = (1 / (np.sqrt(2 * np.pi) * std_dev)) * np.exp(-0.5 * ((x - mean) / std_dev)**2)
    trnData = trnData.reshape((n_samples, n_features))
    return trnData
    

class TestMVDC:
    def test_init(self, trn_data, tst_data):
        """Test that the detector is instantiated correctly"""
        dc = MVDC(trn_data, tst_data, n_folds=5, chunk_sz=10, threshold=(0.6,0.9))
        assert dc._x_trn.shape == (100,4)
        assert dc._x_tst.shape == (100,4)
        assert dc._n_trn_samples == 100
        assert dc._n_tst_samples == 100
        assert dc.n_feats == 4
        assert dc._col_lbl == ['0', '1', '2', '3']
        assert dc.trndf.size == 400
        assert dc.tstdf.size == 400
        assert dc._n_folds == 5
        assert dc._chunk_sz == 10
        npt.assert_array_equal(dc.thr, (0.6,0.9))# threshold specific to this example data

    def test_sequence(self, trn_data, tst_data):
        """Sequential tests, each step is required before proceeding to the next"""
        dc = MVDC(trn_data, tst_data, n_folds=5, chunk_sz=10)
        dc.fit()
        assert dc.calc._is_fitted

        dc.predict()
        tstdf = dc.resdf[dc.resdf['chunk']['period']=='analysis']
        tst_auc_vals = tstdf['domain_classifier_auroc']['value'].values
        assert np.all(tst_auc_vals > dc.thr[-1])
        isdrift = tstdf['domain_classifier_auroc']['alert'].values
        assert np.all(isdrift)
        
        # Verify plot generates the figure and it saves correctly, then remove it
        dc.plot(showme=False, savedir=os.getcwd())
        x_data = dc.ax.lines[0].get_xdata()
        x_values = np.arange(0,20,dtype=int)
        npt.assert_array_equal(x_data, x_values)
        assert dc.fig._dpi == 300
        figfn = os.path.join(os.getcwd(),'DomainClassification.png')
        assert os.path.isfile(figfn)
        os.remove(figfn)
        
        
if __name__ == "__main__":
    
    # Demo code (uses more features than the pytest, but has the same result)
        
    # Data defined params
    n_samples, n_features = 100, 1024
    mean, std_dev = 0, 1
    
    # User defined params
    chunksz = 10
    bounds = (0.6,0.9)
    cvfold = 5
    
    # Create train/test sample data just for the demo
    size = n_samples*n_features
    x = np.linspace(-3, 3, size)
    trnData = (1 / (np.sqrt(2 * np.pi) * std_dev)) * np.exp(-0.5 * ((x - mean) / std_dev)**2)
    trnData = trnData.reshape((n_samples, n_features))   
    tstData = np.zeros((n_samples, n_features))
    
    # Domain classifier training (fit) and inference (predict)
    dc = MVDC(trnData, tstData, n_folds=cvfold, chunk_sz=chunksz, threshold=bounds)
    dc.fit()
    dc.predict()
    dc.plot(showme=True, savedir=os.getcwd())  # fig: DomainClassification.png will be to cwd
    
    # Test domain dataframe and classification
    tstdf = dc.resdf[dc.resdf['chunk']['period']=='analysis']
    isdrift = tstdf['domain_classifier_auroc']['alert'].values
    

