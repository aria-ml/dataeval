.. _ood_detection_how_to:

===========================================================================
Initialize an Out-of-Distribution Detector with Custom Prediction Arguments
===========================================================================

*Add small blurb*

.. testsetup:: *

    import numpy as np
    dataset = np.ones((10,3,25,25), dtype=np.float32)

.. testcode::
    
    from daml.detectors import OOD_VAE
    from daml.models.tensorflow import VAE, create_model
    
    # instantiate an OOD detector metric
    metric = OOD_VAE(create_model(VAE, dataset[0].shape))

    # the training set has about 15% out-of-distribution so set the fit threshold at 85%
    metric.fit(dataset, threshold_perc=85, batch_size=128, verbose=False)
    
    # detect OOD at the 'feature' level
    metric.predict(dataset, ood_type="feature")
