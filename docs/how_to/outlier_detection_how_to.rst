.. _outlier_detection_how_to:

===============================================================
Initialize an Outlier Detector with Custom Prediction Arguments
===============================================================

*Add small blurb*

.. testsetup:: *

    import numpy as np
    dataset = np.ones((10,3,25,25), dtype=np.float32)

.. testcode::
    
    from daml.metrics.outlier import VAEOutlier
    from daml.models.tensorflow import VAE, create_model
    
    # instantiate an outlier detector metric
    metric = VAEOutlier(create_model(VAE, dataset[0].shape))

    # the training set has about 15% outliers so set the fit threshold at 85%
    metric.fit(dataset, threshold_perc=85, batch_size=128, verbose=False)
    
    # detect outliers at the 'feature' level
    metric.predict(dataset, outlier_type="feature")
