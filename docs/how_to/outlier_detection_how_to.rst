.. _outlier_detection_how_to:

===============================================================
Initialize an Outlier Detector with Custom Prediction Arguments
===============================================================

*Add small blurb*

.. testsetup:: *

    import numpy as np
    dataset = np.ones((10,3,25,25), dtype=np.float32)

.. testcode::
    
    from daml.metrics.outlier_detection import OD_VAE, Threshold, ThresholdType

    # instantiate an outlier detector metric
    metric = OD_VAE()

    # update the metric's prediction args to make use of additional available compute
    metric.set_prediction_args(batch_size=128, outlier_perc=80)

    # fit and evaluate to detect outliers
    metric.fit_dataset(dataset)
    metric.evaluate(dataset)
