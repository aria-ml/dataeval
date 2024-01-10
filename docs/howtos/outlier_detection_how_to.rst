==============================
Outlier Detection How-To Guide
==============================

---------------------------------------------------------------
Initialize an Outlier Detector with Custom Prediction Arguments
---------------------------------------------------------------

.. code-block:: python
    
    from daml.metrics.outlier_detection import OD_VAEGMM

    # instantiate an outlier detector metric
    metric = OD_VAEGMM()

    # update the metric's prediction args to make use of additional available compute
    metric.set_prediction_args(batch_size=128, outlier_perc=80)
    metric.fit_dataset(dataset)
    metric.evaluate(dataset)
