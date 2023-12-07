==============================
Outlier Detection How-To Guide
==============================

---------------------------------------------------------------
Initialize an Outlier Detector with a Custom, Pre-Trained Model
---------------------------------------------------------------

.. code-block:: python

    # instantiate an outlier detector metric
    metric = VAE()

    # update the metric's model with a new custom, pre-trained version
    metric.set_model(encoder_net=pretrained_encoder, decoder_net=pretrained_decoder)
    metric.initialize_detector()

---------------------------------------------------------------
Initialize an Outlier Detector with Custom Prediction Arguments
---------------------------------------------------------------

.. code-block:: python
    
    # instantiate an outlier detector metric
    metric = VAEGMM()

    # update the metric's prediction args to make use of additional available compute
    metric.set_prediction_args(batch_size=128, outlier_perc=80)
    metric.initialize_detector()