(oodvae-ref)=
# Out-of-Distribution VAE

% Create small blurb here that answers:

% 1. What it is

% 2. What does it solve

## How-To Guides

Check out this **how to** to begin using the Out-of-Distribution Detection class

{doc}`Out-of-Distribution Detection Tutorial<../../how_to/notebooks/OODDetectionTutorial>`

## DataEval API

```{eval-rst}
.. autoclass:: dataeval.detectors.OOD_VAE
   :members:
   :inherited-members:
```

```{eval-rst}
.. testsetup:: *

    import numpy as np
    dataset = np.ones((10,3,25,25), dtype=np.float32)


```
```{eval-rst}
.. testcode::
    
    from dataeval.detectors import OOD_VAE
    from dataeval.models.tensorflow import VAE, create_model
    
    # instantiate an OOD detector metric
    metric = OOD_VAE(create_model(VAE, dataset[0].shape))

    # the training set has about 15% out-of-distribution so set the fit threshold at 85%
    metric.fit(dataset, threshold_perc=85, batch_size=128, verbose=False)
    
    # detect OOD at the 'feature' level
    metric.predict(dataset, ood_type="feature")

```
