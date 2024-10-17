# divergence

```{testsetup}
import numpy as np
import sklearn.datasets as dsets
from dataeval.metrics.estimators import divergence

datasetA, _ = dsets.make_blobs(n_samples=50, centers=[(-1,-1), (1,1)], cluster_std=0.3, random_state=712)
datasetB, _ = dsets.make_blobs(n_samples=50, centers=[(-0.5,-0.5), (1,1)], cluster_std=0.3, random_state=712)
```

```{eval-rst}
.. autofunction:: dataeval.metrics.estimators.divergence
```
