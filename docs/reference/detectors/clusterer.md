(clusterer-ref)=
# Clusterer

```{testsetup}
import sklearn.datasets as dsets
from dataeval.detectors import Clusterer

dataset, _ = dsets.make_blobs(n_samples=50, centers=[(-1, -1), (1, 1)], cluster_std=0.5, random_state=33)
dataset[9] = dataset[24]
dataset[23] = dataset[48] + 1e-5
```

```{eval-rst}
.. autoclass:: dataeval.detectors.Clusterer
   :members:
   :inherited-members:
```
