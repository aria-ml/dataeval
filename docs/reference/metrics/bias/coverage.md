# coverage

```{testsetup}
import numpy as np
import sklearn.datasets as dsets
from dataeval.metrics.bias import coverage

embeddings, _ = dsets.make_blobs(n_samples=500, centers=[(1,1), (3,3)], cluster_std=0.5, random_state=498)
```

```{eval-rst}
.. autofunction:: dataeval.metrics.bias.coverage
```
