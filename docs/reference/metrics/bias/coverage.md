(coverage-ref)=
# coverage

```{testsetup}
import numpy as np
import sklearn.datasets as dsets
from dataeval.metrics.bias import coverage

embeddings, _ = dsets.make_blobs(n_samples=50, centers=[(1,1)], cluster_std=0.5, random_state=498)
```

```{eval-rst}
.. autofunction:: dataeval.metrics.bias.coverage
```
