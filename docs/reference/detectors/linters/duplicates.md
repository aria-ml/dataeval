# Duplicates

```{testsetup}
import numpy as np
from dataeval.detectors.linters import Duplicates
from dataeval.metrics.stats import hashstats

rng = np.random.default_rng(273)
base = np.concatenate([np.ones((5,10)), np.zeros((5,10))])
images = np.stack([rng.permutation(base)*i for i in range(50)], axis=0)
images[16] = images[37]
images[3] = images[20]

hashes1 = hashstats(images[:24])
hashes2 = hashstats(images[25:])
```

```{eval-rst}
.. autoclass:: dataeval.detectors.linters.Duplicates
   :members:
   :inherited-members:
```
