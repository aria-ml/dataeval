(outliers-ref)=
# Outliers

```{testsetup}
import numpy as np
from dataeval.detectors.linters import Outliers
from dataeval.metrics.stats import pixelstats

images = np.ones((30,1,128,128), dtype=np.int32)*2 + np.repeat(np.arange(10), 3*128*128).reshape(30,-1,128,128)
images[10:13,:,50:80,50:80] = 0
images[[7,11,18,25]] = 512

stats1 = pixelstats(images[:14])
stats2 = pixelstats(images[15:])
```

```{eval-rst}
.. autoclass:: dataeval.detectors.linters.Outliers
   :members:
   :inherited-members:
```
