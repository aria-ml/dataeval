(dimensionstats_ref)=
# dimensionstats

```{testsetup}
import numpy as np
from dataeval.metrics.stats import dimensionstats

images = np.zeros((10,1,128,96))
images = [img for img in images]
images[6] = np.zeros((3,96,128))
images[9] = np.zeros((3,64,64))
```

```{eval-rst}
.. autofunction:: dataeval.metrics.stats.dimensionstats
```
