# channelstats

```{testsetup}
import numpy as np
from dataeval.metrics.stats import channelstats

images = np.repeat(np.arange(65536, dtype=np.int32), 4*10).reshape(10,-1,128,128)[:,:3,:,:]
for i in range(10):
    for j in range(3):
        images[i,j,30:50,50:80] = i*j
```

```{eval-rst}
.. autofunction:: dataeval.metrics.stats.channelstats
```
