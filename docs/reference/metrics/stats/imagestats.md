(imagestats_ref)=
# imagestats

```{testsetup}
import numpy as np
from dataeval.flags import ImageStat
from dataeval.metrics.stats import imagestats

images = np.repeat(np.arange(65536, dtype=np.int32), 10).reshape(40,-1,128,128)
for i in range(40):
    for j in range(2,128,4):
        images[i,0,j,i:j] = i*j if i % 4 !=0 else 0
```

```{eval-rst}
.. autofunction:: dataeval.metrics.stats.imagestats
```
