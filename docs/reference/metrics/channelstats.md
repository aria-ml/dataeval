(channelstats_ref)=
# Channel Statistics

```{testsetup}
import numpy as np
from dataeval.flags import ImageStat
from dataeval.metrics import channelstats

images = np.repeat(np.arange(65536, dtype=np.int32), 4*30).reshape(30,-1,128,128)[:,:3,:,:]
for i in range(30):
    for j in range(3):
        images[i,j,30:50,50:80] = i*j
```

```{eval-rst}
.. autofunction:: dataeval.metrics.channelstats
```