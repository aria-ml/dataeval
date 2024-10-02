(datasetstats_ref)=
# datasetstats

```{testsetup}
import numpy as np
from dataeval.metrics.stats import datasetstats

images = np.repeat(np.arange(65536, dtype=np.int32), 4*30).reshape(30,-1,128,128)[:,:3,:,:]
for i in range(30):
    for j in range(3):
        images[i,j,30:50,50:80] = i*j

bboxes = [
    np.array([[ 5, 21, 24, 43], [ 7,  4, 17, 21]]),
    np.array([[12, 23, 28, 24]]),
    np.array([[13,  9, 29, 23], [17,  7, 39, 20], [ 2, 14,  9, 26]]),
    np.array([[18, 14, 28, 29]]),
    np.array([[21, 18, 44, 27], [15, 13, 28, 23]]),
    np.array([[13,  2, 23, 14]]),
    np.array([[ 4, 16,  8, 20], [16, 14, 25, 29]]),
    np.array([[ 1, 22, 13, 45], [12, 20, 27, 21], [16, 22, 39, 28]]),
    np.array([[16,  5, 30, 13]]),
    np.array([[ 2, 18, 11, 30], [ 9, 22, 23, 42]])
]
```

```{eval-rst}
.. autofunction:: dataeval.metrics.stats.datasetstats
```
