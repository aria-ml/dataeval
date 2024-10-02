(boxratiostats_ref)=
# boxratiostats

```{testsetup}
import numpy as np
from dataeval.metrics.stats import boxratiostats, dimensionstats

images = np.zeros((10,1,128,96))
images = [img for img in images]
images[6] = np.zeros((3,96,128))
images[9] = np.zeros((3,64,64))

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
.. autofunction:: dataeval.metrics.stats.boxratiostats
```
