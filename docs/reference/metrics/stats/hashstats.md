# hashstats

```{testsetup}
import numpy as np
from dataeval.metrics.stats import hashstats

images = np.repeat(np.arange(16384, dtype=np.int32), 10).reshape(4,-1,64,64)
```

```{eval-rst}
.. autofunction:: dataeval.metrics.stats.hashstats
```
