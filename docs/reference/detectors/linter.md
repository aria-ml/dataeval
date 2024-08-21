(linter-ref)=
# Linter

```{testsetup}
import numpy as np
from dataeval.flags import ImageVisuals
from dataeval.detectors import Linter

dataset = np.ones((30,1,128,128), dtype=np.int32)*2 + np.repeat(np.arange(10), 3*128*128).reshape(30,-1,128,128)
dataset[[7,11,18,25]] *= 25
```

```{eval-rst}
.. autoclass:: dataeval.detectors.Linter
   :members:
   :inherited-members:
```
