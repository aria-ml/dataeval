(sufficiency-ref)=

# Sufficiency

```{testsetup}
from typing import Sequence
import numpy as np
from unittest.mock import MagicMock, patch
from dataeval.workflows import Sufficiency

np.random.seed(0)

model = MagicMock()
train_ds = MagicMock()
train_ds.__len__.return_value = 100
test_ds = MagicMock()
test_ds.__len__.return_value = 10
train_fn = MagicMock()
eval_fn = MagicMock()
eval_fn.return_value = {"test": 1.0}
```


```{eval-rst}
.. autoclass:: dataeval.workflows.Sufficiency
   :members:
   :inherited-members:
```

