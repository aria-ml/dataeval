# Sufficiency

```{testsetup}
from typing import Sequence
import numpy as np
from unittest.mock import MagicMock, patch
from dataeval.workflows import Sufficiency

model = MagicMock()
train_ds = MagicMock()
train_ds.__len__.return_value = 100
test_ds = MagicMock()
test_ds.__len__.return_value = 10
train_fn = MagicMock()
eval_fn = MagicMock()
eval_fn.return_value = {"test": 1.0}

mock_params = patch("dataeval.workflows.sufficiency.calc_params").start()
mock_params.return_value = np.array([0.0, 42.0, 0.0])
```

```{eval-rst}
.. autoclass:: dataeval.workflows.Sufficiency
   :members:
   :inherited-members:
```
