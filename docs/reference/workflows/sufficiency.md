(sufficiency-ref)=

# Sufficiency

```{testsetup}
from typing import Sequence
from unittest.mock import MagicMock, patch
from dataeval.workflows import Sufficiency

model = MagicMock()
train_ds = MagicMock()
train_ds.__len__.return_value = 2
test_ds = MagicMock()
test_ds.__len__.return_value = 2
train_fn = MagicMock()
eval_fn = MagicMock()
eval_fn.return_value = {"test": 1.0}
```


```{eval-rst}
.. autoclass:: dataeval.workflows.Sufficiency
   :members:
   :inherited-members:
```

