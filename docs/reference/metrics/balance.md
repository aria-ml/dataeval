(balance-ref)=
# Balance

Balance and classwise balance are metrics that measure distributional
correlation between metadata factors and class label.  Balance and classwise
balance can indicate opportunities for shortcut learning and disproportionate
dataset sampling with respect to class labels or between metadata factors.

```{testsetup}
import numpy as np
from dataeval.metrics import balance, balance_classwise
np.random.seed(7)
def get_index_one_class(class_label, corr_val):
    # correlate this metadata with class label
    if class_label == 0 and np.random.rand() < corr_val:
        return class_label
    else:
        return np.random.randint(low=0, high=2)
vals = ["a", "b"]
corr_val = 0.75
num_samples = 20
class_labels = [np.random.randint(low=0, high=2) for _ in range(num_samples)]
metadata = [
    {
        "var_cat": vals[get_index_one_class(class_labels[idx], corr_val)],
        "var_cnt": np.random.randn(),
        "var_float_cat": np.random.randint(low=0, high=len(vals)) + 0.1,
    }
    for idx in range(num_samples)
]
```

## DataEval API

```{eval-rst}
.. autofunction:: dataeval.metrics.balance
.. autofunction:: dataeval.metrics.balance_classwise
```
