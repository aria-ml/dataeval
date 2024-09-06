(balance-ref)=
# Balance

Balance and classwise balance are metrics that measure distributional
correlation between metadata factors and class label.  Balance and classwise
balance can indicate opportunities for shortcut learning and disproportionate
dataset sampling with respect to class labels or between metadata factors.


```{testsetup}
import numpy as np
from dataeval.metrics import balance, balance_classwise
str_vals = ["b", "b", "b", "b", "b", "a", "a", "b", "a", "b", "b", "a"]
class_labels = [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0]
cnt_vals = np.array(
    [
        -0.54425898,
        -0.31630016,
        0.41163054,
        1.04251337,
        -0.12853466,
        1.36646347,
        -0.66519467,
        0.35151007,
        0.90347018,
        0.0940123,
        -0.74349925,
        -0.92172538,
    ]
)
cat_vals = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0])
metadata = [
    {
        "var_cat": strv,
        "var_cnt": cntv,
        "var_float_cat": catv + 0.1,
    }
    for strv, cntv, catv in zip(str_vals, cnt_vals, cat_vals)
]
```


## DataEval API

```{eval-rst}
.. autofunction:: dataeval.metrics.balance
.. autofunction:: dataeval.metrics.balance_classwise
```
