# balance

{term}`Balance` and classwise balance are metrics that measure distributional
correlation between metadata factors and class label.  Balance and classwise
balance can indicate opportunities for shortcut learning and disproportionate
dataset sampling with respect to class labels or between metadata factors.

```{testsetup}
from dataeval.metrics.bias import balance
from dataeval.metrics.bias.metadata_preprocessing import metadata_preprocessing

str_vals = ["b", "b", "b", "b", "b", "a", "a", "b", "a", "b", "b", "a"]
class_labels = [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0]
cnt_vals = [-0.54, -0.32, 0.41, 1.04, -0.13, 1.37, -0.67, 0.35, 0.90, 0.09, -0.74, -0.92]
cat_vals = [1.1, 1.1, 0, 0, 1.1, 0, 1.1, 0, 0, 1.1, 1.1, 0]
metadata_dict = [{"var_cat": str_vals, "var_cnt": cnt_vals, "var_float_cat": cat_vals}]
continuous_factor_bincounts = {"var_cnt": 5, "var_float_cat": 2}
metadata = metadata_preprocessing(metadata_dict, class_labels, continuous_factor_bincounts)
```

```{eval-rst}
.. autofunction:: dataeval.metrics.bias.balance
```
