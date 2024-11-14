# diversity

{term}`Diversity` and classwise diversity measure the evenness or uniformity of metadata
factors either over the entire dataset or by class. Diversity indices may
indicate which intrinsic or extrinsic metadata factors are sampled
disproportionately to others.

```{testsetup}
from dataeval.metrics.bias import diversity
class_labels = [0, 0, 0, 1, 0, 1, 0, 0, 0, 1]
str_vals = ["a", "a", "a", "a", "b", "a", "a", "a", "b", "b"]
cnt_vals = [0.63784, -0.86422, -0.1017, -1.95131, -0.08494, -1.02940, 0.07908, -0.31724, -1.45562, 1.03368]
metadata = {"var_cat": str_vals, "var_cnt": cnt_vals}
continuous_factor_bincounts = {"var_cnt": 5}
```

```{eval-rst}
.. autofunction:: dataeval.metrics.bias.diversity
```
