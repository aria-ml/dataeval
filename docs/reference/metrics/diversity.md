(diversity-ref)=
# Diversity

Diversity and classwise diversity measure the evenness or uniformity of metadata
factors either over the entire dataset or by class.  Diversity indices may
indicate which intrinsic or extrinsic metadata factors are sampled
disproportionately to others.

```{testsetup}
import numpy as np
from dataeval.metrics import diversity, diversity_classwise
class_labels = [0, 0, 0, 1, 0, 1, 0, 0, 0, 1]
str_vals = ["a", "a", "a", "a", "b", "a", "a", "a", "b", "b"]
cnt_vals = np.array(
    [0.63784, -0.86422, -0.1017, -1.95131, -0.08494, -1.02940, 0.07908, -0.31724, -1.45562, 1.03368]
)
metadata = [{"var_cat": sv, "var_cnt": cv} for sv, cv in zip(str_vals, cnt_vals)]
```


## DataEval API

```{eval-rst}
.. autofunction:: dataeval.metrics.diversity
.. autofunction:: dataeval.metrics.diversity_classwise
```
