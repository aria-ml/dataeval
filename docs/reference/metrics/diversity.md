(diversity-ref)=
# Diversity

Diversity and classwise diversity measure the evenness or uniformity of metadata
factors either over the entire dataset or by class.  Diversity indices may
indicate which intrinsic or extrinsic metadata factors which are sampled
disproportionately to others.

```{testsetup}
import numpy as np
from dataeval.metrics import diversity, diversity_classwise
np.random.seed(7)
vals = ["a", "b"]
class_labels = [np.random.randint(low=0, high=2) for _ in range(20)]
metadata = [
    {
        "var_cat": vals[np.random.randint(low=0, high=len(vals))],
        "var_cnt": np.random.randn(),
        "var_float_cat": np.random.randint(low=0, high=len(vals)) + 0.1,
    }
    for idx in range(20)
]
```

## DataEval API

```{eval-rst}
.. autofunction:: dataeval.metrics.diversity
.. autofunction:: dataeval.metrics.diversity_classwise
```
