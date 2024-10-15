(labelstats_ref)=
# labelstats

```{testsetup}
import numpy as np
from dataeval.metrics.stats import labelstats

np.random.seed(4)
label_array = np.random.choice(['horse', 'cow', 'sheep', 'pig', 'chicken'], 50)

labels = []
for i in range(10):
    num_labels = np.random.choice(5) + 1
    selected_labels = list(label_array[5*i:5*i+num_labels])
    labels.append(selected_labels)
```

```{eval-rst}
.. autofunction:: dataeval.metrics.stats.labelstats
```
