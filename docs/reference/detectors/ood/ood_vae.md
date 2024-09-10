(oodvae-ref)=
# OOD_VAE

```{testsetup}

import numpy as np
from dataeval.detectors.ood import OOD_VAE
from dataeval.tensorflow.models import VAE, create_model

dataset = np.ones((10,3,25,25), dtype=np.float32)
```

```{eval-rst}
.. autoclass:: dataeval.detectors.ood.OOD_VAE
   :members:
   :inherited-members:
```