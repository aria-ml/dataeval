(oodvae-ref)=
# Out-of-Distribution VAE

```{testsetup}

import numpy as np
from dataeval.detectors import OOD_VAE
from dataeval.models.tensorflow import VAE, create_model

dataset = np.ones((10,3,25,25), dtype=np.float32)
```

```{eval-rst}
.. autoclass:: dataeval.detectors.OOD_VAE
   :members:
   :inherited-members:
```