# drift

```{eval-rst}
.. automodule:: dataeval.detectors.drift
   :synopsis:
```

```{currentmodule} dataeval.detectors.drift
```

## Detector Classes

```{eval-rst}
.. autosummary::

    DriftCVM
    DriftKS
    DriftMMD
    DriftUncertainty
```

## Supporting Classes

### Kernels

Kernels are used to map non-linear data to a higher dimensional space.

```{eval-rst}
.. autosummary::

    kernels.GaussianRBF
```

### Update Strategies

Update strategies inform how the drift detector classes update the reference data when monitoring for drift.

```{eval-rst}
.. autosummary::

    updates.LastSeenUpdate
    updates.ReservoirSamplingUpdate
```

## Output Classes

```{eval-rst}
.. autosummary::

    DriftOutput
    DriftMMDOutput
```

:::{toctree}
:hidden:
:titlesonly:
:maxdepth: 1

drift/driftcvm
drift/driftks
drift/driftmmd
drift/driftuncertainty
drift/kernels.guassianrbf
drift/updates.lastseenupdate
drift/updates.reservoirsamplingupdate
drift/driftoutput
drift/driftmmdoutput
:::
