(torch-models-ref)=

# PyTorch Models

DataEval uses PyTorch as its main backend for metrics that require neural networks.
While these metrics can take in custom models, DataEval provides utility classes
to create a seamless integration between custom models and DataEval's metrics.

## How-To Guides

Check out this **how to** to begin using the `AETrainer` class

{doc}`Autoencoder Trainer<../../how_to/notebooks/AETrainerTutorial>`

## DataEval API

### Trainers

```{eval-rst}
.. autoclass:: dataeval.models.torch.AETrainer
   :members:
   :inherited-members:
```
