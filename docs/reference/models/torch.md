(torch-models-ref)=

# PyTorch Models

DataEval uses PyTorch as its main backend for metrics that require neural networks.
While these metrics can take in custom models, DataEval provides utility classes
to create a seamless integration between custom models and DataEval's metrics.

## Tutorials

Check out this tutorial to begin using the `AETrainer` class

{doc}`Autoencoder Trainer<../../tutorials/notebooks/AETrainerTutorial>`

## How To Guides

There are currently no how to's for AETrainer.
If there are scenarios that you want us to explain, contact us!

## DataEval API

### Trainers

```{eval-rst}
.. autoclass:: dataeval.models.torch.AETrainer
   :members:
   :inherited-members:
```
