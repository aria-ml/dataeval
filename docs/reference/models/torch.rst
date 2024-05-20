.. _torch_models_ref:

==============
PyTorch Models
==============

DAML uses PyTorch as its main backend for metrics that require neural networks.
While these metrics can take in custom models, DAML provides utility classes 
to create a seamless integration between custom models and DAML's metrics.

---------
Tutorials
---------

Check out this tutorial to begin using the ``AETrainer`` class

:doc:`Autoencoder Trainer<../../tutorials/notebooks/AETrainerTutorial>`

-------------
How To Guides
-------------

There are currently no how to's for AETrainer. 
If there are scenarios that you want us to explain, contact us!

--------
DAML API
--------

Trainers
========

.. autoclass:: daml.models.torch.AETrainer
   :members:
   :inherited-members: