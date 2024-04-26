.. _aetrainer_ref:

===================
Autoencoder Trainer
===================

Autoencoders (AEs) are a type of neural network architecture that contain two parts: an encoder and decoder.
While there are many uses of AEs, DAML uses them for dimensionality reduction on datasets with large images.

**How does it work?**

The encoder is trained to create dense embeddings for the images while the decoder is trained
to reconstruct the new embedding into the original input image. This allows the dense embedding to 
become an efficient downsampling of the images, allowing for faster model inference and metric computation.

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

.. autoclass:: daml.models.ae.AETrainer
   :members:
   :inherited-members: