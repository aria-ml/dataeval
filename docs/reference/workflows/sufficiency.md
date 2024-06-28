(sufficiency-ref)=

# Sufficiency

% Create small blurb here that answers:

% 1. What it is

% 2. What does it solve

## Tutorials

Check out this tutorial to begin using the `Sufficiency` class

{doc}`Sufficiency Tutorial<../../tutorials/notebooks/ClassLearningCurvesTutorial>`

## How To Guides

There are currently no how to's for `Sufficiency`.
If there are scenarios that you want us to explain, contact us!

## DAML API

```{eval-rst}
.. autoclass:: daml.workflows.Sufficiency
   :members:
   :inherited-members:
```

### Initializing Sufficiency

```{eval-rst}
.. testsetup:: *

    from typing import Sequence

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset
    from unittest.mock import MagicMock, patch

    from daml.workflows import Sufficiency
    model = MagicMock()
    train_ds = MagicMock()
    train_ds.__len__.return_value = 2
    test_ds = MagicMock()
    test_ds.__len__.return_value = 2
    train_fn = MagicMock()
    eval_fn = MagicMock()
    eval_fn.return_value = {"test": 1.0}
    device = "cpu"
```

### Defining a Custom Training Function

Use a small step size and around 50 epochs per step on the curve.

```{eval-rst}
.. testcode::

    def custom_train(model: nn.Module, dataset: Dataset, indices: Sequence[int]):
        # Defined only for this testing scenario
        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        epochs = 10

        # Define the dataloader for training
        dataloader = DataLoader(Subset(dataset, indices), batch_size=16)

        for epoch in range(epochs):
            for batch in dataloader:
                # Load data/images to device
                X = torch.Tensor(batch[0]).to(device)
                # Load targets/labels to device
                y = torch.Tensor(batch[1]).to(device)
                # Zero out gradients
                optimizer.zero_grad()
                # Forward propagation
                outputs = model(X)
                # Compute loss
                loss = criterion(outputs, y)
                # Back prop
                loss.backward()
                # Update weights/parameters
                optimizer.step()
```

### Recommended parameters for Sufficiency

We recommend at least 5 bootstrap samples (runs) and 10 steps along the training curve per model (substeps). 

```{eval-rst}
.. testcode::
    
    # Create data indices for training
    suff = Sufficiency(
        model=model,
        train_ds=train_ds,
        test_ds=test_ds,
        train_fn=train_fn,
        eval_fn=eval_fn,
        runs=5,
        substeps=10)

    # Train & test model
    output = suff.evaluate()
```
