# Sufficiency

## What is it

## When to use it

The {term}`sufficiency<Sufficiency>` class should be used when you would like
to extrapolate hypothetical performance. For example, if you have a small
dataset, and would like to know if it is worthwhile to collect more data.

## Theory behind it

## Tips and Tricks

### Defining a Custom Training Function

Use a small step size and around 10 epochs per step on the curve.

```python

class CustomTrainingStrategy:
    def __init__(self, batch_size=16, epochs=10, device="cpu", learning_rate=0.01, momentum=0.9):
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.learning_rate = learning_rate
        self.momentum=momentum

    def train(model: nn.Module, dataset: Dataset, indices: Sequence[int]):
        """Trains a model using CrossEntropy"""

        device = self.device
        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum)

        # Define the dataloader for training
        dataloader = DataLoader(Subset(dataset, indices), batch_size=self.batch_size)

        for epoch in range(self.epochs):
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

We recommend at least 5 bootstrap samples (runs) and 10 steps along the
training curve per model (substeps).

```python

train_strategy = CustomTrainingStrategy()
eval_strategy = CustomEvaluationStrategy()

custom_config = Sufficiency.Config(
    training_strategy=train_strategy,
    evaluation_strategy=eval_strategy,
    runs=5,
    substeps=10,
)

suff = Sufficiency(
    model=model,
    train_ds=train_ds,
    test_ds=test_ds,
    config=custom_config,
    )

# Train & test model
output = suff.evaluate()
```
