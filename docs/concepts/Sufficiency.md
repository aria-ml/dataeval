# Sufficiency

## What is it

## When to use it

The `Sufficiency` class should be used when you would like to extrapolate hypothetical performance. For example, if you have a small dataset, and would like to know if it is worthwhile to collect more data.

## Theory behind it

## Tips and Tricks

### Defining a Custom Training Function

Use a small step size and around 50 epochs per step on the curve.

```python
def custom_train(model: nn.Module, dataset: Dataset, indices: Sequence[int]):
    # Defined only for this testing scenariov
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

```python
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
