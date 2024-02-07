=================================
Sufficiency Training How-To Guide
=================================

------------------------
Initializing Sufficiency
------------------------

Initialize a dataset, training function, eval function, and attach them to a sufficiency object.

.. code-block:: python

    train_ds, test_ds = load_dataset()
    model = Net()
    length = len(train_ds)

    # Instantiate sufficiency metric
    suff = Sufficiency()
    # Set predefined training and eval functions
    suff.set_training_func(custom_train)
    suff.set_eval_func(custom_eval)

-----------------------------------
Defining a Custom Training Function
-----------------------------------

Use a small step size and around 50 epochs per step on the curve.

.. code-block:: python
    
    def custom_train(model: nn.Module, X: torch.Tensor, y: torch.Tensor):
    """
    Passes data once through the model with backpropagation

    Parameters
    ----------
    model : nn.Module
        The trained model that will be evaluated
    X : torch.Tensor
        The training data to be passed through the model
    y : torch.Tensor
        The training labels corresponding to the data
    """
    # Defined only for this testing scenario
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    epochs = 50

    for _ in range(epochs):
        # Zero out gradients
        optimizer.zero_grad()
        # Forward Propagation
        outputs = model(X)
        # Back prop
        loss = criterion(outputs, y)
        loss.backward()
        # Update optimizer
        optimizer.step()

-----------------------------------
Parameters for Setup, Eval, and Run
-----------------------------------

We recommend at least 5 bootstrap samples (m_count) and 10 steps along the training curve per model (num_steps). 

.. code-block:: python
    
    # Create data indices for training
    m_count = 5
    num_steps = 10
    suff.setup(length, m_count, num_steps)
    # Train & test model
    output = suff.run(model, train_ds, test_ds)
