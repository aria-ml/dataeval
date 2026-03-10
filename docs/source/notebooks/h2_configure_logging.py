# ---
# jupyter:
#   jupytext:
#     default_lexer: ipython3
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: dataeval
#     language: python
#     name: python3
# ---

# %% [markdown]
# # How to configuring logging with DataEval

# %% [markdown]
# ## Problem statement
#
# DataEval uses Python's standard logging module to provide visibility into operations and debugging information. This
# guide demonstrates how to configure logging to display messages in the console or save them to disk when using DataEval
# functions.

# %% [markdown]
# ### When to use
#
# - You want to see detailed information about DataEval operations
# - You need to debug issues or understand internal processing
# - You want to save logs to a file for later analysis
# - You need different logging levels for different parts of your code

# %% [markdown]
# ### What you will need
#
# 1. A Python environment with dataeval installed
# 1. Basic understanding of Python's logging module

# %% [markdown]
# ## Getting started

# %%
# Google Colab Only
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval
except Exception:
    pass

# %%
import logging
import os

import sklearn.datasets as dsets

from dataeval.core._ber import ber_knn, ber_mst

# %% [markdown]
# ## Understanding logging levels
#
# Python's logging module supports several severity levels:
#
# - **DEBUG**: Detailed information, typically for diagnosing problems
# - **INFO**: Confirmation that things are working as expected
# - **WARNING**: An indication that something unexpected happened
# - **ERROR**: A serious problem that prevented a function from completing
# - **CRITICAL**: A very serious error
#
# DataEval primarily uses **DEBUG**, **INFO**, and **WARNING** levels for normal operation.

# %% [markdown]
# ## Logging to console
#
# This example demonstrates how to configure logging to display DataEval messages in the console.

# %% [markdown]
# ### Basic console logging (INFO level)

# %%
# Create console handler with formatting
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))

# %%
# Configure logging to show INFO level messages to console
dataeval_logger = logging.getLogger("dataeval")
dataeval_logger.setLevel(logging.INFO)
dataeval_logger.addHandler(console_handler)

# %%
# Create sample dataset
blobs = dsets.make_blobs(n_samples=100, centers=3, n_features=5, random_state=42)
embeddings, labels = blobs[0], blobs[1]

print("Running ber_knn with INFO logging:\n")
result = ber_knn(embeddings, labels, k=3)
print(f"\nResult: {result}")

# %% [markdown]
# ### Detailed console logging (DEBUG level)
#
# For more detailed information, you can enable DEBUG level logging:

# %% tags=["remove_cell"]
# Clear previous handlers
for handler in dataeval_logger.handlers[:]:
    dataeval_logger.removeHandler(handler)

# %%
# Configure logging to show DEBUG level messages to console
dataeval_logger = logging.getLogger("dataeval")
dataeval_logger.setLevel(logging.DEBUG)
dataeval_logger.addHandler(console_handler)

# %%
print("Running ber_mst with DEBUG logging:\n")
result = ber_mst(embeddings, labels)
print(f"\nResult: {result}")

# %% [markdown]
# ## Logging to disk
#
# This example demonstrates how to save DataEval logs to a file for later analysis.

# %% [markdown]
# ### Basic file logging
#
# Add the filename and filemode parameters to `logging.basicConfig`.

# %%
# Clear previous handlers
for handler in dataeval_logger.handlers[:]:
    dataeval_logger.removeHandler(handler)

# %%
# Configure logging to write to a file
log_file = "dataeval_operations.log"

# Create file handler with formatting
file_handler = logging.FileHandler(log_file, mode="w")  # 'w' to overwrite, 'a' to append
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

dataeval_logger = logging.getLogger("dataeval")
dataeval_logger.setLevel(logging.INFO)
dataeval_logger.addHandler(file_handler)

# %%
print(f"Running operations with logging to {log_file}...\n")

# Run multiple operations
result1 = ber_mst(embeddings, labels)
result2 = ber_knn(embeddings, labels, k=5)

print(f"ber_mst result: {result1}")
print(f"ber_knn result: {result2}")
print(f"\nLogs have been saved to '{log_file}'")

# Display the log file contents
if os.path.exists(log_file):
    print("\n--- Log File Contents ---")
    with open(log_file) as f:
        print(f.read())

# %% [markdown]
# ### Combined console and file logging
#
# You can log to both console and file simultaneously:

# %% tags=["remove_cell"]
# Clear previous handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# %%
# Create logger
logger = logging.getLogger("dataeval")
logger.setLevel(logging.DEBUG)

# Create file handler (DEBUG level)
log_file = "dataeval_detailed.log"
file_handler = logging.FileHandler(log_file, mode="w")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)

# Create console handler (INFO level only)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# %%
print("Running with dual logging (INFO to console, DEBUG to file):\n")
result = ber_knn(embeddings, labels, k=7)
print(f"\nResult: {result}")
print("\nNote: Console shows only INFO messages, but file contains DEBUG details too.")

# Display the log file contents
if os.path.exists(log_file):
    print("\n--- Log File Contents ---")
    with open(log_file) as f:
        print(f.read())

# %% [markdown]
# ## Temporarily disabling logs

# %%
# Disable all logging at CRITICAL level and below
logging.disable(logging.CRITICAL)

# %%
print("Running with logging disabled:\n")
result = ber_mst(embeddings, labels)
print(f"Result: {result}")
print("(No log messages should appear above)\n")

# %%
# Re-enable logging
logging.disable(logging.NOTSET)

# %%
print("Running with logging re-enabled:\n")
result = ber_mst(embeddings, labels)
print(f"Result: {result}")

# %% [markdown]
# ## Best practices
#
# 1. **Configure logging early**: Set up logging configuration at the start of your script or notebook
#
# 1. **Use file logging for production**: Console logging is great for development, but file logging is better for
#    production environments

# %% tags=["remove_cell"]
# Clean up log files created during the notebook execution
log_files = ["dataeval_operations.log", "dataeval_detailed.log"]
for log_file in log_files:
    if os.path.exists(log_file):
        os.remove(log_file)
        print(f"Removed {log_file}")

print("\nCleanup complete!")

# %% [markdown]
# ## See also
#
# ### How-to guides
#
# - [How to configure global hardware configuration defaults in DataEval](./h2_configure_hardware_settings.py)
