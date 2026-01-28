# DataEval for Machine Learning Engineers

A machine learning (ML) engineer is focused on the practical application
of machine learning models:

- designing,
- building,
- training,
- testing, and
- deploying ML models.

While they are deeply involved in the model development process, they also
share some common ground with [data scientists](data_scientist.md) and
[test and evaluation engineers](te_engineer.md), as they are constantly
evaluating and iterating on their models and the data used to train them.

An ML engineer's workflow is highly iterative. They are in a continuous loop of
data preparation, model training, evaluation, and deployment. DataEval is a
powerful toolkit for the ML engineer, especially in the early stages of this loop,
where the quality of the data directly impacts the performance of the final model.

```{graphviz}
   
   digraph flowchart {
      node [shape=box,width=1.7,height=0.6]
      edge [arrowsize=0.6]
      layout="neato"

      1 [xref="{ref}`Scope And Objectives <scope-and-objectives>`",style="rounded,filled",pos="1.7,2.5!",fillcolor="#4151B0",fontcolor="white"]
      2 [xref="{ref}`Data Engineering <data-engineering>`",style="rounded,filled",pos="3.4,1.8!",fillcolor="#4151B0",fontcolor="white"]
      3 [xref="{ref}`Model Development<model-development>`",style="rounded,filled",pos="3.4,0.9!",fillcolor="#4151B0",fontcolor="white"]
      4 [xref="{ref}`Deployment <deployment>`",style="rounded,filled",pos="1.7,0.2!",color="#97979730",fillcolor="#97979730",fontcolor="gray"]
      5 [xref="{ref}`Monitoring <monitoring>`",style="rounded,filled",pos="0.0,0.9!",fillcolor="#4151B0",fontcolor="white"]
      6 [xref="{ref}`Analysis <analysis>`",style="rounded,filled",pos="0.0,1.8!",color="#97979730",fillcolor="#97979730",fontcolor="gray"]
      
      1:e->2:n
      2:s->3:n
      3:s->4:e [color="#97979730",fillcolor="#97979730"]
      4:w->5:s [color="#97979730",fillcolor="#97979730"]
      5:n->6:s [color="#97979730",fillcolor="#97979730"]
      6:n->1:w [color="#97979730",fillcolor="#97979730"]

      1:s->2:w [dir=both,style=dashed]
      1:s->3:w [dir=both,style=dashed]
      1:s->4:n [dir=both,style=dashed,color="#97979730",fillcolor="#97979730"]
      1:s->5:e [dir=both,style=dashed]
      2:w->5:e [dir=both,style=dashed]
      2:w->6:e [dir=both,style=dashed,color="#97979730",fillcolor="#97979730"]
      3:w->5:e [dir=both,style=dashed]
      3:w->6:e [dir=both,style=dashed,color="#97979730",fillcolor="#97979730"]
   }
```

## Key ML engineer tasks and relevant DataEval functions

The following sections highlight some ML engineer tasks along with the different DataEval tools that can
be leveraged in order to accomplish the task.

### Clean and preprocess data for training

Perform necessary data cleaning, normalization, resizing, and transformation
steps required by the model. Additionally, apply augmentation techniques during
training to improve model generalization.

Data can be cleaned with DataEval's {class}`.Outliers` class and {class}`.Duplicates`
class and be analyzed for biases and correlations with DataEval's {func}`.balance`,
{func}`.diversity`, and {func}`.parity` functions.

DataEval's {class}`.Dataset` class supports preprocessing/augmentation libraries such as
torchvision, albumentations, and others which perform the normalization,
resizing, and transformation steps.

### Determine problem feasibility

Analyze the dataset to determine if the cleaned dataset is an adequate dataset
given the problem requirements and complexity.

DataEval's {func}`.ber` and {func}`.uap` functions calculate the upper performance
bound given the specific dataset. It allows for comparison of different datasets
to determine the best dataset for the problem.

### Create dataset splits

Analyze the dataset to create a training, validation and testing subset. Ensure that
each split adequately represents the target operational environment and that there
are no correlations between the splits.

Datasets can be split using DataEval's {func}`.split_dataset`, which has options that
enable the user to split the data based on metadata. DataEval's bias functions,
{func}`.balance` and {func}`.diversity` can help identify when there may be spurious
correlations between the splits.

<!-- ### Build robust and scalable data ingestion pipelines

Create automated pipelines that can efficiently ingest images and their corresponding
labels from various sources (e.g., cloud storage buckets, on-premise servers),
handle different annotation formats (COCO, Pascal VOC, YOLO), and prepare them
for training.

DataEval's `Dataset` class is built following the [MAITE](https://mit-ll-ai-technology.github.io/maite/)
library protocols, which allow for versatile source formats and interoperability
with [PyTorch](https://pytorch.org/) datasets and dataloaders.

### Version datasets and models

Implement a strategy (e.g., using tools like DVC or conventions in cloud storage)
to version datasets so that any model training run can be traced back to the
exact data it was trained on.

DataEval supplies and supports Data and Model cards to allow for efficient
and traceable versioning of datasets and models. -->

### Build and evaluate models

Train standard models to establish a performance baseline against and then train
experimental and complex models to systematically evaluate model architectures.

While DataEval does not assist in the building and training of ML models, it
does contain {class}`.Sufficiency` which allows the user to compare model performance
of multiple models, including current model performance and predicted performance
at different amounts of data, along with the predicted model saturation point.

### Analyze and interpret model errors

Go beyond top-line metrics to perform detailed error analysis. Visualize the
false positives and false negatives to understand why the model is failing
(e.g., it confuses similar objects, fails on small objects, or struggles in low light).

By combining multiple DataEval functions -- {class}`.Select` class, {func}`.imagestats`,
{func}`.labelstats`, {func}`.cluster`, {func}`.balance`, and {func}`.diversity`
-- model failures can be investigated at the image level.

### Monitor model performance

Implement monitoring to track operational metrics (latency, throughput) and
to detect data drift. Analyze why a model's performance is decaying in production
by comparing the distribution of image statistics (or embeddings) between the
new data and the training data, then propose a retraining or calibration strategy.

DataEval has a set of [drift](../Drift.md)
and [out-of-distribution (OOD)](../OOD.md) detection functions, along with
{func}`.divergence` and {func}`.label_parity`, to identify differences between
operational and training distributions of both images and labels.
