# DataEval for Data Scientists

A data scientist is focused on exploration and analysis to extract actionable insights
from data. While they share the goal of building effective models with [ML engineers](ml_engineer.md),
their role is often more exploratory and research-oriented. They are often involved
in the early stages of a project:

- defining the problem,
- understanding the data's potential and limitations, and
- experimenting with various modeling approaches.

They are deeply involved in understanding the data, formulating hypotheses, and
using statistical methods to test those hypotheses. For a data scientist, data
is not just an input to a model; it is the object of study itself.

A data scientist's workflow is centered around data preparation.
DataEval provides a rich set of tools that align perfectly with the exploratory
nature of this role, helping the data scientist to quickly understand, clean,
and prepare data for modeling.

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
      6 [xref="{ref}`Analysis <analysis>`",style="rounded,filled",pos="0.0,1.8!",fillcolor="#4151B0",fontcolor="white"]
      
      1:e->2:n
      2:s->3:n
      3:s->4:e [color="#97979730",fillcolor="#97979730"]
      4:w->5:s [color="#97979730",fillcolor="#97979730"]
      5:n->6:s
      6:n->1:w

      1:s->2:w [dir=both,style=dashed]
      1:s->3:w [dir=both,style=dashed]
      1:s->4:n [dir=both,style=dashed,color="#97979730",fillcolor="#97979730"]
      1:s->5:e [dir=both,style=dashed]
      2:w->5:e [dir=both,style=dashed]
      2:w->6:e [dir=both,style=dashed]
      3:w->5:e [dir=both,style=dashed]
      3:w->6:e [dir=both,style=dashed]
   }
```

## Key data scientist tasks and relevant DataEval functions

The following sections highlight some data scientist tasks along with the different DataEval tools that can
be leveraged in order to accomplish the task.

### Perform initial data profiling

Compute statistics on image properties like brightness, contrast, sharpness,
and color distributions. For object detection, analyze the distributions of
bounding box sizes, aspect ratios, and locations.

Use DataEval's {func}`.imagestats` to provide the necessary image statistics
on both the image and any bounding boxes.

### Identify data quality issues

Systematically scan for problems like corrupt or unreadable image files,
incorrect or missing labels, inconsistent annotation formats
(e.g., COCO vs. YOLO), and misaligned bounding boxes.

Use DataEval's {func}`.labelstats` to provide the necessary label distributions
and counts as well as DataEval's {class}`.Dataset` class to identify any loading or annotation
errors. DataEval also includes a {class}`.Outliers` class and a {class}`.Duplicates`
class to identify anomaly and redundant images.

### Discover underlying data structures and patterns

Use visualization techniques to review random samples of images.
Apply clustering on image embeddings (e.g., from a pre-trained model) to discover
natural groupings of scenes or objects that may not be captured by the labels.

Use DataEval's {func}`.cluster` to group the images.

### Perform statistical tests on image properties

Apply formal statistical tests to validate hypotheses about differences in image
characteristics between data subsets (e.g., comparing the average bounding box
in 'day' vs. 'night' images).

Use DataEval's {class}`.Select` class to create different subsets of the dataset
that can then be compared using the results of DataEval's {func}`.imagestats`
function.

### Quantify bias and representativeness

Use quantitative metrics to measure image metadata like class balance, background
diversity, lighting conditions, and camera angles for potential biases, and
dataset coverage of the operational domain.

DataEval has a set of bias metrics -- {func}`.balance`, {func}`.diversity`, and
{func}`.parity` -- to identify potential shortcuts based on the metadata. It
also contains {func}`.completeness` and {func}`.coverage` to determine the
representativeness of the dataset.

### Determine problem feasibility

Analyze the dataset to determine if the cleaned dataset is an adequate dataset
given the problem requirements and complexity.

DataEval's {func}`.ber` and {func}`.uap` functions calculate the upper performance
bound given the specific dataset. It allows for comparison of different datasets
to determine the best dataset for the problem.

### Create dataset splits

Analyze the dataset to create a training, validation and testing subset. Ensure that
each split adequately represents the target operational environment and that there
is no correlations between the splits.

Datasets can be split using DataEval's {func}`.split_dataset`, which has options that
enable to user to split the data based on metadata. DataEval's bias functions,
{func}`.balance` and {func}`.diversity` can help identify when there my be spurious
correlations between the splits.

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
-- false positives and false negatives can be further analyzed.

### Monitor model performance

Implement monitoring to track operational metrics (latency, throughput) and
to detect data drift. Analyze why a model's performance is decaying in production
by comparing the distribution of image statistics (or embeddings) between the
new data and the training data, then propose a retraining or calibration strategy.

DataEval has a set of [drift](../Drift.md)
and [out-of-distribution (OOD)](../OOD.md) detection functions, along with
{func}`.divergence` and {func}`.label_parity`, to identify differences between operational
and training distributions of both images and labels.
