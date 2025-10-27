# DataEval for Testing and Evaluation Engineers

An AI test and evaluation (T&E) engineer is responsible for the independent and
rigorous testing of AI systems:

- to verify that the AI systems meet requirements,
- to validate that the AI systems are suitable for their intended use, and
- to identify any potential risks or limitations before deployment.

T&E engineers typically operate as a third party, separate from the
[ML engineers](ml_engineer.md) and [data scientists](data_scientist.md) to
provide an unbiased assessment of the AI system's performance and safety.

A T&E engineer's workflow is often centered around formal testing events where
they execute a test plan, analyze the results, and generate a final report with
findings and recommendations. DataEval provides a suite of tools that are critical
for many of these activities, especially those related to ensuring the quality
and operational relevance of the data used for testing.

```{graphviz}
   
   digraph flowchart {
      node [shape=box,width=1.7,height=0.6]
      edge [arrowsize=0.6]
      layout="neato"

      1 [xref="{ref}`Scope And Objectives <scope-and-objectives>`",style="rounded,filled",pos="1.7,2.5!",fillcolor="#4151B0",fontcolor="white"]
      2 [xref="{ref}`Data Engineering <data-engineering>`",style="rounded,filled",pos="3.4,1.8!",fillcolor="#4151B0",fontcolor="white"]
      3 [xref="{ref}`Model Development<model-development>`",style="rounded,filled",pos="3.4,0.9!",color="#97979730",fillcolor="#97979730",fontcolor="gray"]
      4 [xref="{ref}`Deployment <deployment>`",style="rounded,filled",pos="1.7,0.2!",fillcolor="#4151B0",fontcolor="white"]
      5 [xref="{ref}`Monitoring <monitoring>`",style="rounded,filled",pos="0.0,0.9!",fillcolor="#4151B0",fontcolor="white"]
      6 [xref="{ref}`Analysis <analysis>`",style="rounded,filled",pos="0.0,1.8!",fillcolor="#4151B0",fontcolor="white"]
      
      1:e->2:n
      2:s->3:n [color="#97979730",fillcolor="#97979730"]
      3:s->4:e [color="#97979730",fillcolor="#97979730"]
      4:w->5:s
      5:n->6:s
      6:n->1:w

      1:s->2:w [dir=both,style=dashed]
      1:s->3:w [dir=both,style=dashed,color="#97979730",fillcolor="#97979730"]
      1:s->4:n [dir=both,style=dashed]
      1:s->5:e [dir=both,style=dashed]
      2:w->5:e [dir=both,style=dashed]
      2:w->6:e [dir=both,style=dashed]
      3:w->5:e [dir=both,style=dashed,color="#97979730",fillcolor="#97979730"]
      3:w->6:e [dir=both,style=dashed,color="#97979730",fillcolor="#97979730"]
   }
```

## Key T&E engineer tasks and relevant DataEval functions

The following sections highlight some T&E engineer tasks along with the different DataEval tools that can
be leveraged in order to accomplish the task.

### Ensure test data quality and annotation integrity

Perform a thorough analysis of the test data to identify and flag quality issues,
such as blurry or corrupt images, and annotation errors like misaligned bounding
boxes, incorrect class labels, or inconsistent labeling standards.

Use DataEval's {func}`.labelstats` function to provide the necessary label
distributions and counts as well as DataEval's {func}`.imagestats` function to
identify any loading or annotation errors. DataEval also includes a
{class}`.Outliers` class and a {class}`.Duplicates` class to identify anomaly
and redundant images.

### Validate test data is operationally relevant

Scrutinize the test dataset to ensure it contains images that accurately
represent the target operational conditions, including sensor types, camera
angles, weather, lighting, and environments.

For datasets that contain acquisition conditions as metadata, DataEval has
a set of bias metrics, {func}`.balance` and {func}`.diversity`, that can assist
in determining relevant conditions. It also contains a {func}`.completeness` and
{func}`.coverage` metric to determine the representativeness of the dataset.

### Evaluate performance on critical data subgroups

Measure and compare model performance on specific, operationally relevant subgroups
of the image data (e.g., performance on small objects, low-light images,
rainy conditions, or partially occluded targets).

DataEval's {class}`.Sufficiency` class allows the user to compare model performance
of multiple models, including current model performance and predicted performance
at different amounts of data, along with the predicted model saturation point.
DataEval also has a {class}`.Select` class that allows the user to create subsets
of the data based on a user defined selection.

### Perform error analysis to identify systemic weaknesses

Conduct a deep dive into the model's failures (false positives, false negatives,
misclassifications). Visualize these errors to identify patterns, such as the
model consistently confusing two similar-looking objects or failing to detect
objects at a distance.

By combining multiple DataEval functions -- {class}`.Select` class, {func}`.imagestats`,
{func}`.labelstats`, {func}`.cluster`, {func}`.balance`, and {func}`.diversity`
-- model failures can be investigated at the image level.

### Explore unknown risks and potential failure modes

Proactively search for unexpected failure modes by testing the system against
visual edge cases, anomalies, and adversarial attacks (e.g., adversarial patches
that can make an object invisible to the detector).

While DataEval does not address [adversarial robustness](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
or [natural robustness](https://github.com/Kitware/nrtk), it does contain a
{class}`.linter` class to identify visual anomalies and a {func}`.cluster`
function which can help identify edge cases.

### Monitor model performance

Implement monitoring to track operational metrics (latency, throughput) and
to detect data drift. Analyze why a model's performance is decaying in production
by comparing the distribution of image statistics (or embeddings) between the
new data and the training data, then propose a retraining or calibration strategy.

DataEval has a set of [drift](../Drift.md)
and [out-of-distribution (OOD)](../OOD.md) detection functions, along with
{func}`.divergence` and {func}`.label_parity`, to identify differences between
operational and training distributions of both images and labels.

<!--### Develop summary reports and provide recommendations

Synthesize all test findings into a clear report that details the vision system's
capabilities, limitations (e.g., "fails to detect pedestrians in heavy rain"),
and risks. Provide an explicit recommendation to decision-makers regarding deployment
readiness, often with recommended operational constraints.-->