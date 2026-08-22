# DataEval for Data Engineers

A data engineer is focused on the collection, manipulation and storage of data streams.
They are often involved across multiple stages of a project:

- identifying sources of data and validating them against protocol requirements,
- extracting, formatting, and normalizing data for downstream use,
- detecting data quality problems,
- configuring execution infrastructure for reliable large-scale evaluation, and
- monitoring operational pipelines for distribution shift.

While they share the goal of creating and maintaining excellent datasets with [data scientists](data_scientist.md),
their role is less research-oriented and more production-oriented — ensuring that
the data pipeline is reliable and that the data quality is built in at every stage
for trustworthy datasets.

A data engineer's workflow is centered around pipeline integrity and data readiness.
DataEval provides a set of tools that align with the production-oriented nature of this role,
helping the data engineer to validate, clean, and prepare datasets before they reach the
analysis and modeling stages.

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
      6 [xref="{ref}`Analysis <analysis>`",style="rounded,filled",pos="0.0,1.8!",color="#97979730",fillcolor="#97979730",fontcolor="gray"]

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

## Key data engineer tasks and relevant DataEval functions

The following sections highlight some data engineer tasks along with the different DataEval tools that can
be leveraged in order to accomplish the task.

### Validate incoming data against protocol requirements

Ensure that raw datasets — whether image collections, text embeddings, or tabular telemetry — conform
to the structural requirements of DataEval's evaluation tools before they reach downstream analysis.

Use `dataeval.protocols` to verify that datasets meet the interface contracts expected by drift
detection and out-of-distribution (OOD) engines. Use {func}`.compute_stats` with {class}`.ImageStats`
flags to run a fast linting pass over incoming images, catching schema violations, unexpected
channel counts, and dimension inconsistencies before they propagate through the pipeline.

### Detect data quality problems at ingestion time

Systematically check for corrupt or unreadable files, near-duplicate samples introduced by
repeated collection passes, and statistical outliers that may indicate sensor failures or
pipeline errors.

Use {class}`.Duplicates` to flag exact and near-duplicate images using perceptual hashing,
including rotation-invariant variants for datasets where images may be re-oriented during
ingestion. Use {class}`.Outliers` to identify samples with unusual brightness, blur, contrast,
or dimension properties — common signatures of sensor malfunctions or encoding errors.

### Audit label structure before training begins

Verify that annotation files are complete and consistent before passing a dataset to model
development. Missing annotations and empty image records are common pipeline artifacts that
are expensive to discover after a training run has started.

Use {func}`.label_stats` to count label distributions per class, identify images with no
annotations, and check for structural inconsistencies between image and label files. The
`empty_image_indices` output is the most immediately actionable field: images with no
annotations in an object detection dataset are almost always a pipeline error and should be
resolved before training.

### Normalize and preprocess data for evaluation pipelines

Define the filtering, padding, and normalization sequences that sanitize raw inputs before
they reach the higher-level analysis tools used by data scientists and ML engineers. Inconsistent
preprocessing across collection events is a common source of spurious drift signals.

Use `dataeval.utils.preprocessing` to apply consistent normalization across batches and
`dataeval.utils.arrays` to handle format conversions. Use {func}`.compute_stats` with
`channels={"r": 0, "g": 1, "b": 2}` — or whichever band groups your sensor produces — to
verify that band-level statistics are consistent before and after normalization passes.

### Configure hardware and execution settings for large-scale evaluation

Tune multi-process and multi-device execution contexts to ensure high-throughput evaluation
runs complete reliably in resource-constrained environments.

Use `dataeval.config` to control parallelism and memory allocation during large-scale
evaluation passes. Use the `h2_configure_hardware_settings` how-to for guidance on setting
process counts and device assignments that prevent memory exhaustion on the datasets typical
in T&E programs.

### Monitor data pipelines for distribution shift

Track whether new data arriving through an operational pipeline remains consistent with
the reference distribution used to qualify the model. Distribution shift introduced silently
at ingestion — by sensor changes, format conversions, or upstream pipeline updates — can
degrade model performance without any obvious error.

DataEval provides a set of [drift](../../concepts/DistributionShift.md#drift-detection)
detection functions to compare incoming data against a reference distribution. Use
{func}`.divergence` to quantify how far a new batch has moved from the reference, and
{func}`.label_parity` to check whether class frequency distributions have shifted between
collection periods.
