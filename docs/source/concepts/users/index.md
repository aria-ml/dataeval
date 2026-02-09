# Supported Roles

DataEval is a versatile toolkit that provides critical value at every stage
of an [AI project's lifecycle](ML_Lifecycle.md). It is not just a tool for one
specific role, but a common thread of data-centric analysis that connects management,
the technical team, and the end user. It is designed to apply to multiple stages
of the lifecycle to facilitate creating robust, reliable, and generalizable AI products.

We have created a few entry-level guides for common roles found in
{term}`machine learning<Machine Learning (ML)>` and AI development, that expound
upon how DataEval can assist in many of the day-to-day tasks. For example personas
related to these roles, see the [JATIC roles and personas webpage](https://cdao.pages.jatic.net/public/program/roles-personas/).

- [Testing & Evaluation Engineer](te_engineer.md)
- [Machine Learning Engineer](ml_engineer.md)
- [Data Scientist](data_scientist.md)

:::{toctree}
:hidden:

Operational Machine Learning Lifecycle <ML_Lifecycle.md>
Testing & Evaluation Engineer <te_engineer.md>
Machine Learning Engineer <ml_engineer.md>
Data Scientist <data_scientist.md>
:::

Below is an example of how DataEval supports a project from conception to deployment,
along with the different roles that may be involved:

## DataEval in AI Projects Example

### Project Inception & Planning

A Program Manager (PM) is tasked with developing a model for a new ATR project.
They are given a small set of images with the target present from other operations
and a model that is currently deployed on a different project. The technical
team assigned to the project consists of a [Data Scientist](data_scientist.md),
a [Machine Learning (ML) Engineer](ml_engineer.md), and a
[Test and Evaluation (T&E) Engineer](te_engineer.md). To begin the project,
the team generates a list of questions that will need to be answered to ensure
a successful project.

_Example Questions_:

- "Is our initial dataset good enough to build on?"
- "Are there hidden biases or correlations in the data?"
- "Does our test data truly represent the operational environment,
including rare edge cases and adverse conditions?"
- "Is our target performance achievable?"
- "Does the existing model meet performance requirements?"
- "Since this will be deployed in a resource-constrained system, how small can my
model be and still achieve the target performance?"
- "Is the model's performance equitable across all classes in all relevant environments?"
- "Which specific data points are the most difficult for the model to learn,
and what makes them unique?"
- "How can we increase the model's performance?"
- "Why did the model give this result? What was the key driver?"
- "How robust is the model to operational conditions?"
- "How does the model's performance degrade?"
- "Are we ready to deploy? What's the operational risk?"
- "How will we know when the model needs to be retrained?"
- "How can we version our system to ensure that the model results are reproducible?"

After generating the list, the team reviews the list and determines which questions
can be answered right away and which ones will be answered during development
and testing. The sooner each question can be answered, the more likely the
project is to succeed.

Before committing millions in funding, the PM asks the Data Scientist
to analyze the small dataset they have and the ML Engineer to analyze the given
model to determine if the provided data and model are satisfactory.

#### DataEval in Planning

The Data Scientist begins with [data cleaning](../../notebooks/tt_clean_dataset.md)
to ensure that the dataset is free of duplicates and errors. They follow this up
with an analysis of the dataset for [bias or correlations](../../notebooks/tt_identify_bias.md).
They then [assess the data space](../../notebooks/tt_assess_data_space.md)
for coverage and completeness. Next they assess whether the required performance
as currently constituted for the project is [feasible](../../notebooks/h2_measure_ic_feasibility.md).
After running this final assessment, they generate a report for the manager which
shows that the required performance is feasible, but the dataset has severe
class imbalance and there is a lack of data in a nighttime environment.

The ML Engineer begins by [generating a set of dataset splits](../../notebooks/h2_dataset_splits.md)
with the cleaned dataset from the Data Scientist. Then they test the model by finetuning
on the new data train split and test against the test split. They also assess
the [dataset's sufficiency](../../notebooks/h2_measure_ic_sufficiency.md)
given this specific model. After performing these assessments, they generate a
report for the manager which shows that the existing model does not meet performance
with the provided dataset, but could potentially reach the target performance
given additional data.

After receiving the reports from the Data Scientist and ML Engineer,
the PM has a clear understanding of the data gaps. They approve a portion of
the funds of the project to focus on targeted data collection, mitigating a
huge risk from the outset.

### Development & Training

After receiving the newly collected data, the Data Scientist and ML Engineer
can now begin data analysis on the full dataset and model development.

#### DataEval in Development

To begin phase 2, the Data Scientist curates the dataset by repeating the process
from phase 1. They [clean the data](../../notebooks/tt_clean_dataset.md),
analyze the dataset for [bias or correlations](../../notebooks/tt_identify_bias.md),
and [assess the data space](../../notebooks/tt_assess_data_space.md).
After curating the dataset, they [split the dataset](../../notebooks/h2_dataset_splits.md)
into a training, validation and a hold out test set. Again, they generate a report
for the manager, this time showing the images removed and the potential lack of
coverage for a specific type of truck.

To begin phase 2, the ML Engineer selects a couple of additional models to test
in addition to the provided model. Once they receive the data splits, they train
each model and test against the validation set. They even perform a [data sufficiency](../../notebooks/h2_measure_ic_sufficiency.md)
test with each model to get a better idea of how quickly each model is nearing
its maximum performance. They then [perform an error analysis](../../notebooks/tt_error_analysis.md)
to determine what objects each model is struggling with. After performing the analysis,
they discover that the models consistently fail on a specific type of truck.
Again, they generate a report for the manager, this time detailing out how well
each model is doing, how close each model is to it's theotrical maximum and
the images that each model is struggling to identify. They also recommend that
a few additional images be gathered for the truck that all the models failed on.

After receiving the reports the manager is able to approve the collection of
some additional imagery for the specific truck in the specific environments
outlined in the data report.

### Test & Evaluation

After receiving the additional data for the specific truck class, two of the models'
performance improves and reaches the performance requirement for all classes.
The models are now candidates for deployment and handed off to the T&E Engineer
for final testing.

#### DataEval in T&E

The T&E Engineer receives the held out test set from the Data Scientist for testing
the two candidate models. They begin by analyzing the [divergence](../../notebooks/h2_measure_divergence.md)
of the data distributions between the test set and the train and validation sets.
After the test set is adequately divergent from both the train and validation sets,
the T&E Engineer focuses on identifying the edge cases in the dataset. They use
a combination of [imagestats](../../notebooks/h2_visualize_cleaning_issues.md),
{func}`.labelstats`, [coverage](../Coverage.md) and the
[out-of-distribution (OOD) detectors](../OOD.md). After identifying the edge cases,
they test both models on the entire test set, noting the performance of each model
on the edge case images. They discover that the convolutional-based model has an
unexpected failure mode: the model fails to detect objects that are partially
obscured by other objects of interest. They also discover that the transformer-based
model has an unexpected failure mode: the model fails to detect objects that are
farther away. After testing the models, they generate a report for the manager
highlighting the failure modes of each model.

### Deployment & Monitoring

Based on the results of the T&E report, the PM makes a risk-informed decision
to deploy the transformer-based model with a specific operational constraint
that the model be used only in close detection scenarios.

#### DataEval in Deployment

Now that a model is being deployed, the ML Engineer is tasked with monitoring the
model to ensure that the model continues to meet the performance requirements.
To do this, the ML Engineer collects periodic batches of operational data.
Testing it using both [data drift detection](../../notebooks/tt_monitor_shift.md)
methods and [OOD detectors](../OOD.md). Having established specific drift and OOD
thresholds, the ML Engineer is ready to initiate a retrain should the data begin
to shift from the training distribution.

From conception to deployment, DataEval provides data-centric insights that allow
risk to be managed throughout the entire cycle, teams to make smarter decision
about their product, and develop trustworthy AI.
