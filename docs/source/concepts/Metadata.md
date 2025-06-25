# Metadata

This page explains the role and importance of metadata in vision tasks and
guides you through analyzing it using our tools. For further details, see our
[tutorials](../tutorials/index.md).

## What is it

Metadata provides descriptive information about images or object detections in
your dataset. It consists of values paired with whole images or object
detections, and can be either scalar (like 3.74) or categorical (like "truck").

The table below shows the three common ways to classify metadata values:

1. From a user perspective: real numbers, integers/discrete values, or
categories
2. From a functional perspective: numerical inputs or categorical inputs
3. From a processing perspective: continuous or discrete values

When working with metadata, identifying where your values fit in one
classification helps determine how they relate to the other classifications.
For example, if you have real numbers, they are numerical inputs and typically
processed as continuous values.

![metadata value types](../_static/images/concepts/metadataverse.png "Metadata value types")

## How is it used

Machine learning models don't use metadata directly for training. Instead,
metadata serves critical purposes at two points in the ML lifecycle:

1. **During development**: Reveals subtle aspects of training data that, if
ignored, might otherwise lead a model to focus on irrelevant patterns and thus
perform poorly in the real world.
2. **After deployment**: Helps identify and understand shifts in incoming data
that affect performance.

Examples of metadata include data sources, feature descriptions, timestamps,
labels, sensor types, and preprocessing information; these examples are
inherently extrinsic. Some metadata—explicitly referred to as "intrinsic
metadata"—is derived from calculations on the data itself. For example, we
might want to compute overall brightness or contrast of an image, or compute a
measure of graininess or color saturation. DataEval can compute intrinsic
metadata and add it to your datasets.

Metadata enables:

- Better understanding of dataset context and origins
- Enhanced reproducibility of results
- Identification of potential biases or limitations
- Diagnosis of unexpected changes in production data

By leveraging metadata effectively, you can improve model interpretability,
enhance feature engineering, and boost overall performance by incorporating
contextual information.

Metadata also strengthens monitoring and auditing pipelines, ensuring systems
meet real-world requirements and ethical standards. When you detect dataset
drift or an increase in out-of-distribution (OOD) examples, examining metadata
can often reveal the underlying causes and suggest appropriate actions.

## Why is it important?

{term}`Statistical independence<Statistical Independence>` between class labels
and metadata attributes prevents models from learning misleading correlations.
Without this independence, models often develop shortcuts based on contextual
factors rather than relevant features.

Consider this example: You want to train a model to classify images as either
cows or horses. In your dataset, all cow pictures were taken in Washington
state, while all horse pictures were taken in Arizona. A model trained on this
data might learn to associate:

- Cows with grass and evergreen trees
- Horses with sand and cacti

This model would likely misclassify a horse photographed on a grassy field in
Washington as a cow, because it learned the wrong associations. By analyzing
metadata before training, you can identify and address these misleading
relationships, resulting in models that generalize more effectively.

Early detection and mitigation of metadata bias is essential for developing
fair and reliable models. After deployment, continuous monitoring of bias
metrics and investigation of metadata changes helps maintain robust performance
when facing unexpected shifts in your data.

## How to analyze it

Understanding the correlational relationships within your dataset is crucial
for both model and dataset development, as well as after deployment.

### Pre-training

In practice, data collection opportunities are often limited by practical
constraints—available only in specific locations, under certain conditions, or
with limited target diversity. Models trained on such constrained datasets might
learn to rely on secondary information rather than the primary features you want
them to learn. This reduces their ability to generalize to new domains and can
lead to unexpected behavior with new data.

DataEval provides several metrics to help you identify and address these issues:

- {func}`.balance` - Evaluates distribution of factors across your dataset
- {func}`.diversity` - Measures variety within dataset attributes
- {func}`.parity` - Identifies relationships between dataset factors and class
labels

Testing engineers and model developers should use these insights to:

1. Design targeted tests for model generalization
2. Implement data augmentation to prevent shortcut learning
3. Address sampling imbalances before training

### Post-deployment

DataEval offers post-deployment metadata exploratory tools:

- {func}`.find_most_deviated_factors` - Identifies metadata elements that
differ most from expected patterns
- {func}`.find_ood_predictors` - Identifies metadate features that might
predict out-of-distribution examples
- {func}`.metadata_distance` - Measures featurewise similarity between metadata
distributions

These tools essentially ask: "What stands out about our incoming metadata?" and
"Can we predict which examples are OOD based on metadata?"

### Metadata formats

To use DataEval's bias metrics or exploration tools, you must provide metadata
in DataEval's specific format, which is implemented in DataEval's
{class}`.Metadata` class.
