# Dataset balancing

This page describes dataset balancing, including the theory and
application of the tool, how to use it, and why to use it.

## What is dataset balancing?

Dataset balancing is the process of adjusting the contribution of different classes during model
training to ensure that rare or minority classes are effectively learned. In many real-world scenarios,
datasets are naturally imbalanced—where one class (the majority) significantly outnumbers others (the minority).

A critical component of understanding dataset balancing is the [**Accuracy Paradox**](https://www.google.com/url?q=https://en.wikipedia.org/wiki/Accuracy_paradox&sa=D&source=docs&ust=1767042624671451&usg=AOvVaw1336ED_Dn6C702RAyMMiZ8).
When a dataset is heavily imbalanced, metrics like "accuracy" can be misleading; a model can
achieve 99% accuracy by simply predicting the majority class every time, and thus failing on
the classes that make up the remaining 1% of data. To account for this, balancing focuses on
**macro-averaging** rather than **micro-averaging**. Macro-averaging treats every _class_ with
equal weight regardless of the number of samples while micro-averaging treats every _sample_
with equal weight.

## Why should you balance your dataset?

Addressing class imbalance is essential for fielding robust, generalizable models,
particularly in safety-critical applications. Balancing your dataset provides the following benefits:

1. **Robustness and Generalization**:
Models trained on imbalanced data often overfit to the majority class. Balancing ensures the model
learns the distinguishing features of rare classes, leading to better performance in diverse real-world conditions.

2. **Mitigation of Catastrophic Failures**:
In fields like autonomous driving or medical imaging, misidentifying a rare class
(e.g., a plastic bag in the road or a rare tumor) can have catastrophic consequences.
Balancing ensures these "long-tail" events are prioritized during the learning process.

3. **Improved Decision Boundaries**:
By emphasizing minority classes, balancing helps the model define more accurate decision boundaries
in the latent space, preventing the majority class from "crowding out" the features of rarer instances.

## When should you balance your dataset?

Typical times to implement dataset balancing strategies are:

1. **During Exploratory Data Analysis (EDA)**:
Once initial [cleaning](DataCleaning.md) is complete, engineers should calculate class frequencies.
If the majority class accounts for a disproportionate amount of the data (e.g., >90%),
balancing strategies should be planned immediately.

2. **When Macro-metrics Underperform**:
If a model shows high overall accuracy but poor precision or recall on specific minority classes,
it is a clear signal that the loss function is being dominated by majority class samples.

3. **After Incorporating New Data**:
Newly collected data often follows a different distribution than the original training set.
Prioritization and cleaning tools should be used together to identify if the new data exacerbates
existing imbalances or helps mitigate them.

## Current dataset balancing techniques

Modern machine learning offers three primary approaches to addressing dataset imbalance.
These range from simple algorithmic adjustments to physical manipulation of the dataset.
We recommend trying these techniques in the order listed below to minimize the amount
of dataset manipulation for previously currated datasets. However, if you are curating
a dataset, then start with resampling.

### Loss Weighting

A simple and effective approach to balancing is adjusting the loss function,
as it requires no changes to the underlying data.

* **Weighted Cross-Entropy**: This assigns weights to each class, allowing a user to adjust the
contribution to the total loss for each class. Using this method, contributions by rare classes
can be proportional to their importance rather than their frequency.
This implicitly corrects for class imbalance by ensuring that the optimization step isn’t
dominated by the majority class. The optimal weights to assign each class vary depending
on the loss function, but are typically proportional to the inverse of the class frequency
–that is, the more samples of a class, the smaller the class weight.

* **Focal Loss**: This dynamically assigns weights based on how "correct" a prediction is.
As the model masters the easy, majority-class samples, their loss contribution is downweighted,
forcing the model to focus on the more difficult, minority-class samples.

### Image Augmentation

Generating composite images made up of multiple training images creates "synthetic" variety
for minority classes. These composite augmentations combine multiple randomly selected
images into a single composite image whose labels are similarly transformed to preserve
spatial and categorical information. This technique is especially effective with low to
moderate class imbalance, as the rarest classes are still reliably selected during
random sampling of the training dataset. In cases with extreme class imbalance,
the rarest classes may see only a slight increase in effective looks without
additional sampling criteria.

* **Mosaic (CutMix)**: Stitches random crops from multiple (usually two or four) images into
a single training sample, preserving the spatial and categorical information of all included classes.
Mosaic is widely available in many computer vision libraries, and was designed to improve robustness
and generalization in object detection datasets for single-stage detectors.

* **MixUp**: Generates a new image via a weighted sum of two images: $x = \lambda x_i + (1-\lambda) x_j$.
The mixing weight $\lambda$ is drawn from a $\beta$-distribution
$\beta (\alpha, \alpha)$, where $\alpha$ is a user defined parameter.
The labels for the resulting image are simply the concatenation of labels from the input image.
This forces the model to learn smoother transitions between class boundaries.

### Resampling

Resampling is more involved than either loss weighting or image augmentation
as it involves duplicating or removing data. The overall goal of resampling is
to generate a more uniform class distribution. This can be accomplished through
oversampling the minority samples, undersampling the majority samples or some
combination of the two.

* **Oversampling**: Duplicates minority samples. While this increases looks at rare targets,
it significantly increases the risk of overfitting.

* **Undersampling**: Removes majority samples. This prevents the majority class from dominating
but may result in the loss of valuable, non-redundant background information.

Resampling can be especially difficult outside of simple image classification datasets.
Datasets whose images contain multiple objects or labels may not be able to achieve a
uniform class distribution. DataEval's class balance function is designed
to find a class distribution that compromises between attaining a uniform class
distribution and reducing the number of under/oversampled data. We recommend using
DataEval's function rather than trying to achieve a perfectly uniform distribution.
After resampling, you may still want to train the new data using one or both of the
techniques listed above.

## Related concepts

Dataset balancing is closely tied to other curation tasks in DataEval:

* **Dataset Prioritization**: While balancing focuses on class counts, [prioritization](Prioritization.md)
focuses on sample "difficulty" and redundancy.
* **Outlier Detection**: Rare classes can sometimes be misidentified as [outliers](Outliers.md).
Balancing ensures these rare but valid samples are retained rather than pruned.
* **OOD (Out-of-Distribution)**: A model that is well-balanced is typically more resilient when encountering
[out-of-distribution](OOD.md) data that may share features with minority training classes.

## References and related work

1. He, K., et al. (2017). "Focal Loss for Dense Object Detection." _IEEE International Conference on Computer Vision (ICCV)_.
2. Zhang, H., et al. (2017). "mixup: Beyond Empirical Risk Minimization." _arXiv preprint arXiv:1710.09412_.
3. Bochkovskiy, A., et al. (2020). "YOLOv4: Optimal Speed and Accuracy of Object Detection."
_arXiv preprint arXiv:2004.10934_. (Discussion on Mosaic augmentation).

<!--
Additional information from Justin's original document, still trying to decide what
to do with it.

Instead of resampling the dataset before training, consider passing the balancer
to the “DataLoader” object with the “Sampler” argument. Doing so will cause the
balancer to rebalance the original dataset after each training epoch, likely
resulting in a dataset with images not included in the prior training epoch.

If one class is dramatically overrepresented, consider passing it as the
“background_class” argument to the resampler. This will prevent the resampler
from specifically adding images containing the background class.

If one class is dramatically underrepresented, we recommend using the
InterClassBalancer with the “minimize_duplicates” set to True.
This ensures that every image containing the rare class is included at
least once in the dataset. Setting “minimize_duplicates” to False may result
in a better balance, but at the expense of valuable looks at rare targets. -->
