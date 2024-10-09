# Upperbound on Average Precision

## What is it

Upperbound average precision refers to the irreducible error in a particular object detection problem.
The `UAP` metric assesses the feasibility of a machine learning object detection task by estimating this error.
Specifically, it takes an object detection problem, and reduces it to an (easier) classification problem.

## When to use it

The `UAP` metric should be used when you would like to measure the feasibility of a machine learning object detection task.
For example, you would like to know if the operational mean average precision (mAP) requirement of 50% is achievable given the imagery.

This quantity is of interest because it informs an engineer about the inherent difficulty of a problem. If this difficulty surpasses operational performance requirements, then the problem must be changed in order to become feasible.

## Theory behind it

In general, object detection tasks can be broken down into two related subtasks, localization and classification. The former determines where the object is located, and the latter determines what the object is. Object detectors are typically evaluated based on the mean average precision, or mAP. This quantity takes the mean over the average precision of every class in the dataset. The average precision for a given class is typically the area under the precision recall curve averaged over a variety of bounding box overlap thresholds. More information on the details of object detection is widely available. 

Rather than train an expensive object detector, UAP instead trains a classifier on only the bounding boxes in a dataset. Put simply, we upperbound the mAP by removing localization from the equation. The mAP of the resulting, easier classification problem is what is reported by the UAP metric. More information on UAP and its origin can be found [here](https://arxiv.org/abs/1911.12451).

## References
 [1] [Borji, A., & Iranmanesh, S. M. (2019). Empirical Upper Bound in Object Detection and More.](https://arxiv.org/abs/1911.12451)