# Coverage

## What is it

Coverage informs users if there are images that are uncovered in a particular dataset.
An uncovered image is one which does not have a sufficient number of similar images in the dataset.

## When to use it

Coverage can be used to determine if certain meta-features are undersampled in a dataset. 
For example, perhaps a dataset contains a sufficient number of images of cows, but only one or two instances of cows on beaches. 
In this scenario, a typical diagnostic test of dataset metadata would likely say there aren't any bias or undercoverage issues, but there is undercoverage of cows on beaches in reality. 
Users should investigate undercovered images to determine what the underlying meta-features are which may be undercovered.

## Theory behind it

Coverage, like many non-parametric data diagnostic tools, is rooted in dimension reduction. 
We assume, for the sake of tractability, that a set of images can be embedded in a low-dimensional subspace of the original pixel space.
This space characterizes the difference between images.
Typically, we learn this space via an autoencoder.

Once we have a lower dimensional embedding subspace, we look at the distribution of the images within it.
Specifically, with coverage, we are looking for images which do not have many other images in their immediate vicinity.
By default, we look at the smallest d-dimensional ball around a given image, where d is the dimension of the embedding, which covers $30$ images.
If the radius of that ball is above a certain prespecified value, or if the radius of that ball is in the top $x$ % of all the radii in the dataset, we say an image is uncovered, and recommend the user investigate it further.
While the decision to label an individual image as uncovered is dependent on an (arbitrary) radius, it stands to reason that uncovered images will have more systemic meta-feature undersampling than covered images on average.
