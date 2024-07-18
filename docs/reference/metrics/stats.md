(stats_ref)=

% ============================

% Dataset Stats

% ============================

% The basic Dataset Stats class assists with exploratory dataset analysis (EDA).

% The class delivers stats for the following aspects of images:

% * height

% * width

% * size

% * aspect ratio

% * number of channels

% * pixel value range

% * mean pixel value

% * missing values (NaNs)

% * number of 0 value pixels

% * pixel variance

% * pixel skew

% * pixel kurtosis

% * max/min pixel value along with the 25th and 75th percentiles

% * overall brightness of the image

% * blurriness of the image

% * entropy of the image

% The above stats are also calculated on a per channel basis if the images have more than one channel.

% This class can be used to determine if there are any issues with any of the images in the dataset.

% As well as give a big picture view of how similar the images are to one another.

% ---------

% Tutorials

% ---------

% Check out this tutorial to begin using the basic ``DatasetStats`` class

% :doc:`Dataset Stats and Deduplication Tutorial<../../tutorials/notebooks/DatasetStats-HashTutorial>`

% -------------

% How To Guides

% -------------

% There are currently no how to's for the Basic Stats Class.

% If there are scenarios that you want us to explain, contact us!

# Image Statistics

% Create small blurb here that answers:

% 1. What it is

% 2. What does it solve

## Tutorials

There are currently no tutorials for `ImageStats`.

## How To Guides

There are currently no how to's for `ImageStats`.
If there are scenarios that you want us to explain, contact us!

## DataEval API

```{eval-rst}
.. autoclass:: dataeval.metrics.ImageStats
   :members:
   :inherited-members:
```

# Channel Statistics

% Create small blurb here that answers:

% 1. What it is

% 2. What does it solve

## Tutorials

There are currently no tutorials for `ChannelStats`.

## How To Guides

There are currently no how to's for `ChannelStats`.
If there are scenarios that you want us to explain, contact us!

## DataEval API

```{eval-rst}
.. autoclass:: dataeval.metrics.ChannelStats
   :members:
   :inherited-members:
```

# Image Flags

```{eval-rst}
.. autoflag:: dataeval.flags.ImageHash
```

```{eval-rst}
.. autoflag:: dataeval.flags.ImageProperty
```

```{eval-rst}
.. autoflag:: dataeval.flags.ImageVisuals
```

```{eval-rst}
.. autoflag:: dataeval.flags.ImageStatistics
```
