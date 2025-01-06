# Detecting {term}`duplicates<Duplicates>`

## What is it

The {term}`Duplicates` class identifies exact and near duplicate data.

## When to use it

The {term}`Duplicates` class should be used if you need to check for duplicates
in your dataset.

## Theory behind it

With the {term}`Duplicates` class, exact matches are found using a byte hash of
the data information, while near matches (such as a crop of another image or a
distoration of another image) use a perception based hash.

The byte hash is achieved through the use of the
[python-xxHash](https://github.com/ifduyue/python-xxhash) Python module,
which is based on Yann Collet's [xxHash](https://github.com/Cyan4973/xxHash) C
library.

The perceptual hash is achieved on an image by resizing to a square NxN image
using the Lanczos algorithm where N is 32x32 or the largest multiple of 8 that
is smaller than the input image dimensions. The resampled image is compressed
using a discrete cosine transform and the lowest frequency component is encoded
as a bit array of greater or less than median value and returned as a hex
string.
