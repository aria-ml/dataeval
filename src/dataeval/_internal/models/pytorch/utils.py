from numpy import float32, ndarray
from torch import Tensor, from_numpy


def torch_to_numpy(tensor: Tensor) -> ndarray:
    """
    Converts a PyTorch tensor to a NumPy array
    """
    if isinstance(tensor, ndarray):  # Already array, return
        return tensor
    if not isinstance(tensor, Tensor):
        raise TypeError("Tensor is not of type Tensor")

    x: ndarray = tensor.detach().cpu().numpy()
    return x


def numpy_to_torch(array: ndarray) -> Tensor:
    """
    Converts a NumPy array to a PyTorch tensor
    """
    if isinstance(array, Tensor):  # Already tensor, return
        return array
    if not isinstance(array, ndarray):
        raise TypeError("Array is not of type numpy.ndarray")
    x: Tensor = from_numpy(array.astype(float32))
    return x


def permute_to_torch(array: ndarray) -> Tensor:
    """
    Converts and permutes a NumPy image array into a PyTorch image tensor.

    Parameters
    ----------
    array: ndarray
        Array containing image data in the format NHWC

    Returns
    -------
    Tensor
        Tensor containing image data in the format NCHW
    """
    x = numpy_to_torch(array)
    x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
    return x


def permute_to_numpy(tensor: Tensor) -> ndarray:
    """
    Converts and permutes a PyTorch image tensor into a NumPy image array.

    Does not permute if given ndarray

    Parameters
    ----------
    tensor: Tensor
        Tensor containing image data in the format NCHW

    Returns
    -------
    ndarray
        Array containing image data in the format NHWC
    """
    x = tensor.permute(0, 2, 3, 1)
    x = torch_to_numpy(x)  # NCHW -> NHWC
    return x
