import numpy as np
import torch


def pytorch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a PyTorch tensor to a NumPy array
    """
    if isinstance(tensor, np.ndarray):  # Already array, return
        return tensor
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Tensor is not of type torch.Tensor")

    x: np.ndarray = tensor.detach().cpu().numpy()
    return x


def numpy_to_pytorch(array: np.ndarray) -> torch.Tensor:
    """
    Converts a NumPy array to a PyTorch tensor
    """
    if isinstance(array, torch.Tensor):  # Already tensor, return
        return array
    if not isinstance(array, np.ndarray):
        raise TypeError("Array is not of type numpy.ndarray")
    x: torch.Tensor = torch.from_numpy(array.astype(np.float32))
    return x


def permute_to_torch(array: np.ndarray) -> torch.Tensor:
    """
    Converts and permutes a NumPy image array into a PyTorch image tensor.

    Parameters
    ----------
    array: np.ndarray
        Array containing image data in the format NHWC

    Returns
    -------
    torch.Tensor
        Tensor containing image data in the format NCHW
    """
    x = numpy_to_pytorch(array)
    x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
    return x


def permute_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts and permutes a PyTorch image tensor into a NumPy image array.

    Does not permute if given np.ndarray

    Parameters
    ----------
    tensor: torch.Tensor
        Tensor containing image data in the format NCHW

    Returns
    -------
    np.ndarray
        Array containing image data in the format NHWC
    """
    x = tensor.permute(0, 2, 3, 1)
    x = pytorch_to_numpy(x)  # NCHW -> NHWC
    return x
