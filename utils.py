"""
This module contains general utility functions used throughout the repo.
"""
import torch
import numpy as np
from typing import List, Union
from torchvision.utils import save_image


def get_device():
    """
    Auto-detects what hardware is available and returns the appropriate device.

    :returns: A torch device denoting what device is available.
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def get_amp_dtype(device: str = "cuda"):
    """
    Determines the Automatic Mixed Precision data type that can be used on the current hardware.

    :param device: The device currently available as a string e.g. "cpu" or "cuda".
    :returns: A torch float type for auto mixed precision training.
    """
    assert isinstance(device, str), "device must be a str"
    if device != "cuda" or not torch.cuda.is_available():
        return torch.float16

    # Get compute capability (major, minor)
    major, minor = torch.cuda.get_device_capability()

    # Ampere (8.x), Hopper (9.x), Ada (8.9) → BF16 supported
    bf16_supported = (major >= 8)

    return torch.bfloat16 if bf16_supported else torch.float16


def decode_caption(word_ids: Union[np.ndarray, List[int]], sp_model) -> str:
    """
    Converts a list or np.ndarray of sub-word token ids to an output string sentences. Removes the special
    start <s>, end </s>, and <pad> padding tokens.

    :param word_ids: A numpy array of integers or a list of integers that correspond to sub-piece tokens.
    :param sp_model: A sub-piece token model for converting token indices to strings.
    :returns: A string that is the decoded word sequence represented by the word tokens.
    """
    word_ids = word_ids.tolist() if hasattr(word_ids, "tolist") else word_ids  # Also accept an np.ndarray

    output = []  # Filter the word token IDs to remove the special tokens
    for i in word_ids:
        if i == sp_model.eos_id():  # Stop early if we see the end sentence token
            break
        if i in (sp_model.pad_id(), sp_model.bos_id()):  # Skip padding and start of sentence tokens
            continue
        output.append(i)

    return "".join(sp_model.decode(output)).replace("_", "")


def normalize_patches(patches: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Independently normalizes each image patch where patches is (N, num_patches, patch_dim) where
    patch_dim = P * P * C = patch_size * patch_size * in_channels.

    :param patches: An input batch of image patches of size (N, num_patches, patch_dim).
    :returns: Applies a whitening transform to each patch independently and returns a tensor that is the
        same size as the original input patches.
    """
    mean = patches.mean(dim=-1, keepdim=True)  # Compute the per patch mean pixel value
    var = patches.var(dim=-1, keepdim=True, unbiased=False)  # Compute the per patch pixel variance
    whitened_patches = (patches - mean) / torch.sqrt(var + eps)  # Apply the whitening normalization
    return whitened_patches


def denormalize_patches(patches: torch.Tensor, pred_patches: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    This helper function reverses patch normalization.

    :param patches: An input batch of image patches of size (N, num_patches, patch_dim) from which to compute
        means and variances.
    :param pred_patches: An input batch of image patches of size (N, num_patches, patch_dim) that will be
        un-normalized using the mean and variance from patches.
    :returns: pred_patches after reversing the patch-level normalization.
    """
    mean = patches.mean(dim=-1, keepdim=True)  # Compute the per patch mean pixel value
    var = patches.var(dim=-1, keepdim=True, unbiased=False)  # Compute the per patch pixel variance
    return pred_patches * torch.sqrt(var + eps) + mean


def save_patch_grid(images: torch.Tensor, patch_size: int, grid_size: int, n_channels: int,
                    filepath: str) -> None:
    """
    Takes an input tensor of images of size (N, num_patches, patch_dim), reshapes them into an image grid,
    and saves them to disk.

    :param images: An input tensor of size (N, num_patches, patch_dim) containing image pixels to be saved to
        disk as an image grid.
    :param patch_size: The size of each patch as an int e.g. 14 or 16. The relation to patch_dim is:
        patch_dim = n_channels * patch_size ** 2.
    :param grid_size: The size of the image grid, N should equal grid_size ** 2.
    :param n_channels: The number of color channels of the images.
    :param filename: The file path to save the image grid to.
    :returns: None, saves the image grid to disk.
    """
    N, num_patches, patch_dim = images.shape
    H = W = int(num_patches ** 0.5)
    img_grid = images.view(N, H, W, n_channels, patch_size, patch_size).permute(0, 3, 1, 4, 2, 5).contiguous()
    img_grid = img_grid.view(N, n_channels, H * patch_size, W * patch_size)
    save_image(img_grid, filepath, nrow=grid_size)
    print(f"Saved {N} images as a [{grid_size}x{grid_size}] grid to {filepath}")
