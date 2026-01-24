"""
This module contains general utility functions used throughout the repo.
"""
import sys, os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, CURRENT_DIR)

import torch, yaml
import numpy as np
import pandas as pd
from typing import List, Union
import matplotlib.pyplot as plt
from pathlib import Path


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


def denormalize_patches(original_patches: torch.Tensor, pred_patches: torch.Tensor,
                        eps: float = 1e-6) -> torch.Tensor:
    """
    This helper function reverses patch normalization.

    :param original_patches: An input batch of image patches of size (N, num_patches, patch_dim) from which
        to compute means and variances to reverse the whitening operation.
    :param pred_patches: An input batch of image patches of size (N, num_patches, patch_dim) that will be
        un-normalized using the mean and variance from original_patches.
    :returns: pred_patches after reversing the patch-level normalization.
    """
    mean = original_patches.mean(dim=-1, keepdim=True)  # Compute the per patch mean pixel value
    var = original_patches.var(dim=-1, keepdim=True, unbiased=False)  # Compute the per patch pixel variance
    return pred_patches * torch.sqrt(var + eps) + mean


def denormalize_imagenet(imgs: torch.Tensor) -> torch.Tensor:
    """
    Reverse the standard mean and stddev scaling normalizations applied in the data-loader.

    :param imgs: An input batch of images of size (N, C, H, W) to reverse the normalizations of.
    :returns: An output batch of images of the same size as the input with the normalizations undone.
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=imgs.device).view(1, 3, 1, 1)
    return (imgs * std + mean).clamp(0, 1)


def plot_and_save_loss(loss_dir: str) -> None:
    """
    Combines all the data cached to a directory of loss value outputs and combines them together to create a
    loss plot which is then saved down in the same directory as well.

    :param loss_dir: A directory containing losses-{milestone}.csv files.
    :returns: None, generates a plot that is then saved to disk.
    """
    filenames = [x for x in os.listdir(loss_dir) if x.startswith("losses") and x.endswith(".csv")]
    if len(filenames) > 0:  # Otherwise do nothing
        all_losses = []
        milestones = [int(x.replace("losses-", "").replace(".csv", "")) for x in filenames]
        for m in sorted(milestones):
            df = pd.read_csv(os.path.join(loss_dir, f"losses-{m}.csv"), index_col=0)
            all_losses.extend(df.iloc[:, 0].tolist())
        all_losses = pd.Series(all_losses)  # Convert to a pd.Series for ease of use
        all_losses.index += 1  # Set the index to begin at 1
        # Create a plot and save it to the same directory
        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
        ax.plot(all_losses, zorder=3)
        ax.set_ylabel("Loss")
        ax.set_xlabel("Training Step")
        ax.set_title("Training Loss")
        ax.grid(color="lightgray", zorder=-3)
        fig.savefig(os.path.join(loss_dir, "training_loss.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)


def read_config(config_name: str, dataset_dir: str = "dataset/preprocessed") -> dict:
    """
    Helper function that reads in a yaml config file specified and returns the associated data as a dict.

    :param config_name: A str denoting the name of the config e.g. "debug" or "prod".
    :param dataset_dict: The location of the dataset to add to the config file.
    :return: A dictionary of data read in from the yaml config file.
    """
    file_path = os.path.join(CURRENT_DIR, f"config/{config_name}.yml")
    with open(file_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["dataset_dir"] = dataset_dir
    cfg["DataLoaderTrain"]["dataset_dir"] = dataset_dir
    cfg["DataLoaderTrain"]["dataset_dir"] = dataset_dir
    cfg["DataLoaderVal"]["dataset_dir"] = dataset_dir
    device = get_device()
    cfg["DataLoaderTrain"]["device"] = device
    cfg["DataLoaderVal"]["device"] = device
    return cfg


