"""
This module contains general utility functions used throughout the repo.
"""
import torch
import numpy as np
from typing import List, Union


def get_device():
    """
    Auto-detects what hardware is available and returns the appropriate device.
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
