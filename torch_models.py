"""
This module defines Transformer models and the layers used to create them.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
import copy
from utils import decode_caption, normalize_patches
from typing import List, Union, Tuple
import clip


########################
### Helper Functions ###
########################
# TODO: Section marker


def clone(module: nn.Module, N: int):
    "Produces N identical layers of the input module."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def random_masking(x: torch.Tensor, mask_ratio: float = 0.75) -> Tuple[torch.Tensor]:
    """
    Helper function which randomly masks some ratio of the image patches in the input image embeddings.
    Returns the reduced subset of visible image patch embeddings, a tensor denoting which image patches were
    masked and a tensor denoting how to reverse the random shuffling of image patches to restore the original
    image patch ordering. Note that x_visible has been shuffled.

    :param x: An input batch of image patch embeddings of size (N, num_patches, embed_size).
    :param mask_ratio: The proportion of image patches that should be randomly masked.
    :returns:
        - x_visible: A batch of visible image patches of size (N, num_visible, embed_size).
        - mask: A tensor denoting which image patches were masked (encoded as 1s) of size (N, num_patches).
        - ids_restore: A tensor denoting the reverse mapping of the shuffled image patch indices to restore
            the original ordering. This can be used to re-construct the original image and tells us where the
            patches of x_visible belong.
    """
    N, num_patches, embed_dim = x.shape  # Unpack the dimensions of the input
    len_keep = int(num_patches * (1 - mask_ratio))  # Determine how many patches to keep i.e. not mask

    noise = torch.rand(N, num_patches, device=x.device)  # Generate random noise values [0, 1], one for each
    # image patch in each batch obs so that we can randomly sort the patches within each image (N, n_patches)

    ids_shuffle = torch.argsort(noise, dim=1)  # Sort the random noise values generated above within each
    # image to get a random shuffle of the image patches within each image (N, n_patches), this is a tensor
    # of image patch indices for each image i.e. [0, num_patches - 1]
    ids_restore = torch.argsort(ids_shuffle, dim=1)  # This gives us the reverse mapping i.e. how to index
    # ids_shuffle to obtain the original ordering (N, n_patches) e.g. the location of 0 in ids_shuffle[i, :]
    # is where the first image patch will be in the shuffled set of the ith image

    ids_keep = ids_shuffle[:, :len_keep]  # Randomly select which image patches to keep (N, len_keep)
    # Extract out the image patches for each image to keep i.e. the ones that will be visible to the encoder
    # to get a shuffled batch of image patches of size (N, len_keep, embed_dim)
    x_visible = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, embed_dim))

    mask = torch.ones(N, num_patches, device=x.device)  # (N, num_patches) Start off with all ones
    mask[:, :len_keep] = 0  # Use zeros to denote where the masking took place
    mask = torch.gather(mask, dim=1, index=ids_restore)  # Re-arrange the masking values to reflect which of
    # the original image patches were masked (N, num_patches)

    # (N, n_vis, embed_dim), (N, num_patches), (N, num_patches)
    return x_visible, mask, ids_restore


def word_dropout(word_idx_seq: torch.Tensor, dropout: float = 0.1, pad_idx: int = 0):
    """
    Applies word token dropout by randomly masking some of the work token indices in word_idx_seq by replacing
    them with the padding token index value (usually 0). This encourages the decoder to use the visual
    features instead of memorizing word sequences during training.

    :param word_idx_seq: A tensor containing a batch of word index sequences i.e. integers of size
        (N, T) where T is the max length among the sequences.
    :param dropout: The probability of randomly replacing any given word index with the padding token index.
    :param pad_idx: The index value associated with the padding token, usually 0.
    :returns A tensor of the same shape (N, T) as the input word_idx_seq with some tokens ids randomly
        replaced with the padding token ID.
    """
    mask = (torch.rand_like(word_idx_seq.float()) < dropout) & (word_idx_seq != pad_idx) # (N, T)
    mask[0, :] = False # Never mask the first BOS <s> special token that begins the sentence
    word_idx_seq = word_idx_seq.clone()
    word_idx_seq[mask] = pad_idx
    return word_idx_seq # (N, T)

NO_DECAY_LAYERS = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.Embedding)


def get_param_groups(model, weight_decay: float, lr: float, verbose: bool = False) -> List:
    """
    This method is used to create 2 groups of parameters for a model:
        1). A group of parameters to apply weight decay to e.g. Linear, Attention, etc.
        2). A group of parameters to exclude from weight decay e.g. biases, LayerNorm, Embedding, positional
            encodings etc.

    :param model: A nn.Module model object.
    :param weight_decay: The amount of weight decay to apply to those parameter groups which are not excluded
        from weight decay.
    :param lr: The learning rate to apply to all parameters within the model.
    :returns: A list of dictionaries suitable to be passed to an optimizer for model training.
    """
    decay = []
    no_decay = []
    seen_params = set()

    if hasattr(model, "num_patches") and hasattr(model, "embed_dim"):
        img_patch_embed_shape = (1, model.num_patches + 1, model.embed_dim)
        cls_token_shape = (1, 1, model.embed_dim)
    else:
        img_patch_embed_shape, cls_token_shape = None, None

    for module in model.modules():
        for param in module.parameters(recurse=False):
            if not param.requires_grad:
                continue

            if id(param) in seen_params:
                continue  # skip duplicate references, handles the case of tied parameters

            if param.ndim == 1 or isinstance(module, NO_DECAY_LAYERS):
                no_decay.append(param)
            elif cls_token_shape is not None and param.shape in [img_patch_embed_shape, cls_token_shape]:
                no_decay.append(param)

            else:
                decay.append(param)

            seen_params.add(id(param))

    # Check that all parameters have been accounted for i.e. either in the decay or no decay category
    total_params = sum(p.numel() for p in model.parameters())
    decay_params = sum(p.numel() for p in decay)
    no_decay_params = sum(p.numel() for p in no_decay)
    assert decay_params + no_decay_params == total_params, "decay_params + no_decay_params =/= total_params"

    if verbose:
        print("Total param count:", total_params)
        print("Decay param count:", decay_params, f"{decay_params / total_params:.1%}")
        print("No-decay param count:", no_decay_params, f"{no_decay_params / total_params:.1%}")

    return [
        {"params": decay, "weight_decay": weight_decay, "lr": lr},
        {"params": no_decay, "weight_decay": 0.0, "lr": lr},
    ]


##############################
### Transformer Sub-Blocks ###
##############################
# TODO: Section marker

class MultiHeadedAttention(nn.Module):
    """
    A transformer model layer which implements a simplified version of masked, multi-headed, attention as
    introduced by "Attention Is All You Need" (https://arxiv.org/abs/1706.03762).

    Example Usage:
        attn = MultiHeadedAttention(embed_dim, num_heads=8)

        # Self-Attention layer for some input 'data' of size (B, T, E)
        self_attn_output = attn(query=data, key=data, value=data)

        # Cross-Attention using 2 inputs 'data_1' and 'data_2' of size (B, T, E)
        attn_output = attn(query=data_1, key=data_2, value=data_2)
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, cross_attn: bool = False):
        """
        Initializes a MultiHeadedAttention layer. The hidden dimension of the key, query, and value vectors
        is set equal to embed_dim as per the usual convention.

        :param embed_dim: The size of the embedding dimension of input tokens.
        :param num_heads: The number of attention heads to use when computing attention outputs.
        :param dropout: The probability of dropout used during training.
        :param cross_attn: Denotes if this attention layer is a cross-attention layer, which only impacts how
            the key-value cache is utilized to minimize redundant computations.
        """
        super().__init__()

        # Create linear transforms from input (token embeddings + positional embeddings) to key, query, and
        # value vectors using a nn.Linear layer to perform the matrix multiplication of learned weights
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.proj = nn.Linear(embed_dim, embed_dim)  # A final projection layer after computing attention
        self.attn_drop = nn.Dropout(dropout)

        self.num_heads = num_heads
        self.embed_dim = embed_dim  # Also used as the size of the key, query, and value vectors
        # The size of each key, query, and value vector within attention head
        msg = f"embed_dim={self.embed_dim} must be a multiple of num_heads={self.num_heads}"
        assert self.embed_dim % self.num_heads == 0, msg
        self.head_dim = self.embed_dim // self.num_heads

        self.is_cross_attn = cross_attn # Record a bool flag for whether this is a cross-attention layer,
        # which impacts how the key-value caching is updated each step i.e. if it is cross-attention, then we
        # don't update the kv cache after the first iteration i.e. the image patches need only be processed 1x
        self.kv_cache = None # Cache key-value pairs to speed up auto-regressive rollouts

    def clear_cache(self) -> None:
        """
        This method clears out the key-value cache and should be called before starting step-wise decoding.
        """
        self.kv_cache = None

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: torch.Tensor = None, pad_mask: torch.Tensor = None,
                return_attn: bool = False, step: bool = False) -> torch.Tensor:
        """
        Calculates the masked attention output for the provided input data, computes all attention heads in
        parallel for computational efficiency.

        In the shape definitions below, N is the batch size, S is the reference data sequence length i.e.
        used for the keys and values and often called the source, and T is the target sequence length i.e.
        used for the query values. E is the embedding dimension, which is equal to the hidden dimension as
        well.

        :param query: Input data to be used to create the query vectors, of shape (N, T, E).
        :param key: Input data to be used to create the key vectors, of shape (N, S, E).
        :param value: Input data to be used to create the value vectors, of shape (N, S, E).
        :param attn_mask: A tensor of shape (T, S) where mask[i, j] == 0 indicates that token i in the target
            should not be influenced by token j in the source.
        :param pad_mask: A tensor of shape (N, S) denoting the location of padding tokens so that they are
            not attended to during evaluation by any of the query tokens.
        :param return_attn: If True, then the attention scores are output in addition to the tokens.
        :param step: Indicates step-wise auto-regressive decoding is being performed, KV caching is utilized.
        :returns: A tensor of shape (N, T, E) giving the weighted combination of the value vectors according
            to the attention weights computed using the key and query vectors.
        """
        # Implements multiheaded attention
        N, S, E = key.shape  # Source -> creates keys and value vectors
        N, T, E = query.shape  # Target -> creates query vectors

        # We always compute the values below whether we are doing step-wise auto-regressive rollouts or not
        # If not doing step-wise decoding, then this is exactly what we want to set up K, V, Q for attention
        # calcs below. If we are doing step-wise decoding, then the input query is (N, T=1, E). If this is
        # self-attention, then key and value are the same as query, if doing cross-attention, then the key
        # and value vectors are the image patch embeddings

        # Target (queries) reshape (N, T, E) -> (N, H, T, E/H), these are always computed
        Q = self.query(query)  # (N, T, E) @ (E, E) = (N, T, E) convert to query vectors
        Q = Q.reshape(N, T, self.num_heads, self.head_dim)  # Reshape to (N, T, H, E/H)
        Q = torch.permute(Q, (0, 2, 1, 3))  # Reshape to (N, H, T, E/H)

        if step and self.is_cross_attn and self.kv_cache is not None: # If performing step-wise decoding and
            # the KV cache is not empty, then no need to re-compute the KV values again, they're based on the
            # image patches which have not changed since the last work token processed, extract from the cache
            # directly
            K, V = self.kv_cache

        else:  # Otherwise, we will compute the KV values here based on the inputs. This is because:
            # A). step is False so we want to compute them as normal
            # B). step is True, this is cross-attention, the cache is empty so we need to fill it
            # C). step is True, this is self-attention, the cache can be empty or not, either way we need to
            #     update it with the most recent input provided

            # In Cross-Attention: Keys and values are from the image patches, queries are word vectors
            # In Self-Attention: Keys, values, and queries are all from word vectors or image patches

            # Source (keys and values) reshape (N, S, E) -> (N, S, E/H, H)
            K = self.key(key)  # (N, S, E) @ (E, E) = (N, S, E) convert to key vectors
            K = K.reshape(N, S, self.num_heads, self.head_dim)  # Reshape to (N, S, H, E/H)
            K = torch.permute(K, (0, 2, 3, 1))  # Reshape to (N, H, E/H, S)

            V = self.value(value)  # (N, S, E) @ (E, E) = (N, S, E) convert to value vectors
            V = V.reshape(N, S, self.num_heads, self.head_dim)  # Reshape to (N, S, H, E/H)
            V = torch.permute(V, (0, 2, 1, 3))  # Reshape to (N, H, S, E/H)

            if step: # Extract from the cache and update the cache if needed
                if self.is_cross_attn and self.kv_cache is None:
                    # If this is a cross-attention block and the cache is empty, store the computed KV values
                    self.kv_cache = (K, V)
                elif not self.is_cross_attn: # If this is self-attention instead, pull out what prior KV
                    # values, append the new ones, and update the KV cache after appending the new vectors
                    if self.kv_cache is not None: # If non-empty, append the new to the existing KV vectors
                        K = torch.concat((self.kv_cache[0], K), dim=3)  # Concat along the S dimension
                        V = torch.concat((self.kv_cache[1], V), dim=2)  # Concat along the S dimension
                    self.kv_cache = (K, V)  # Update the cache after appending the new token vectors

        # Apply batch multiplication along the last 2 dimensions of these input tensors
        # (N, H, [T, E/H]) @ (N, H, [E/H, S]) = (N, H, T, S)
        att_scores = torch.matmul(Q, K) / math.sqrt(self.head_dim)  # (N, H, T, S)

        if pad_mask is not None:
            # att_scores is (N, H, T, S), fill it with -inf wherever there are padding tokens in the source
            # tokens i.e. along the last dimension, so that they are not attended to by any of the query items
            att_scores.masked_fill_(pad_mask[:, None, None, :], -1e9)

        if attn_mask is not None:
            # attn_mask is (T, S), unsqueeze to create (1, 1, T, S) for broadcasting, then wherever there is a
            # zero in the att_mask, fill it with -inf so that when we do a softmax transform, the wts are zero
            att_scores.masked_fill_(attn_mask.unsqueeze(0).unsqueeze(0) == 0, -1e9)  # (N, H, T, S)

        # Apply softmax normalization to the attention weights along the last dimension i.e. for each input
        # vector, we want to find a normalized weight vector distribution across all the target vectors and
        # also apply dropout regularization
        att_scores = self.attn_drop(F.softmax(att_scores, dim=-1))  # (N, H, T, S)

        # Use the attention weights to compute a weighted value vectors from each head to create a weighted
        # avg of weight vectors each of size E/H for each attention head
        # (N, H, [T, S]) @ (N, H, [S, E/H]) = (N, H, T, E/H)
        Y = torch.matmul(att_scores, V)  # (N, H, T, E/H)

        # Concatenate across the attention heads (N, H, T, E/H) -> (N, T, H, E/H) -> (N, T, E)
        Y = torch.permute(Y, (0, 2, 1, 3)).reshape(N, T, E)

        # Apply the final linear projection: (N, T, E) @ (E, E) = (N, T, E)
        output = self.proj(Y)  # (N, T, E) which matches the original size of the query inputs
        return (output, att_scores) if return_attn else output


class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.2):
        """
        Initializes a simple two-layer feed-forward network with dropout and GeLU activation.

        Input (N, T, E) -> Linear Layer (N, T, ffn_dim) -> Activation -> Dropout -> Linear (N, T, E)

        :param embed_dim: The size of the embedding dimension of input tokens.
        :param ffn_dim: The size of the hidden layer in the feed-forward neural network. Generally set to be
            4x the embed_dim.
        :param dropout: The probability of dropout used during training.
        """
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feedforward network with dropout applied to the inner hidden layer.

        :param x: Batch of input sequences of size (batch, max_len, embed_dim) = (N, T, E)
        :returns: A tensor of the same size (batch, max_len, embed_dim) = (N, T, E).
        """
        x = self.gelu(self.fc1(x))
        x = self.fc2(self.dropout(x))
        return x


########################
### Embedding Layers ###
########################
# TODO: Section marker

class PatchEmbedding(nn.Module):
    """
    THis layer splits an image into patches and projects each patch to an embedding vector and is used as the
    input layer of a Vision Transformer (ViT).
    """

    def __init__(self, img_size: int, patch_size: int, in_channels: int = 3, embed_dim: int = 128):
        """
        Constructs a PatchEmbedding layer.

        :param img_size: An integer representing the height & width of input image (assumes square image).
        :param patch_size: An integer representing height & width of each patch (square patch).
        :param in_channels: The number of input image channels (e.g. 3 for RGB).
        :param embed_dim: The dimension of the linear embedding space i.e. the size of the embedding vectors
            each image patch is turned into.
        """
        super().__init__()

        self.img_size = img_size  # The size of the input images (assumed to be square)
        self.in_channels = in_channels  # The number of input color channels (e.g. 3 for RGB)
        self.patch_size = patch_size  # The size of each patch (height == width, square patches)
        self.embed_dim = embed_dim  # The size of the output embedding vectors for each image patch

        assert img_size % patch_size == 0, "Image dimensions must be divisible by the patch size."

        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * in_channels

        # # For extracting out the image patches
        # C, P = in_channels, patch_size  # Input channels, pixel size of each patch side
        # patch_dim = C * P * P  # The number of pixels in each patch (length when flattened)
        # self.img_patch_conv2d = nn.Conv2d(in_channels=C, out_channels=patch_dim,
        #                                   kernel_size=P, stride=P, bias=False)

        # # Initialize weights to extract patches, create an identity function using conv2d to create patches
        # with torch.no_grad():
        #     self.img_patch_conv2d.weight.zero_()  # Start with all zeros and then selectively add 1s
        #     out_idx = torch.arange(patch_dim)  # Values [0, 1, 2, ... patch_dim - 1]
        #     c_idx = out_idx // (P * P)  # Channel index values [0, 0, ... 1, 1, ..., 2, 2]
        #     # Which patch each idx belongs to:
        #     # [0, ... n_patches-1, 0, ..., n_patches - 1, 0, ..., n_patches - 1]
        #     idx_in_patch = out_idx % (P * P)
        #     i_idx = idx_in_patch // P  # Row
        #     j_idx = idx_in_patch % P  # Col
        #     self.img_patch_conv2d.weight[out_idx, c_idx, i_idx, j_idx] = 1.0

        # A final linear projection layer applied to flattened patches to convert them to the embed_dim
        self.proj = nn.Linear(self.patch_dim, embed_dim)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Converts a batch of input images of size (N, C, H, W) into image patches (N, num_patches, patch_dim)
        where patch_dim = patch_size * patch_size * in_channels. Input images are expected to be square and
        divisible by patch_size. This operation is a deterministic inverse of unpatchify.

        :param imgs: An input image tensor of shape (N, C, H, W) where H == W, square images.
        :returns: A patch embedding tensor of shape (N, num_patches, patch_dim) where num_patches is equal to
            (img_size // patch_size) ** 2. Patches are tiled from top left to bottom right.
        """
        N, C, H, W = imgs.shape  # Unpack the input dimension of the images
        msg = f"Expected image size ({self.img_size}, {self.img_size}), but got ({H}, {W})"
        assert H == self.img_size and W == self.img_size, msg
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "Img not evenly divisible by patch_size"
        h = H // self.patch_size  # Determine how many patches from top to bottom
        w = W // self.patch_size  # Determine how many patches from left to right
        x = imgs.reshape(N, C, h, self.patch_size, w, self.patch_size)
        x = x.permute(0, 2, 4, 3, 5, 1)
        patches = x.reshape(N, h * w, self.patch_size * self.patch_size * C)
        return patches  # (N, num_patches, patch_dim)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Converts a collection of image patches of size (N, num_patches, patch_dim) to a image tensor of size
        (N, C, H, W). Images are expected to be square and divisible by patch_size. This operation is a
        deterministic inverse of patchify.

        :param patches: An input image tensor of shape (N, num_patches, patch_dim).
        :returns: An image tensor of size (N, C, H, W) that reverses the patching operation.
        """
        N, num_patches, patch_dim = patches.shape  # Unpack the dimensions of the input image patches
        H = W = self.img_size  # Images are expected to be square
        C = patch_dim // (self.patch_size * self.patch_size)
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "Img not evenly divisible by patch_size"
        h = H // self.patch_size  # Determine how many patches from top to bottom
        w = W // self.patch_size  # Determine how many patches from left to right
        assert num_patches == h * w, "num_patches does not match expectations"
        x = patches.reshape(N, h, w, self.patch_size, self.patch_size, C)  # (N, h, w, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4)  # (N, C, h, patch_size, w, patch_size)
        imgs = x.reshape(N, C, H, W)  # (N, C, H, W)
        return imgs

    # def patchify_conv2d(self, imgs: torch.Tensor) -> torch.Tensor:
    #     """
    #     Converts a batch of input images (imgs of size (N, C, H, W)) into image patches for each obs of size
    #     (N, num_patches, patch_size * patch_size * in_channels)

    #     :param x: An input image tensor of shape (N, C, H, W) where H == W, square images.
    #     :returns: A collection of image patches for each image of size (N, num_patches, P * P * C) where
    #         num_patches is equal to (img_size // patch_size) ** 2. Patches are tiled from top left to bottom
    #         right.
    #     """
    #     N, C, H, W = imgs.shape
    #     msg = f"Expected image size ({self.img_size}, {self.img_size}), but got ({H}, {W})"
    #     assert H == self.img_size and W == self.img_size, msg
    #     # Divide the image into non-overlapping patches of size (patch_size x patch_size x c)
    #     # (N, C, H, W) -> (N, num_patches, C, patch_size, patch_size)
    #     # E.g. x.shape = ([2, 3, 16, 16]) 2 images, 3 channels, each img is 16 x 16
    #     x = self.img_patch_conv2d(imgs)  # (N, P*P*C, H/P=patch_size, W/P=patch_size)
    #     # patch_dim = P * P * C = patch_size * patch_size * in_channels
    #     patches = x.flatten(start_dim=2).transpose(1, 2)  # (N, num_patches, patch_dim)
    #     return patches  # (N, num_patches, patch_dim)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass for each image in the input batch and converts them into a set of image patch
        embeddings i.e. (N, C, H, W) -> (N, num_patches, embed_dim)

        :param x: An input image tensor of shape (N, C, H, W) where H == W, square images.
        :returns: A patch embedding tensor of shape (N, num_patches, embed_dim) where num_patches is equal to
            (img_size // patch_size) ** 2. Patches are tiled from top left to bottom right.
        """
        return self.proj(self.patchify(imgs))


class PositionalEmbedding(nn.Module):
    """
    Encodes information about the positions of the tokens in the sequence. In this case, the layer has no
    learnable parameters, since it is a simple function of sines and cosines.
    """

    def __init__(self, embed_dim: int, dropout: float = 0.05, max_len: int = 5000):
        """
        Initializes the PositionalEmbedding layer.

        :param embed_dim: The size of the embedding dimension of input tokens.
        :param dropout: The probability of dropout used during training.
        :param max_len: The max length sequence that can be provided i.e. the sinusodal positional embeddings
            will be computed 1x and cached to avoid needless repeat calculations, this max_len determines how
            many positional embeddings to compute and cache.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert embed_dim % 2 == 0, f"embed_dim expected to be even, but got {embed_dim}"
        # Create and array with a batch dimension of 1 which will allow for broadcasting to all examples in
        # the actual batch of data provided to add the positional embeddings the same way to each obs
        pe = torch.zeros(1, max_len, embed_dim)  # (N=1, T, E)
        # Construct the positional embedding array where each row alternatives between sine and cosine and
        # have exponents 0, 0, 2, 2, 4, 4 etc. up to embed_dim.
        exponents = torch.arange(start=0, end=embed_dim, step=2) / embed_dim  # [0, 2, 4, ... embed_dim]
        x = torch.pow(10000, -exponents)  # Pre-compute the values used over and over again
        for i in range(max_len):  # Fill in for each token embedding a positional embedding vector
            pe[0, i, ::2] = torch.sin(i * x)  # Even indices, use sine
            pe[0, i, 1::2] = torch.cos(i * x)  # Odd indices, use cosine

        # Cache these values so that they will be saved along with the model parameters
        self.register_buffer("pos_emb", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        A forward pass through this object adds positional embeddings element-wise to each element in the
        batch provided of input sequences.

        :param x: Batch of input sequences of size (batch, max_len, embed_dim) = (N, T, E)
        :returns: The input sequence embeddings with positional embeddings added element-wise, which has the
            same shape as the original input x i.e. (N, T, E).
        """
        # (N, T, E) + (1, T, E) = (N, T, E) broadcast along the batch dim
        output = self.dropout(x + self.pos_emb[:, :x.shape[1], :])  # (N, T, E)
        return output


##########################
### Transformer Layers ###
##########################
# TODO: Section marker

class TransformerEncoderLayer(nn.Module):
    """
    A single layer of a Transformer encoder, to be used with TransformerEncoder.
    """

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int = 2048, dropout: float = 0.1,
                 ffn_dropout: float = 0.2):
        """
        Constructs a TransformerEncoderLayer instance.

        :param embed_dim: The size of the embedding dimension of input tokens.
        :param num_heads: The number of attention heads to use when computing attention outputs.
        :param ffn_dim: The size of the hidden layer in the feed-forward neural network. Generally set to be
            4x the embed_dim.
        :param dropout: The probability of dropout used during training for the attention components.
        :param ffn_dropout: The probability of dropout used during training for the feed forward network.
        """
        super().__init__()
        self.self_attn = MultiHeadedAttention(embed_dim, num_heads, dropout, False)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim, ffn_dropout)

        self.norm_self_attn = nn.LayerNorm(embed_dim)
        self.norm_ffn = nn.LayerNorm(embed_dim)

        self.dropout_self_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(ffn_dropout)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Computes a forward pass through the encoder layer for the src input, where full bi-directional self
        attention is applied to all tokens. No masking is used, all tokens attend to all others.

        :param src: The sequence input to the encoder layer of shape (N, S, E).
        :returns: A sequence of transformed features of shape (N, S, E).
        """
        x = src  # Name change for typical naming conventions

        # Self-attention sub-block
        residual = x  # Record for the residual connection below
        x = self.norm_self_attn(x)  # Use layer norm pre-processing
        x = self.self_attn(query=x, key=x, value=x, attn_mask=None, pad_mask=None)
        x = self.dropout_self_attn(x)
        x = x + residual  # Residual connection

        # Add a feed-forward block sub-block
        residual = x  # Record for the residual connection below
        x = self.ffn(self.norm_ffn(x))
        x = self.dropout_ffn(x)
        x = x + residual  # Residual connection
        return x


class TransformerEncoder(nn.Module):
    """
    Multi-layered encoder transformer.
    """

    def __init__(self, encoder_layer: TransformerEncoderLayer, num_layers: int):
        """
        Creates a multi-layered transformer by initializing num_layers of encoder_layer instances stacked on
        top of one another i.e. the output of layer (n) is fed into layer (n+1).

        :param encoder_layer: An encoder layer object to be copied N times and stacked sequentially.
        :param num_layers: The number of encoder layers to sequentially stack.
        """
        super().__init__()
        # Create N copies of the same layer configuration, but with different parameters in memory
        self.layers = clone(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass through all layers of the encoder transformer where full bi-directional self
        attention is applied to all tokens. No masking is used, all tokens attend to all others.

        :param src: The sequence input to the encoder layer of shape (N, S, E).
        :returns: A sequence of transformed features of shape (N, S, E).
        """
        x = src  # Name change for typical naming conventions
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerDecoderLayer(nn.Module):
    """
    A single layer of a Transformer decoder for decoding, to be used with TransformerDecoder.
    """

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int = 2048, dropout: float = 0.1,
                 ffn_dropout: float = 0.2):
        """
        Constructs a TransformerDecoderLayer instance.

        :param embed_dim: The size of the embedding dimension of input tokens.
        :param num_heads: The number of attention heads to use when computing attention outputs.
        :param ffn_dim: The size of the hidden layer in the feed-forward neural network. Generally set to be
            4x the embed_dim.
        :param dropout: The probability of dropout used during training. The default is 0.1.
        :param ffn_dropout: The probability of dropout used during training for the feed-forward network
            components of the transformer blocks. Usually a higher amount of dropout is used here e.g. 0.2.
        """
        super().__init__()
        self.self_attn = MultiHeadedAttention(embed_dim, num_heads, dropout, False)
        self.cross_attn = MultiHeadedAttention(embed_dim, num_heads, dropout, True)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim, ffn_dropout)

        self.norm_self_attn = nn.LayerNorm(embed_dim)
        self.norm_cross_attn = nn.LayerNorm(embed_dim)
        self.norm_ffn = nn.LayerNorm(embed_dim)

        self.dropout_self_attn = nn.Dropout(dropout)
        self.dropout_cross_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(ffn_dropout)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor = None,
                pad_mask: torch.Tensor = None, step: bool = False) -> torch.Tensor:
        """
        Computes a forward pass through the decoder layer of the tgt input and tgt_mask.

        :param tgt: The sequence input to the decoder layer of shape (N, T, E).
        :param memory: The sequence input from the last layer of the encoder of shape (N, S, E).
        :param tgt_mask: Masks out parts of the target sequence which should not be attended to by earlier
            token embeddings within the target sequence, is size (T, T).
        :param pad_mask: Denotes which word tokens are padding tokens so that they are masked as well and
            not attended to, this is used by word dropout.
        :param step: Indicates step-wise auto-regressive decoding is being performed, KV caching is utilized.
        :returns: A sequence of transformed features of shape (N, T, E).
        """
        x = tgt  # Name change for typical naming conventions

        # Self-attention sub-block
        residual = x  # Record for the residual connection below
        x = self.norm_self_attn(x)
        x = self.self_attn(query=x, key=x, value=x, attn_mask=tgt_mask, pad_mask=pad_mask, step=step)
        x = self.dropout_self_attn(x)
        x = x + residual  # Residual connection

        # Add a cross-attention sub-block using the encoder output (memory) as the key and value vectors,
        # there is no attention mask here since all input encoder values are visible to all decoder timesteps
        residual = x  # Record for the residual connection below
        x = self.norm_cross_attn(x)
        x = self.cross_attn(query=x, key=memory, value=memory, attn_mask=None, pad_mask=None, step=step)
        x = self.dropout_cross_attn(x)
        x = x + residual  # Residual connection

        # Add a feed-forward block sub-block
        residual = x  # Record for the residual connection below
        x = self.ffn(self.norm_ffn(x))
        x = self.dropout_ffn(x)
        x = x + residual  # Residual connection

        return x  # (N, T, E)

    def clear_cache(self) -> None:
        """
        This method clears out the key-value cache and should be called before starting step-wise decoding.
        """
        self.self_attn.clear_cache()
        self.cross_attn.clear_cache()


class TransformerDecoder(nn.Module):
    """
    Multi-layered decoder transformer.
    """

    def __init__(self, decoder_layer: TransformerDecoderLayer, num_layers: int):
        """
        Creates a multi-layered transformer by initializing num_layers of decoder_layer instances stacked on
        top of one another i.e. the output of layer (n) is fed into layer (n+1).

        :param decoder_layer: A decoder layer object to be copied N times and stacked sequentially.
        :param num_layers: The number of decoder layers to sequentially stack.
        """
        super().__init__()
        # Create N copies of the same layer configuration, but with different parameters in memory
        self.layers = clone(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor = None,
                pad_mask: torch.Tensor = None, step: bool = False) -> torch.Tensor:
        """
        Computes a forward pass through all layers of the decoder transformer, which attends the input tgt
        tokens to themselves (subject to tgt_mask) and also the memory tokens from the source sequence.

        :param tgt: The sequence input to the decoder layer of shape (N, T, E).
        :param memory: The sequence input from the last layer of the encoder of shape (N, S, E).
        :param tgt_mask: Masks out parts of the target sequence which should not be attended to by earlier
            token embeddings within the target sequence.
        :param pad_mask: Denotes which word tokens are padding tokens so that they are masked as well and
            not attended to, this is used by word dropout.
        :param step: Indicates step-wise auto-regressive decoding is being performed, KV caching is utilized.
        :returns: A sequence of transformed features of shape (N, T, E).
        """
        x = tgt
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask, step=step, pad_mask=pad_mask)
        return x

    def clear_cache(self) -> None:
        """
        This method clears out the key-value cache and should be called before starting step-wise decoding.
        """
        for layer in self.layers: # Clear the cache from each transformer layer
            layer.clear_cache()


###########################
### Image Encoder Model ###
###########################
# TODO: Section marker

class ImageEncoder(nn.Module):
    """
    A vision-transformer (ViT) encoder which processes input images by segmenting them into non-overlapping
    image patches, which are then converted into embedding vectors, added to positional embeddings, and passed
    through multiple self-attention layers to output a set of rich image feature representations.
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 512,
                 num_layers: int = 3, num_heads: int = 6, ffn_dim: int = 256, dropout: float = 0.1,
                 ffn_dropout: float = 0.2):
        """
        Initializes a Vision-Transformer (ViT) for processing input images and generating deep latent
        representations of the image patches.

        :param img_size: An integer representing the height & width of input image (assumes square image).
        :param patch_size: An integer representing height & width of each patch (square patch).
        :param in_channels: The number of input image channels (e.g. 3 for RGB).
        :param embed_dim: The dimension of the embedding space i.e. the size of the embedding vectors each
            image patch is turned into. This is also the size of all key, query, and value vectors during
            attention operations.
        :param num_layers: The number of encoder layers to sequentially stack, and 2x the number of language
            model decoder layers.
        :param num_heads: The number of attention heads to use when computing attention outputs.
        :param ffn_dim: The size of the hidden layer in the feed-forward neural network. Generally set to be
            2x or 4x the embed_dim.
        :param dropout: The probability of dropout used during training. The default is 0.1.
        :param ffn_dropout: The probability of dropout used during training for the feed-forward network
            components of the transformer blocks. Usually a higher amount of dropout is used here e.g. 0.2.
        """
        super().__init__()
        # 1). Record input parameters
        self.img_size = img_size  # The size of input images, H == W, images are assumed to be square
        self.patch_size = patch_size  # The size of each patch, H == W, patches are square
        self.in_channels = in_channels  # The number of input image color channels
        self.embed_dim = embed_dim  # The size of the image patch embeddings
        self.num_layers = num_layers  # The number of transformer blocks in the encoder and decoder
        self.num_heads = num_heads  # The number of attention heads
        self.ffn_dim = ffn_dim  # Size of the hidden layer of the feed-forward neural nets
        self.dropout = dropout  # Dropout probability used during training
        self.ffn_dropout = ffn_dropout  # Dropout probability used during training of the FFN components

        # 2). Set up the parameters and layers of vision-transformer i.e. the encoder
        # A patch embedding layer to convert input images into patch embedding vectors
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        # Add learnable positional embeddings for the image patches relative to one another, +1 for CLS token
        self.img_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)  # Apply dropout after adding positional encodings to patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # Add a CLS token for global semantics
        # of size (N=1, n_patches=1, E) to concat with the image patches of size (N, n_patches, E)
        # Construct the vision transformer encoder with self-attention to generate rich image patch embeddings
        encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, ffn_dim, dropout, ffn_dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)

        # 3). Initialize the weights of the network randomly
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize the weights of the network.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode_mae(self, imgs: torch.Tensor, mask_ratio: float = 0.75) -> Tuple[torch.Tensor]:
        """
        Passes a batch of input images (N, C, H, W) through the vision transformer for MAE pre-training.
        The input images are split into non-overlapping patches, normalized, converted to patch embeddings,
        randomly masked, given positional embeddings, and then passed through the encoder model. The deep
        latent representations of the image patches, the masked image patch indices, and the indices to
        restore the origina image patch ordering are output and can be passed directly into the MAE decoder.

        :param imgs: A batch of input images of size (N, C, H, W) to be encoded into deep latent
            representations by the vision transformer.
        :param mask_ratio: The ratio of the image patches to randomly mask out.
        :returns:
            - x_vis_latent: A batch of deep latent representations of visible (unmasked) image pathces of size
                (N, num_patches_vis, embed_dim) which does not include the CLS token.
            - mask: A tensor of size (N, num_patches) which records which image patches were masked with a 1.
            - ids_restore: A tensor of size (N, num_patches) which records the original ordering of the image
                patches for re-construction since x_vis_latent is shuffled and only contains the visible ones.
        """
        # 1). Segment the images into image patches and apply per patch normalizations
        patches = normalize_patches(self.patch_embed.patchify(imgs))  # (N, num_patches, patch_dim)

        # 2). Apply the linear projection to convert the patch pixels into patch embeddings
        patch_embeddings = self.patch_embed.proj(patches)  # (N, num_patches, embed_dim)

        # 3). Apply random masking to the image patches, 1s in the mask tensor denote which were masked
        x_vis, mask, ids_restore = random_masking(patch_embeddings, mask_ratio)

        # 4). Add positional encodings to retain spatial information
        N, num_patches_vis, embed_dim = x_vis.shape
        # img_pos_embed is (1, num_patches + 1, embed_dim) with the first positional embedding at index 0 for
        # the special CLS token so we will add 1 to all the ids_restore values to convert them into the
        # correct image embedding indices (N, num_patches_vis, embed_dim)
        idx = ids_restore[:, :num_patches_vis].unsqueeze(-1).expand(-1, -1, embed_dim) + 1
        # Expand to match the batch dim (1, num_patches + 1, embed_dim) -> (N, num_patches + 1, embed_dim)
        pos_embed = self.img_pos_embed.expand(idx.size(0), -1, -1)
        pos_embed_vis = torch.gather(pos_embed, dim=1, index=idx)
        x_vis = x_vis + pos_embed_vis  # Add the learned positional encodings to the visible image patches

        # 5). Add a CLS token for capturing global semantics
        cls_embed = self.cls_token + self.img_pos_embed[:, 0]  # Add the CLS positional encoding
        x_vis = torch.concat([cls_embed.expand(N, 1, embed_dim), x_vis], dim=1)

        # 6). Pass the visible image patch tokens through the vision-transformer encoder
        x_vis_latent = self.encoder(x_vis)  # (N, num_patches_vis + 1, embed_dim)
        # Drop the CLS token for the decoder, which shouldn't use it
        x_vis_latent = x_vis_latent[:, 1:, :]  # (N, num_patches_vis + 1, E) -> (N, num_patches_vis, E)

        return x_vis_latent, mask, ids_restore

    def encode(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Passes a batch of input images (N, C, H, W) into the vision transformer which is an encoder that
        transforms them into a set of image patch vector embeddings for each observation in the batch with
        one extra for the CLS token i.e. (N, C, H, W) -> (N, num_patches + 1, embed_dim).

        :param imgs: A batch of input images of size (N, C, H, W) to be encoded into deep latent
            representations by the vision transformer.
        :returns: A batch of deep latent representations of the image patches of size (N, num_patches + 1, E).
        """
        # 1). Convert the input image into a sequence of patch vectors
        patch_embeddings = self.patch_embed(imgs)  # (N, num_patches, embed_dim)

        # 2). Add a CLS token for capturing global semantics
        N, n_patches, embed_dim = patch_embeddings.shape
        patch_embeddings = torch.concat([self.cls_token.expand(N, 1, embed_dim), patch_embeddings], dim=1)

        # 3). Add positional encodings to retain spatial information, broadcasts along dim=0
        patch_embeddings = patch_embeddings + self.img_pos_embed  # (N, num_patches + 1, embed_dim)
        patch_embeddings = self.pos_dropout(patch_embeddings)  # Apply dropout

        # 4). Pass the sequence through the vision-transformer encoder
        x = self.encoder(patch_embeddings)  # (N, num_patches + 1, embed_dim)

        return x  # (N, num_patches + 1, embed_dim)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        This is the same as the encode method, see help(self.encode) for details.
        """
        return self.encode(imgs)


################################
### CLIP Image Encoder Model ###
################################
# TODO: Section marker

class CLIPimgEncoder(nn.Module):
    def __init__(self, device):
        """
        Instantiates an encoder model that passes images through the CLIP encoder from OpenAI with frozen
        parameters. This model can be used interchangable with the ImageEncoder class for supervised training
        on the captions dataset.
        """
        super().__init__()
        self.device = device

        self.img_size = 224
        self.patch_size = 16
        self.in_channels = 3
        self.embed_dim = 768
        self.num_patches = 196

        # Load CLIP ViT-B/16 (197 tokens, 768 dim)
        clip_model, _ = clip.load("ViT-B/16", device=device)
        self.visual = clip_model.visual

        # Freeze all parameters of the CLIP encoder model
        for p in self.visual.parameters():
            p.requires_grad = False

        self.visual.eval()

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CLIP encoder to produce deep latent image patch representations.

        :param images: An input image tensor of size (N, 3, 224, 224).
        :returns: A tensor of laten image representations of size: (N, 197, 768) = CLS + 196 patches
        """
        # --- This is basically CLIP's internal visual forward ---
        x = self.visual.conv1(images)            # (N, 768, 14, 14)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)                    # (N, 196, 768)

        cls = self.visual.class_embedding.to(x.dtype)
        cls = cls.unsqueeze(0).expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], dim=1)            # (N, 197, 768)

        x = x + self.visual.positional_embedding
        x = self.visual.ln_pre(x)

        x = x.permute(1, 0, 2)                    # (197, N, 768)
        x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)                    # (N, 197, 768)

        x = self.visual.ln_post(x)
        return x


###############################
### CLIP Text Encoder Model ###
###############################
# TODO: Section marker

class CLIPtextEncoder(nn.Module):
    def __init__(self, device):
        """
        Instantiates a text encoder model that passes text tokens through the CLIP text encoder from OpenAI
        with frozen parameters. This model is used during CLIP-style MAE pre-training.
        """
        super().__init__()

        # Load CLIP ViT-B/16 pre-trained model
        self.model, _ = clip.load("ViT-B/32", device=device)
        self.tokenizer = clip.tokenize # Text tokenizer CLIP expects
        self.embed_dim = self.model.text_projection.shape[1] # 512
        self.device = device

        # Freeze all parameters of the CLIP encoder model
        for p in self.model.parameters():
            p.requires_grad = False

        self.model.eval()

    @torch.no_grad()
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CLIP text encoder to produce deep latent representations of input text.

        :param tokens: A tensor of size (N, T) containing image caption text tokens.
        :returns: A tensor of size (N, E=512) containing deep latent representations of each input caption.
        """
        with torch.no_grad():
            x = self.model.encode_text(tokens)  # (N, E=512)
        return x


######################################
### MAE Pre-Training Decoder Model ###
######################################
# TODO: Section marker

class MAEdecoder(nn.Module):
    """
    A Masked Autoencoder (MAE) decoder model for MAE pre-training of the Vision-Language Transformer image
    encoder.

    Self-supervised pre-training can be used to strengthen the image encoder part of the VLM before
    end-to-end training for image captioning in order to increase overall performance of the model. Using an
    image encoder with strong latent representations of image patches is important before adding a language
    model decoder, otherwise it can be easy to overfit and hard to train both transformer models at the same
    time.

    Masked auto-encoding is a self-supervised training objective where input images are patchified and a
    random subset (e.g. 75%) and masked. The unmasked, visible image patches are passed through the image
    encoder (the vision transformer) and then passed to a small decoder transformer (this model) which uses
    the latent representations of the visible image patches from the encoder to re-construct the masked image
    patches. This reqires the image encoder to learn right latent representations and global image structure
    without labels. It is an efficient and effective technique for down-stream task such as image captioning.
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 512,
                 num_layers: int = 3, num_heads: int = 6, ffn_dim: int = 256, dropout: float = 0.1,
                 ffn_dropout: float = 0.2):
        """
        Initializes a Masked Autoencoder decoder model which attempts to re-construct image patches that have
        been masked to create a self-supervised training objective to pre-train the vision-transformer image
        encoder.

        :param img_size: An integer representing the height & width of input image (assumes square image).
        :param patch_size: An integer representing height & width of each patch (square patch).
        :param in_channels: The number of input image channels (e.g. 3 for RGB).
        :param embed_dim: The dimension of the linear embedding space i.e. the size of the embedding vectors
            each image patch is turned into.
        :param num_layers: The number of encoder layers to sequentially stack.
        :param num_heads: The number of attention heads to use when computing attention outputs.
        :param ffn_dim: The size of the hidden layer in the feed-forward neural network. Generally set to be
            2x or 4x the embed_dim.
        :param dropout: The probability of dropout used during training. The default is 0.1.
        :param ffn_dropout: The probability of dropout used during training for the feed-forward network
            components of the transformer blocks. Usually a higher amount of dropout is used here e.g. 0.2.
        """
        super().__init__()
        # 0). Record input parameters
        self.img_size = img_size  # The size of input images, H == W, images are assumed to be square
        self.patch_size = patch_size  # The size of each patch, H == W, patches are square
        self.in_channels = in_channels  # The number of input image color channels
        self.patch_dim = patch_size * patch_size * in_channels  # The number of pixels per patch
        self.num_patches = (img_size // patch_size) ** 2  # The number of patches in the image patch grid
        self.embed_dim = embed_dim  # The size of the image patch and word embeddings
        self.num_layers = num_layers  # The number of transformer blocks in the encoder and decoder
        self.num_heads = num_heads  # The number of attention heads
        self.ffn_dim = ffn_dim  # Size of the hidden layer of the feed-forward neural nets
        self.dropout = dropout  # Dropout probability used during training
        self.ffn_dropout = ffn_dropout  # Dropout probability used in the FFNs of the transformer blocks

        # Add learnable positional embeddings for the image patches relative to one another
        self.img_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # Initialize a mask token as a vector of learnable parameters which will be used during decoding, we
        # will concat copies of this token with the visible token ouputs from the encoder to get a full set of
        # image patches, then posiitonal embeddings will be added and the full set of image patch tokens
        # passed through the decoder model to predict reconstructions of the masked image patches
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # (1, 1, embed_dim)
        nn.init.normal_(self.mask_token, std=0.02)

        # Use a shallow decoder transformer model that uses full bidirectional self-attention across all
        # tokens to re-constructed the masked image patches, this model will take in (N, num_patches, E) and
        # output a tensor of the same size after applying the transformer operations
        decoder_layer = TransformerEncoderLayer(embed_dim, num_heads, ffn_dim, dropout, ffn_dropout)
        self.decoder = TransformerEncoder(decoder_layer, num_layers)
        # A final projection layer to map the embed_dim latent representations of the masked image patches
        # into pixels i.e. (N, num_patches, embed_dim) -> (N, num_patches, patch_dim)
        self.patch_proj = nn.Linear(self.embed_dim, self.patch_dim)

    def forward(self, x_vis_latent: torch.Tensor, mask: torch.Tensor,
                ids_restore: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass through the MAE decoder model which seeks to re-construct the pixel values of
        the masked image patches using the encoder outputs from the VLM model.

        :param x_vis_latent: A batch of deep latent representations of visible (unmasked) image pathces of size
            (N, num_patches_vis, embed_dim) which does not include the CLS token.
        :param mask: A tensor of size (N, num_patches) which records which image patches were masked with a 1.
        :param ids_restore: A tensor of size (N, num_patches) which records the original ordering of the image
            patches for re-construction since x_vis_latent is shuffled and only contains the visible ones.
        :returns:
            - out: A tensor of size (N, num_patches, patch_dim) containing the pixels of the fully
                reconstructed image where patch_dim = patch_size * patch_size * in_channels and denotes the
                number of pixels in each image patch.
            - mask: A tensor of size (N, num_patches) which records which image patches were masked with a 1.
        """
        N, num_patches, embed_dim = x_vis_latent.shape  # Get the batch size
        msg = f"num_patches expected to be <= {self.num_patches}, but got {num_patches}"
        assert num_patches <= self.num_patches, msg
        msg = f"embed_dim expected to be {self.embed_dim}, but got {embed_dim}"
        assert embed_dim == self.embed_dim, msg
        mask_tokens = self.mask_token.repeat(N, mask.sum(1).max().int(),
                                             1)  # (N, num_patches_masked, embed_dim)
        # Combined the visible tokens with the mask tokens to create a full set of tokens for the decoder
        x_full = torch.cat([x_vis_latent, mask_tokens], dim=1)  # (N, num_patches, embed_dim)
        # However, all the visible tokens are at the front and all the masked ones are in the back, restore
        # the original ordering of the tokens by using ids_restore to unshuffle (N, num_patches, embed_dim)
        x_full = torch.gather(x_full, dim=1,
                              index=ids_restore.unsqueeze(-1).expand(-1, -1, self.embed_dim))
        x_full = x_full + self.img_pos_embed  # Add positional encodings to all image patch tokens
        # Then pass the full set of image patch tokens through the decoder model and project to pixels
        out = self.patch_proj(self.decoder(x_full))  # (N, num_patches, patch_dim), patch_dim = P * P * C
        return out, mask  # (N, num_patches, patch_dim), (N, num_patches) with 1s for the masked img patches


##############################
### Language Decoder Model ###
##############################
# TODO: Section marker

class LanguageDecoder(nn.Module):
    """
    A transformer-based language decoder model for image captioning.

    This model processes the sequence of word tokens already output by this model by adding a positional
    embedding and passing the results through a sequence of transformer layers that compute causal (masked)
    self-attention and cross attention to the outputs of the vision transformer encoder model to incorporate
    information from the image patches. Outputs a probability distribution over the vocabulary space.
    """

    def __init__(self, img_feature_shape: Tuple[int] = (512, 196), embed_dim: int = 512, num_layers: int = 3,
                 num_heads: int = 6, ffn_dim: int = 256, dropout: float = 0.1, ffn_dropout: float = 0.2,
                 sp_model=None, max_len: int = 50):
        """
        Initializes a Language Transformer Model for processing prior word tokens and deep latent
        representations of the image patches to output predicted next words.

        :param img_feature_shape: The size of the image features per image i.e. the number of patches, (not
            including the CLS token) and the embedding dimension size of the image encoder model.
        :param embed_dim: The dimension of the word token embedding space. This is also the size of all key,
            query, and value vectors during attention operations.
        :param num_layers: The number of encoder layers to sequentially stack, and 2x the number of language
            model decoder layers.
        :param num_heads: The number of attention heads to use when computing attention outputs.
        :param ffn_dim: The size of the hidden layer in the feed-forward neural network. Generally set to be
            2x or 4x the embed_dim.
        :param dropout: The probability of dropout used during training. The default is 0.1.
        :param ffn_dropout: The probability of dropout used during training for the feed-forward network
            components of the transformer blocks. Usually a higher amount of dropout is used here e.g. 0.2.
        :param sp_model: A sentencepiece.SentencePieceProcessor model used to convert between tokens and
            strings.
        :param max_len: An integer denoting the max length of a caption decoding during sampling.
        """
        super().__init__()
        # 1). Record input parameters
        self.num_patches, self.patch_embed_dim = img_feature_shape  # Unpack and record
        self.embed_dim = embed_dim  # The size of the word embeddings
        self.num_layers = num_layers  # The number of transformer blocks in the encoder and decoder
        self.num_heads = num_heads  # The number of attention heads
        self.ffn_dim = ffn_dim  # Size of the hidden layer of the feed-forward neural nets
        self.dropout = dropout  # Dropout probability used during training
        self.ffn_dropout = ffn_dropout  # Dropout probability used in the FFNs of the transformer blocks
        self.max_len = max_len  # Max caption decode output length
        self.vocab_size = len(sp_model)  # This in part determines the number of parameters in the model
        self.sp_model = sp_model
        # Record the word indices of special word tokens
        self._pad, self._start, self._end = sp_model.pad_id(), sp_model.bos_id(), sp_model.eos_id()

        # 2). Set up the language-transformer decoder model parameters and layers
        # Projects the output of the encoder into features to be input into the decoder, applies layer norm
        self.visual_projection = nn.Sequential(nn.Linear(self.patch_embed_dim, embed_dim),
                                               nn.LayerNorm(embed_dim))
        # Add learnable positional embeddings for the image patches relative to one another, +1 for CLS
        # for use by the decoder when recieving encoder outputs going into cross-attention
        self.img_pos_embed_decoder = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        # An embedding layer to convert word indices to word vectors
        self.word_idx_embed = nn.Embedding(self.vocab_size, embed_dim, padding_idx=self._pad)
        # A positional embedding layer for the positional indices of the work tokens
        self.word_pos_embed = PositionalEmbedding(embed_dim, 0.05, max_len)
        # Construct the language decoder with cross-attention to the image patches embeddings from the encoder
        decoder_layer = TransformerDecoderLayer(embed_dim, num_heads, ffn_dim, dropout, ffn_dropout)
        self.decoder = TransformerDecoder(decoder_layer, num_layers)
        # A final projection layer from the outputs of the decoder to the vocab space
        self.vocab_proj = nn.Linear(embed_dim, self.vocab_size)
        # Tie together the weights of the vocab projection layer with the workd token embedding layer to
        # reduce the overall parameter count and mitigate potential overfitting in the language decoder
        self.vocab_proj.weight = self.word_idx_embed.weight

        # A dummy param to tracking the device of this model during later calls
        self.register_buffer('device_param', torch.empty(0))

        # 3). Initialize the weights of the network randomly
        self.apply(self._init_weights)

        # 4). Set up a local cache
        self.img_features_cache = None

    def _init_weights(self, module):
        """
        Initialize the weights of the network.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def decode(self, img_features: torch.Tensor, word_idx_seq: torch.Tensor) -> torch.Tensor:
        """
        Given a batch of deep latent image patch representations from the encoder (N, num_patches, patch_E)
        and a batch of word token index integers (N, T), this method computes a forward decode operation at
        each time step of the word token sequences to output a distribution of logit scores over the
        vocabulary of size (N, T, V).

        The word indices of the word_idx_seq input are converted to vector embeddings, then positional
        embeddings are added, and then the embeddings are passed through the decoder transformer with causal
        masked self-attention and cross-attention with respect to the img_features to produce a distribution
        over the vocabulary at each timestep corresponding to the model's forward-looking, next-word
        prediction probabilities.

        :param img_features: Deep latent representations of image patches from the encoder model of size
            (N, num_patches, patch_embed_dim).
        :param word_idx_seq: A tensor containing a batch of word index sequences i.e. integers of size
            (N, T) where T is the max length among the sequences.
        :returns: A tensor of logits over the vocabulary denoting the model's next-word predictions at each
            input timestep of word_idx_seq.
        """
        N, T = word_idx_seq.shape  # N examples (batch size), each of at most length T

        # 1). During training, apply word dropout to mitigate memorization of word phrases and encourage the
        # mode to attend to the image features from the vision encoder instead
        if self.training: # Check if the model is in training mode, if so then apply the word dropout
            word_idx_seq = word_dropout(word_idx_seq, 0.0, self._pad)

        # 2). Create a mask (tgt_mask) for masking out attention scores from early words to latter words in
        # the captions sequence i.e. prevent lookahead, i.e. required for causal self-attention
        # Lower right triangular matrix of 1s to prevent lookahead
        tgt_mask = torch.tril(torch.ones(T, T)).to(self.device_param.device) # (T, T) for causal masking
        pad_mask = (word_idx_seq == self._pad)  # Where True, we have padding (N, T), with word dropout we
        # don't want to attend to padding tokens in the middle of the sequence

        # 3). Convert the captions (encoded as integers) into word embeddings using the embedding layer
        captions_emb = self.word_idx_embed(word_idx_seq)  # (batch_size, max_len, embed_dim) = (N, T, E)

        # 4). Add positional encodings to the captions embeddings
        captions_emb = self.word_pos_embed(captions_emb)  # (batch_size, max_len, embed_dim) = (N, T, E)

        # 5). Apply a projection to the img_features before feeding them into the cross-attention mechanism
        # nn.Linear(patch_embed_dim, embed_dim): (batch_size, img_feat_dim) -> (batch_size, wordvec_dim)
        # then also apply layer norm before cross-attention
        img_features = self.visual_projection(img_features)  # (batch_size, num_patches, embed_dim)
        # Also add in the positional encodings again so that the decoder has the spatial info of the patches
        img_features += self.img_pos_embed_decoder  # (N, num_patches + 1, embed_dim), broadcasts along dim=0

        # 6). Pass the captions embeddings through a transformer block so that each word can attend to the
        # prior caption words (causal self-attention) and also to all the image features (cross-attention)
        x = self.decoder(captions_emb, img_features, tgt_mask=tgt_mask, pad_mask=pad_mask)  # (N, T, E)

        # 7). Map the transformer outputs to the vocab dim for a final word prediction at each time step
        # (batch_size, T, embed_dim) @ (embed_dim, vocab_size) = (batch_size, T, vocab_size)
        logit_scores = self.vocab_proj(x)

        return logit_scores  # (N, T, V) decoder_outputs

    def clear_cache(self) -> None:
        """
        This method clears out the key-value cache and should be called before starting step-wise decoding.
        """
        self.img_features_cache = None
        self.decoder.clear_cache()

    def decode_step(self, img_features: torch.Tensor, word_idx_seq: torch.Tensor) -> torch.Tensor:
        """
        Same as decode but useful for auto-regressive roll out sampling
        """
        N, T = word_idx_seq.shape  # N examples (batch size), each of at most length T

        # 1). Convert the captions (encoded as integers) into word embeddings using the embedding layer
        captions_emb = self.word_idx_embed(word_idx_seq)  # (batch_size, max_len, embed_dim) = (N, T, E)

        # 2). Add positional encodings to the captions embeddings
        captions_emb = self.word_pos_embed(captions_emb)  # (batch_size, max_len, embed_dim) = (N, T, E)

        if img_features is not None: # Only re-compute and update if not None
            # 3). Apply a projection to the img_features before feeding them into the cross-attention layer
            # nn.Linear(patch_embed_dim, embed_dim): (batch_size, img_feat_dim) -> (batch_size, wordvec_dim)
            # then also apply layer norm before cross-attention
            img_features = self.visual_projection(img_features)  # (batch_size, num_patches, embed_dim)
            # Also add in the positional encodings again so that the decoder has the spatial info of patches
            img_features += self.img_pos_embed_decoder  # (N, num_patches + 1, embed_dim), broadcasts dim=0
            self.img_features_cache = img_features # Cache for later re-use, no need to re-run every time
        else: # Otherwise, rely on what's already in the cache instead of re-computing
            assert self.img_features_cache is not None, "img_features_cache is empty!"
            img_features = self.img_features_cache

        # 4). No attention mask needed because we're passing in the last token embedding vector only which
        #     is allowed to see all others prior and itself so no masking needed, we're already using casual
        #     attention by the step-wise rollout proceadure
        tgt_mask = None

        # 5). Pass the captions embeddings through the transformer blocks so that each word can attend to the
        # prior caption words (causal self-attention) and also to all the image features (cross-attention)
        x = self.decoder(captions_emb[:, -1:, :], img_features, tgt_mask, step=True)  # (N, T=1, E)

        # 6). Map the transformer outputs to the vocab dim for a final word prediction at each time step
        # (batch_size, T=1, embed_dim) @ (embed_dim, vocab_size) = (batch_size, T=1, vocab_size)
        logit_scores = self.vocab_proj(x)

        return logit_scores  # (N, T=1, V) decoder_outputs

    def forward(self, img_features: torch.Tensor, word_idx_seq: torch.Tensor) -> torch.Tensor:
        """
        This is the same as the decode method, see help(self.decode) for details.
        """
        return self.decode(img_features, word_idx_seq)


###############################################
### Vision-Language Transformer (VLM) Model ###
###############################################
# TODO: Section marker

class VisionLanguageModel(nn.Module):
    """
    A Vision-Language Transformer model for image caption generation.

    Comprised of:
        Encoder: A vision-transformer (ViT) encoder which processes input images by segmenting them into
        non-overlapping image patches, which are then converted into embedding vectors, added to positional
        embeddings, and passed through multiple self-attention layers to output a set of rich image feature
        representations.

        Decoder: A language transformer decoder which processes the sequence of tokens already output by the
        model by adding a positional embedding, and passing the results through a sequence of transformer
        layers that compute causal (masked) self-attention and cross attention to the outputs of the vision
        transformer encoder model to incorporate information from the image patches. Outputs a probability
        distribution over the vocabulary space.
    """

    def __init__(self, encoder: ImageEncoder, decoder: LanguageDecoder):
        """
        Initializes a Vision-Language Transformer model for image captioning.

        Input images are first processed by the vision transformer encoder, and then passed to the language
        model transformer decoder for word token index predictions.

        :param encoder: An image encoder model instance.
        :param decoder: A language decoder model instance.
        """
        super().__init__()
        self.encoder, self.decoder = encoder, decoder  # Record pointers to model components
        A, B = encoder.embed_dim, decoder.patch_embed_dim  # Check dimensional match between models
        assert A == B, f"encoder.embed_dim ({A}) must be equal to decoder.patch_embed_dim ({B})"

        # Record pointers to the sp tokenizer model from the decoder for quick reference
        self.sp_model = decoder.sp_model
        self._pad = self.sp_model.pad_id()
        self._start, self._end = self.sp_model.bos_id(), self.sp_model.eos_id()
        self.max_len = self.decoder.max_len

        # A dummy param to tracking the device of this model during later calls
        self.register_buffer('device_param', torch.empty(0))

    def forward(self, imgs: torch.Tensor, word_idx_seq: torch.Tensor) -> torch.Tensor:
        """
        Given an input batch of images (N, C, H, W) and text captions encoded as word token integer indices
        (N, T), this method computes a full forward pass through the model and outputs a tensor of logits
        for each timestep of the word sequence denoting the distribution of next-word predictions over the
        vocabulary at each time step of size (N, T, V) where V is the size of the vocabulary. This method
        is only used in training

        :param imgs: A batch of input images of size (N, C, H, W).
        :param word_idx_seq: A tensor containing a batch of word index sequences i.e. integers of size
            (N, T) where T is the max length among the sequences.
        :returns: A tensor of logits over the vocabulary denoting the model's next-word predictions at each
            input timestep of word_idx_seq.
        """
        return self.decoder(self.encoder(imgs), word_idx_seq)

    def compute_loss(self, decoder_outputs: torch.Tensor, target_word_idx: torch.Tensor, eps: float = 0.1
                     ) -> torch.float:
        """
        Computes a cross-entropy loss for model training using the outputs from the decoder (logits) and a
        tensor of target word indices expected. This method returns the negative log-likelihood of all the
        target caption words by extracting out the log-likelihood of each correct word expected from the
        predicted distrubtion over the vocab at each timestep.

        :param decoder_outputs: A torch.Tensor of size (N, T, V) where N=batch_size, T=max length of a
            tokensized caption of the ground-truth annotations and V is the size of the vocabulary containing
            logit scores from the model at each timestep predicting the most likely next word to follow.
        :param target_word_idx: A torch.Tensor of size (N, T) containing the ground-truth caption token
            indices for each image in the training batch.
        :param eps: A smoothing parameter used during decoding to smooth the probability distribution
            predicted by the model over the vocabulary space.
        :returns: The sum of negative log-likelihood across all timesteps in the decoding. Log-likelihood
            values are not computed for the padding tokens.
        """
        N, T, V = decoder_outputs.shape # Unpack shape of the inputs

        # Flatten the batch and sequence dims for passing the data into F.cross_entropy, make sure to align
        # the predictions with the ground truths. The first token of target_word_idx is <s> so we skip that
        # one and take the rest. The decoder_outputs are the predictions after observing the true tokens
        # up through index i so they're 1 index position ahead of the target_word_idx values
        logits_flat = decoder_outputs[:, :-1, :].reshape(-1, V)  # (N * (T-1), V)
        targets_flat = target_word_idx[:, 1:].reshape(-1)  # (N * (T-1))

        # Create a mask to ignore the padding tokens
        mask = (targets_flat != self.decoder._pad) # True = No padding token, False = Padding token
        logits_flat = logits_flat[mask]
        targets_flat = targets_flat[mask]

        # Use PyTorch's built-in cross-entropy with label smoothing to compute the loss
        loss = F.cross_entropy(logits_flat, targets_flat, label_smoothing=eps)
        return loss

    def sample(self, imgs: torch.Tensor, max_len: int = None, return_strings: bool = True,
               track_gradients: bool = False, temp: float = 0.0) -> Union[torch.Tensor, List[str]]:
        """
        Given an input batch of images, this method uses a greedy decoding approach to predict the image
        caption for each and outputs a np.ndarray of vocab word indices for each image in the image batch
        detailing what image caption would be predicted by the model for each or a list of strings
        (sentences).

        :param imgs: A batch of input images of size (N, C, H, W).
        :param max_len: The max length of any caption generated.
        :param return_strings: If False, a tensor of ints of size (N, T <= max_len) is returned denoting
            the predicted word token indices for the decoded captions. If True, then this method returns a
            list of length N containing strings for each caption.
        :param track_gradients: If set to True, then gradients are tracked through the returned log probs.
        :param temp: A temperature parameter used to sample from the predicted distribution of probabilities
            of the next token at each timestep. Set to 0 for greedy decoding, set higher for more uniform
            sampling i.e. more creative and varied responses.
        :returns: A np.ndarray of size (N, T) containing the integer word indices of the predicted captions
            or a list of caption sentences (strings).
        """
        assert temp >= 0, f"temp must be a value >= 0, got {temp}"
        max_len = self.decoder.max_len if max_len is None else max_len
        device = imgs.device # Infer the correct device from the images passed in

        was_training = self.training
        self.train() if track_gradients else self.eval() # Switch to training mode if gradient tracking
        self.decoder.clear_cache() # Clear out the KV and image features cache before running

        with torch.set_grad_enabled(track_gradients):

            N = imgs.shape[0]  # The number of images in the batch
            img_features = self.encoder(imgs)  # Process the images and generate image features (N, S, E)

            # Create an empty captions tensor to record the outputs, where all tokens are NULL initially
            captions = torch.full((N, max_len), self._pad, dtype=torch.long,
                                  device=device)  # Record ints (N, max_len)

            # Create a starting partial caption to begin to decoding sequence for each using the start token
            partial_captions = torch.full((N, 1), self._start, dtype=torch.long, device=device)
            log_probs_sum = torch.zeros(N, dtype=torch.float).to(device)

            # Record True for each sentence in the batch if it has reached the </s> end token
            eos_mask = torch.zeros(N, dtype=bool, device=device)
            for t in range(max_len):  # Run the decoding time steps to fill in the caption word idx
                # Predict the next token index for all images in the input batch
                logits = self.decoder.decode_step((img_features if t == 0 else None), partial_captions)
                logits = logits[:, -1, :]  # Use only the last timestep's embed values to make the next
                # word token prediction (N, vocab_size)

                if temp == 0.0: # Then use the argmax for greedy decoding
                    # Choose the most likely word index from the vocabulary for each image, (N, V) -> (N, )
                    word_indices = torch.argmax(logits, axis=1)  # (N, )
                else: # Otherwise randomly sample from the distribution of predicted next word tokens
                    # Apply the temperature scaling and sample
                    probs = F.softmax(logits / temp, dim=-1) # (N, V), apply softmax along vocab dim
                    word_indices = torch.multinomial(probs, num_samples=1).squeeze(1)  # (N, )

                # Update the captions output and the current partial captions tensor
                captions[:, t] = word_indices
                captions[eos_mask, t] = self._pad  # Replace with the padding token beyond </s>
                log_probs = torch.log_softmax(logits, dim=-1)  # (N, V) log probs for each next token pred
                eos_mask = eos_mask | (captions[:, t] == self._end)  # Update the end of sentence bool flags
                selected_logprob = log_probs.gather(1, word_indices.unsqueeze(1)).squeeze(1)   # (N,)
                log_probs_sum += selected_logprob * (~eos_mask).float() # Add the log probs of these new token
                # predictions for each caption, zero out the entires that are the end or padding tokens
                if eos_mask.sum() == len(eos_mask):  # Stop early if all outputs have reached their </s> token
                    break

                word_indices = word_indices.unsqueeze(1)  # (N, 1)
                partial_captions = torch.cat([partial_captions, word_indices], dim=1)  # (N, t) -> (N, t+1)

        self.train() if was_training else self.eval()
        self.decoder.clear_cache() # Clear out the KV and image features cache ater running

        if return_strings is False:  # Return as a np.ndarray
            return captions, log_probs_sum  # (N, T) = (batch_size, max_size), (N, ) float values
        else:  # Return as a list of strings, process each sentence into plaintext and the sum of log probs
            return [decode_caption(x, self.sp_model) for x in captions], log_probs_sum

    def _beam_search(self, img: torch.Tensor, max_len: int = None, return_string: bool = True,
                     beam_size: int = 5, alpha: float = 0.7) -> Union[np.ndarray, List[str]]:
        """
        Given a single input image of size (1, C, H, W), this method uses beam search to predict the image
        caption and outputs a np.ndarray of vocab word indices for the image or a single string that is the
        decoded word tokens.

        :param img: A single input image of size (1, C, H, W).
        :param max_len: The max length of any caption generated.
        :param return_strings: If False, a tensor of ints of size (N, T <= max_len) is returned denoting
            the predicted word token indices for the decoded captions. If True, then this method returns a
            list of length N containing strings for each caption.
        :param beam_size: The number of hypothesis caption roll outs to maintain at any given time.
        :param alpha: A length normalization parameter that penalizes very short caption decodings.
        :returns: A np.ndarray of size (1, T) containing the integer word indices of the predicted caption
            or a caption sentence as a string.
        """
        assert img.shape[0] == 1, "Beam search expects batch size = 1"

        max_len = self.max_len if max_len is None else max_len
        self.eval()  # Switch to eval mode to turn off dropout

        with torch.no_grad():  # Used for inference, no gradient tracking required
            img_features = self.encoder(img)  # (1, S, E)

            # Each beam contains: (token_ids, log_prob, finished_flag)
            beams = [(torch.tensor([[self._start]], device=self.device_param.device), 0.0, False)]

            for i in range(max_len):
                new_beams = []

                for tokens, log_prob, finished in beams:  # Extend the existing beams by 1 more token
                    if finished:  # Nothing further to add if this beam has finished
                        new_beams.append((tokens, log_prob, True))
                        continue

                    logits = self.decoder(img_features, tokens)  # Run the decoder to predict the next token
                    logits = logits[:, -1, :]  # Get the predicted logits over the vocab (1, V)
                    log_probs = torch.log_softmax(logits, dim=-1)  # Softmax over the logits

                    topk_log_probs, topk_ids = torch.topk(log_probs, beam_size, dim=-1)

                    for k in range(beam_size):
                        next_token = topk_ids[0, k].view(1, 1)
                        next_log_prob = log_prob + topk_log_probs[0, k].item()

                        new_tokens = torch.cat([tokens, next_token], dim=1)  # Append new predicted token
                        finished_flag = (next_token.item() == self._end)  # Check if </s> predicted

                        new_beams.append((new_tokens, next_log_prob, finished_flag))

                # Keep the top-k beams sorted by sum(log_prob) / (length)^alpha
                beams = sorted(new_beams, key=lambda x: x[1] / (len(x[0]) ** alpha), reverse=True)[:beam_size]

                if all(finished for _, _, finished in beams):  # Terminate early if all beams finished
                    break

            best_tokens = beams[0][0].squeeze(0).cpu().numpy()  # Already sorted by log_prob / length ** alpha

        self.train() # Make sure the model is in training mode before exiting
        return decode_caption(best_tokens, self.sp_model) if return_string else best_tokens

    def beam_search(self, imgs: torch.Tensor, max_len: int = None, return_strings: bool = True,
                    beam_size: int = 5, alpha: float = 0.7, ) -> Union[np.ndarray, List[str]]:
        """
        Given an input batch of images, this method uses beam search to predict the image caption for each
        and outputs a np.ndarray of vocab word indices for each image in the image batch detailing what
        image caption would be predicted by the model for each or a list of strings (sentences).

        :param imgs: A batch of input images of size (N, C, H, W).
        :param max_len: The max length of any caption generated.
        :param return_strings: If False, a tensor of ints of size (N, T <= max_len) is returned denoting
            the predicted word token indices for the decoded captions. If True, then this method returns a
            list of length N containing strings for each caption.
        :param beam_size: The number of hypothesis caption roll outs to maintain at any given time.
        :param alpha: A length normalization parameter that penalizes very short caption decodings.
        :returns: A np.ndarray of size (N, T) containing the integer word indices of the predicted captions
            or a list of caption sentences (strings).
        """
        was_training = self.training
        outputs = [self._beam_search(imgs[i:i + 1], beam_size, alpha, max_len, return_strings)
                   for i in range(imgs.shape[0])]
        self.train() if was_training else self.eval()
        return outputs
