"""
This module defines Transformer models and the layers used to create them.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
import copy
from utils import decode_caption
from typing import List, Union


def clone(module: nn.Module, N: int):
    "Produces N identical layers of the input module."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


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

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Initializes a MultiHeadedAttention layer. The hidden dimension of the key, query, and value vectors
        is set equal to embed_dim as per the usual convention.

        :param embed_dim: The size of the embedding dimension of input tokens.
        :param num_heads: The number of attention heads to use when computing attention outputs.
        :param dropout: The probability of dropout used during training.
        """
        super().__init__()

        # Create linear transforms from input (token embeddings + positional embeddings) to key, query, and
        # value vectors using a nn.Linear layer to perform the matrix multiplication of learned weights
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.proj = nn.Linear(embed_dim, embed_dim)  # A final projection layer after computing attention
        self.attn_drop = nn.Dropout(dropout)

        self.n_head = num_heads
        self.emd_dim = embed_dim  # Also used as the size of the key, query, and value vectors
        # The size of each key, query, and value vector within attention head
        msg = f"emd_dim={self.emd_dim} must be a multiple of n_head={self.n_head}"
        assert self.emd_dim % self.n_head == 0, msg
        self.head_dim = self.emd_dim // self.n_head

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: torch.Tensor = None) -> torch.Tensor:
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
        :returns: A tensor of shape (N, T, E) giving the weighted combination of the value vectors according
            to the attention weights computed using the key and query vectors.
        """
        # Implements multiheaded attention
        N, S, E = key.shape  # Source -> creates keys and value vectors
        N, T, E = query.shape  # Target -> creates query vectors

        # Source (keys and values) reshape (N, S, E) -> (N, S, E/H, H)
        K = self.key(key)  # (N, S, E) @ (E, E) = (N, S, E) convert to key vectors
        K = K.view(N, S, self.n_head, self.head_dim)  # Reshape to (N, S, H, E/H)
        K = torch.permute(K, (0, 2, 3, 1))  # Reshape to (N, H, E/H, S)

        V = self.value(value)  # (N, S, E) @ (E, E) = (N, S, E) convert to value vectors
        V = V.view(N, S, self.n_head, self.head_dim)  # Reshape to (N, S, H, E/H)
        V = torch.permute(V, (0, 2, 1, 3))  # Reshape to (N, H, S, E/H)

        # Target (queries) reshape (N, T, E) -> (N, H, T, E/H)
        Q = self.query(query)  # (N, T, E) @ (E, E) = (N, T, E) convert to query vectors
        Q = Q.view(N, T, self.n_head, self.head_dim)  # Reshape to (N, T, H, E/H)
        Q = torch.permute(Q, (0, 2, 1, 3))  # Reshape to (N, H, T, E/H)

        # Apply batch multiplication along the last 2 dimensions of these input tensors
        # (N, H, [T, E/H]) @ (N, H, [E/H, S]) = (N, H, T, S)
        att_scores = torch.matmul(Q, K) / math.sqrt(self.head_dim)  # (N, H, T, S)

        if attn_mask is not None:
            # attn_mask is (T, S), unsqueeze to create (1, 1, T, S) for broadcasting, then wherever there is a
            # zero in the att_mask, fill it with -inf so that when we do a softmax transform, the wts are zero
            att_scores.masked_fill_(attn_mask.unsqueeze(0).unsqueeze(0) == 0, -torch.inf)  # (N, H, T, S)

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
        return output


class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
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
        Forward pass through the feedforward network.

        :param x: Batch of input sequences of size (batch, max_len, embed_dim) = (N, T, E)
        :returns: A tensor of the same size (batch, max_len, embed_dim) = (N, T, E).
        """
        x = self.gelu(self.fc1(x))
        x = self.dropout(self.fc2(x))
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

        # For extracting out the image patches
        self.img_patch_conv2d = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim,
                                          kernel_size=patch_size, stride=patch_size)

    #     # A final linear projection layer applied to flattened patches to convert them to the embed_dim
    #     self.proj = nn.Linear(self.patch_dim, embed_dim)

    # def forward_legacy(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     LEGACY VERSION - NOT USED IN PRODUCTION
    #     Computes a forward pass for each image in the input batch and converts them into a set of image patch
    #     embeddings i.e. (N, C, H, W) -> (N, num_patches, embed_dim)

    #     :param x: An input image tensor of shape (N, C, H, W) where H == W, square images.
    #     :returns: A patch embedding tensor of shape (N, num_patches, embed_dim) where num_patches is equal to
    #         (img_size // patch_size) ** 2. Patches are tiled from top left to bottom right.
    #     """
    #     N, C, H, W = x.shape
    #     msg = f"Expected image size ({self.img_size}, {self.img_size}), but got ({H}, {W})"
    #     assert H == self.img_size and W == self.img_size, msg
    #     # Divide the image into non-overlapping patches of size (patch_size x patch_size x c)
    #     # (N, C, H, W) -> (N, num_patches, C, patch_size, patch_size)
    #     # E.g. x.shape = ([2, 3, 16, 16]) 2 images, 3 channels, each img 16 x 16
    #     patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
    #     # E.g. patches.shape = ([2, 3, 2, 2, 8, 8]) 2 images, 3 channels, 2 + 2 = 4 patches, 8x8 patch size
    #     patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(N, -1, C, self.patch_size, self.patch_size)
    #     # E.g. patches.shape = ([2, 4, 3, 8, 8]) 2 images, 4 patches, 3 channels, 8x8 patch size
    #     # Flatten and pass through the final projection layer to turn these into embedding vectors
    #     patch_embeddings = self.proj(torch.flatten(patches, start_dim=2))
    #     return patch_embeddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass for each image in the input batch and converts them into a set of image patch
        embeddings i.e. (N, C, H, W) -> (N, num_patches, embed_dim)

        :param x: An input image tensor of shape (N, C, H, W) where H == W, square images.
        :returns: A patch embedding tensor of shape (N, num_patches, embed_dim) where num_patches is equal to
            (img_size // patch_size) ** 2. Patches are tiled from top left to bottom right.
        """
        N, C, H, W = x.shape
        msg = f"Expected image size ({self.img_size}, {self.img_size}), but got ({H}, {W})"
        assert H == self.img_size and W == self.img_size, msg
        # Divide the image into non-overlapping patches of size (patch_size x patch_size x c)
        # (N, C, H, W) -> (N, num_patches, C, patch_size, patch_size)
        # E.g. x.shape = ([2, 3, 16, 16]) 2 images, 3 channels, each img 16 x 16
        x = self.img_patch_conv2d(x)  # (N, embed_dim, H/P=patch_size, W/P=patch_size)
        patch_embeddings = x.flatten(start_dim=2).transpose(1, 2)  # (N, num_patches, embed_dim)
        return patch_embeddings


class PositionalEmbedding(nn.Module):
    """
    Encodes information about the positions of the tokens in the sequence. In this case, the layer has no
    learnable parameters, since it is a simple function of sines and cosines.
    """

    def __init__(self, embed_dim: int, dropout: float = 0.1, max_len: int = 5000):
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

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int = 2048, dropout: float = 0.1):
        """
        Constructs a TransformerEncoderLayer instance.

        :param embed_dim: The size of the embedding dimension of input tokens.
        :param num_heads: The number of attention heads to use when computing attention outputs.
        :param ffn_dim: The size of the hidden layer in the feed-forward neural network. Generally set to be
            4x the embed_dim.
        :param dropout: The probability of dropout used during training.
        """
        super().__init__()
        self.self_attn = MultiHeadedAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim, dropout)

        self.norm_self = nn.LayerNorm(embed_dim)
        self.norm_ffn = nn.LayerNorm(embed_dim)

        self.dropout_self = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Computes a forward pass through the encoder layer of the src input and src_mask.

        :param src: The sequence input to the encoder layer of shape (N, S, E).
        :param src_mask: Masks out parts of the source sequence which should not be attended to by earlier
            token embeddings within the target sequence, of shape (S, S).
        :returns: A sequence of transformed features of shape (N, S, E).
        """
        # Self-attention sub-block (reference implementation)
        shortcut = src  # Record for the residual connection below
        src = self.self_attn(query=src, key=src, value=src, attn_mask=src_mask)
        src = self.dropout_self(src)
        src = src + shortcut  # Residual connection
        src = self.norm_self(src)

        # Add a feed-forward block sub-block
        shortcut = src  # Record for the residual connection below
        src = self.ffn(src)
        src = self.dropout_ffn(src)
        src = src + shortcut  # Residual connection
        src = self.norm_ffn(src)

        return src


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

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Computes a forward pass through all layers of the encoder transformer.

        :param src: The sequence input to the encoder layer of shape (N, S, E).
        :param src_mask: Masks out parts of the source sequence which should not be attended to by earlier
            token embeddings within the target sequence, of shape (S, S).
        :returns: A sequence of transformed features of shape (N, S, E).
        """
        x = src
        for layer in self.layers:
            x = layer(x, src_mask=src_mask)
        return x


class TransformerDecoderLayer(nn.Module):
    """
    A single layer of a Transformer decoder for decoding, to be used with TransformerDecoder.
    """

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int = 2048, dropout: float = 0.1):
        """
        Constructs a TransformerDecoderLayer instance.

        :param embed_dim: The size of the embedding dimension of input tokens.
        :param num_heads: The number of attention heads to use when computing attention outputs.
        :param ffn_dim: The size of the hidden layer in the feed-forward neural network. Generally set to be
            4x the embed_dim.
        :param dropout: The probability of dropout used during training.
        """
        super().__init__()
        self.self_attn = MultiHeadedAttention(embed_dim, num_heads, dropout)
        self.cross_attn = MultiHeadedAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim, dropout)

        self.norm_self = nn.LayerNorm(embed_dim)
        self.norm_cross = nn.LayerNorm(embed_dim)
        self.norm_ffn = nn.LayerNorm(embed_dim)

        self.dropout_self = nn.Dropout(dropout)
        self.dropout_cross = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Computes a forward pass through the decoder layer of the tgt input and tgt_mask.

        :param tgt: The sequence input to the decoder layer of shape (N, T, E).
        :param memory: The sequence input from the last layer of the encoder of shape (N, S, E).
        :param tgt_mask: Masks out parts of the target sequence which should not be attended to by earlier
            token embeddings within the target sequence.
        :returns: A sequence of transformed features of shape (N, T, E).
        """
        # Self-attention sub-block
        resid_conn = tgt  # Record for the residual connection below
        tgt = self.self_attn(query=tgt, key=tgt, value=tgt, attn_mask=tgt_mask)
        tgt = self.dropout_self(tgt)
        tgt = tgt + resid_conn  # Residual connection
        tgt = self.norm_self(tgt)

        # Add a cross-attention sub-block using the encoder output (memory) as the key and value vectors,
        # there is no attention mask here since all input encoder values are visible to all decoder timesteps
        resid_conn = tgt  # Record for the residual connection below
        tgt = self.cross_attn(query=tgt, key=memory, value=memory, attn_mask=None)
        tgt = self.dropout_cross(tgt)
        tgt = tgt + resid_conn  # Residual connection
        tgt = self.norm_cross(tgt)

        # Add a feed-forward block sub-block
        resid_conn = tgt  # Record for the residual connection below
        tgt = self.ffn(tgt)
        tgt = self.dropout_ffn(tgt)
        tgt = tgt + resid_conn  # Residual connection
        tgt = self.norm_ffn(tgt)

        return tgt  # (N, T, E)


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

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Computes a forward pass through all layers of the decoder transformer, which attends the input tgt
        tokens to themselves (subject to tgt_mask) and also the memory tokens from the source sequence.

        :param tgt: The sequence input to the decoder layer of shape (N, T, E).
        :param memory: The sequence input from the last layer of the encoder of shape (N, S, E).
        :param tgt_mask: Masks out parts of the target sequence which should not be attended to by earlier
            token embeddings within the target sequence.
        :returns: A sequence of transformed features of shape (N, T, E).
        """
        x = tgt
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask)
        return x


###############################################
### Vision-Language Transformer (ViT) Model ###
###############################################
# TODO: Section marker

class VisionLanguageTransformer(nn.Module):
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

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 512,
                 num_layers: int = 3, num_heads: int = 6, ffn_dim: int = 256, dropout: float = 0.1,
                 sp_model=None, max_length: int = 150):
        """
        Initializes a Vision-Language Transformer model for image captioning.

        Input images are first processed by the vision transformer encoder, and then passed to the language
        model transformer decoder for work token index predictions.

        Images are assumed to be square. The same number of attention heads and transformer blocks are used
        for both the encoder and decoder transformers. Also the embedding dimension of the image patches is
        set equal to that of the word token embeddings.

        :param img_size: An integer representing the height & width of input image (assumes square image).
        :param patch_size: An integer representing height & width of each patch (square patch).
        :param in_channels: The number of input image channels (e.g. 3 for RGB).
        :param embed_dim: The dimension of the linear embedding space i.e. the size of the embedding vectors
            each image patch is turned into.
        :param num_layers: The number of encoder layers to sequentially stack.
        :param num_heads: The number of attention heads to use when computing attention outputs.
        :param ffn_dim: The size of the hidden layer in the feed-forward neural network. Generally set to be
            2x or 4x the embed_dim.
        :param dropout: The probability of dropout used during training.
        :param sp_model: A sentencepiece.SentencePieceProcessor model used to convert between tokens and
            strings.
        :param max_length: An integer denoting the max length of a caption decoding.
        """
        super().__init__()
        # 0). Record input parameters
        self.img_size = img_size  # The size of input images, H == W, images are assumed to be square
        self.patch_size = patch_size  # The size of each patch, H == W, patches are square
        self.in_channels = in_channels  # The number of input image color channels
        self.embed_dim = embed_dim  # The size of the image patch and word embeddings
        self.num_layers = num_layers  # The number of transformer blocks in the encoder and decoder
        self.num_heads = num_heads  # The number of attention heads
        self.ffn_dim = ffn_dim  # Size of the hidden layer of the feed-forward neural nets
        self.dropout = dropout  # Dropout probability used during training
        self.max_length = max_length  # Max caption decode output length

        # 1). Set up the vision-transformer for the encoder half
        # A patch embedding layer to convert input images into patch embedding vectors
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        # A positional embedding layer for the image patches and their locations relative to one another
        self.img_pos_embed = PositionalEmbedding(embed_dim, dropout, self.num_patches)
        # Construct the vision transformer encoder with self-attention to generate rich image patch embeddings
        encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, ffn_dim, dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)

        # 2). Set up the language-transformer for the decoder half
        self.vocab_size = len(sp_model)  # This in part determines the number of parameters in the model
        self.sp_model = sp_model

        # Record the word indices of special word tokens
        self._pad, self._start, self._end = sp_model.pad_id(), sp_model.bos_id(), sp_model.eos_id()

        # Projects the output of the encoder into features to be input into the decoder
        self.visual_projection = nn.Linear(embed_dim, embed_dim)
        # An embedding layer to convert word indices to word vectors
        self.word_idx_embed = nn.Embedding(self.vocab_size, embed_dim, padding_idx=self._pad)
        # A positional embedding layer for the positional indices of the work tokens
        self.word_pos_embed = PositionalEmbedding(embed_dim, dropout, max_length)
        # Construct the language decoder with cross-attention to the image patches embeddings from the encoder
        decoder_layer = TransformerDecoderLayer(embed_dim, num_heads, ffn_dim, dropout)
        self.decoder = TransformerDecoder(decoder_layer, num_layers)
        # A final projection layer from the outputs of the decoder to the vocab space
        self.vocab_proj = nn.Linear(embed_dim, self.vocab_size)

        self.register_buffer('device_param', torch.empty(0)) # A dummy param to tracking the device of this
        # model during later calls

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

    def encode(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Passes a batch of input images (N, C, H, W) into the vision transformer which is an encoder that
        transforms them into a set of image patch vector embeddings for each observation in the batch
        i.e. (N, C, H, W) -> (N, num_patches, E).

        :param imgs: A batch of input images of size (N, C, H, W) to be encoded into deep latent
            representations by the vision transformer.
        :returns: A batch of deep latent representations of the image patches of size (N, num_patches, E).
        """
        # 1). Convert the input image into a sequence of patch vectors
        patch_embeddings = self.patch_embed(imgs)  # (N, num_patches, embed_dim)

        # 2). Add positional encodings to retain spatial information
        patch_embeddings = self.img_pos_embed(patch_embeddings)  # (N, num_patches, embed_dim)

        # 3). Pass the sequence through the vision-transformer encoder
        x = self.encoder(patch_embeddings, None)  # (N, num_patches, embed_dim)

        return x  # (N, num_patches, embed_dim)

    def decode(self, img_features: torch.Tensor, word_idx_seq: torch.Tensor) -> torch.Tensor:
        """
        Given a batch of deep latent image patch representations from the encoder (N, num_patches, E) and a
        batch of word token index integers (N, T), this method computes a forward decode operation at each
        time step of the word token sequences to output a distribution of logit scores over the vocabulary of
        size (N, T, V).

        The word indices of the word_idx_seq input are converted to vector embeddings, then positional
        embeddings are added, and then the embeddings are passed through the decoder transformer with causal
        masked self-attention and cross-attention with respect to the img_features to produce a distribution
        over the vocabulary at each timestep corresponding to the model's forward-looking, next-word
        prediction probabilities.

        :param img_features: Deep latent representation of image patches of size (N, num_patches, E).
        :param word_idx_seq: A tensor containing a batch of word index sequences i.e. integers of size
            (N, T) where T is the max length among the sequences.
        :returns: A tensor of logits over the vocabulary denoting the model's next-word predictions at each
            input timestep of word_idx_seq.
        """
        N, T = word_idx_seq.shape  # N examples (batch size), each of at most length T

        # 1). Convert the captions (encoded as integers) into word embeddings using the embedding layer
        captions_emb = self.word_idx_embed(word_idx_seq)  # (batch_size, max_len, embed_dim) = (N, T, E)

        # 2). Add positional encodings to the captions embeddings
        captions_emb = self.word_pos_embed(captions_emb)  # (batch_size, T, wordvec_dim) = (N, T, E)

        # 3). Apply a projection to the img_features before feeding them into the cross-attention mechanism
        # nn.Linear(embed_dim, embed_dim): (batch_size, img_feat_dim) -> (batch_size, wordvec_dim)
        # where img_feat_dim == wordvec_dim == embed_dim
        img_features = self.visual_projection(img_features)  # (batch_size, num_patches, embed_dim)

        # 4). Create a mask (tgt_mask) for masking out attention scores from early words to latter words in
        # the captions sequence i.e. prevent lookahead, i.e. required for causal self-attention
        # Lower right triangular matrix of 1s to prevent lookahead
        tgt_mask = torch.tril(torch.ones(T, T), device=self.device_param.device)

        # 5). Pass the captions embeddings through a transformer block so that each word can attend to the
        # prior caption words (causal self-attention) and also to all the image features (cross-attention)
        x = self.decoder(captions_emb, img_features, tgt_mask)  # (batch_size, T, wordvec_dim)

        # 6). Map the transformer outputs to the vocab dim for a final word prediction at each time step
        # (batch_size, T, embed_dim) @ (embed_dim, vocab_size) = (batch_size, T, vocab_size)
        logit_scores = self.vocab_proj(x)
        return logit_scores  # (N, T, V) decoder_outputs

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
        return self.decode(self.encode(imgs), word_idx_seq)

    def compute_loss(self, decoder_outputs: torch.Tensor, target_word_idx: torch.Tensor, eps: float = 0.0
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
        :returns: The sum of negative log-likelihood across all timesteps in the decoding. Log-likelihood
            values are not computed for the padding tokens.
        """
        # Compute to a prob distribution over the vocabulary for each prediction timestep from the decoder,
        # decoder_outputs is what would be used at each timestep, we can process them all at once since we
        # have them all here at once as one big tensor of size (N, T, V)
        log_prob = F.log_softmax(decoder_outputs, dim=-1) # Softmax along the last dim, then take the log

        # Zero out, probabilities for which we have nothing in the target text i.e. the padding, create a bool
        # mask of 0s and 1s by checking that each entry is not equal to the <pad> token, 0s == padding token
        target_masks = (target_word_idx != self._pad).float()  # (B, T)

        # Compute the log probability of generating the true target words provided in this obs i.e. compute
        # the cross-entropy loss by pulling out the model's y-hat values for the true target words. For each
        # word in each sentence, pull out the y_hat prob associated with the true target word at time t.
        # log_prob is (N, T, V) and describes the probability distribution over the next word after the
        # current time step t. I.e. the first Y_t token is <s> and the first y_hat is the distribution of
        # what the model thinks should come afterwards. Hence log_prob[:, :-1, :] aligns wtih the true Y_t
        # words i.e. target_word_idx[:, 1:]. T = tgt_len includes <s> at the start and </s> at the end. We
        # don't want to include the prob of <s> but we do want to include the prob of predicting </s> to end
        # the sentence.
        target_words_log_prob = torch.gather(log_prob[:, :-1, :], index=target_word_idx[:, 1:].unsqueeze(-1),
                                             dim=-1).squeeze(-1)  # (B, T - 1) result
        if eps > 0:  # Apply label smoothing, put (1-eps) weight on the true class and eps / (|V|-1) on all
            # others when computing the cross-entropy loss values. From the above, we already have the values
            # for the true class label, so we can down-weight that by (1-eps) and then add to reach the goal
            sum_all_others = log_prob[:, :-1, :].sum(-1) - target_words_log_prob  # Sum log prob of all others
            mean_all_others = sum_all_others / (log_prob.shape[-1] - 1)  # Divide by (|V| - 1) to normalize
            # Take the weighted sum, down-weight the log-prob of the true class to (1-eps) and add all the
            # others at a weight of eps each i.e. the sum of all others gets a collective weight of eps
            target_words_log_prob = target_words_log_prob * (1 - eps) + mean_all_others * (eps)

        # Zero out the y_hat values for the padding tokens so that they don't contribute to the sum
        target_words_log_prob = target_words_log_prob * target_masks[:, 1:]  # (b, tgt_len - 1)

        # Return the avg negative log-likelihoods across all target tokens, across all captins
        loss = -target_words_log_prob.sum() / target_masks[:, 1:].sum()  # Compute 1 torch.float loss value
        return loss

    def sample(self, imgs: torch.Tensor, max_length: int = None, return_strings: bool = True
               ) -> Union[np.ndarray, List[str]]:
        """
        Given an input batch of images, this method uses a greedy decoding approach to predict the image
        caption for each and outputs a torch.Tensor of vocab word indices for each image in the image batch
        detailing what image caption would be predicted by the model for each.

        :param imgs: A batch of input images of size (N, C, H, W).
        :returns: A np.ndarray of size (N, T) containing the integer word indices of the predicted captions.
        """
        max_length = self.max_length if max_length is None else max_length
        self.eval()  # Switch to eval mode to turn off dropout

        with torch.no_grad():  # Used for inference, no gradient tracking required
            N = imgs.shape[0]  # The number of images in the batch
            img_features = self.encode(imgs)  # Process the images and generate image features (N, S, E)

            # Create an empty captions tensor to record the outputs, where all tokens are NULL initially
            captions = self._pad * np.ones((N, max_length), dtype=np.int32)  # Record ints (N, max_len)

            # Create a starting partial caption to begin to decoding sequence for each using the start token
            partial_captions = self._start * np.ones(N, dtype=np.int32)  # (N, )
            partial_captions = torch.LongTensor(partial_captions)
            partial_captions = partial_captions.unsqueeze(1)  # (N, 1) = (batch_size, decode seq len)

            # Record True for each sentence in the batch if it has reached the </s> token
            eos_mask = np.zeros(N, dtype=bool)
            for t in range(max_length):  # Run the decoding time steps to fill in the caption word idx
                # Predict the next token index for all images in the input batch
                # TODO: could be made more efficient by using a key-value cache during decoding
                output_logits = self.decode(img_features, partial_captions)
                output_logits = output_logits[:, -1, :]  # Use only last timestep's embedding values to make
                # the next token prediction

                # Choose the most likely word index from the vocabulary for each image, (N, V) -> (N, )
                word_indices = torch.argmax(output_logits, axis=1)  # (N, )

                # Update the captions output and the current partial captions tensor
                captions[:, t] = word_indices.numpy()
                captions[eos_mask, t] = self._pad # Replace with the padding token beyond </s>
                eos_mask = eos_mask & (captions[:, t] == self._end) # Update the end of sentence bool flags
                if eos_mask.sum() == len(eos_mask):  # Stop early if all outputs have reached their </s> token
                    break

                word_indices = word_indices.unsqueeze(1)  # (N, 1)
                partial_captions = torch.cat([partial_captions, word_indices], dim=1)  # (N, t) -> (N, t+1)

        if return_strings is False:  # Return as a np.ndarray
            return captions  # (N, T) = (batch_size, max_size)
        else:  # Return as a list of strings, process each sentence into plaintext
            return [decode_caption(x, self.sp_model) for x in captions]
