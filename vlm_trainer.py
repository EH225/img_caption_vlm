"""
This module contains a Trainer class that is used to perform model training for a VLM.
"""
import os
import numpy as np
import torch
from tqdm.auto import tqdm
from torch.optim import AdamW
import torch.nn as nn
import sentencepiece as spm
from typing import Tuple, List, Dict
from utils import get_device, get_amp_dtype, decode_caption, normalize_patches, denormalize_patches
from utils import plot_and_save_loss, denormalize_imagenet, update_cache_and_plot
import logging
import psutil
import pandas as pd
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch_models import ImageEncoder, MAEdecoder, CLIPtextEncoder, VisionLanguageModel, get_param_groups
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from pycocoevalcap.cider.cider import Cider


def infinite_loader(dataloader: DataLoader):
    """
    Infinitely yields batches of data from the input dataloader (dl) without caching batches.
    """
    while True:
        for batch in dataloader:
            yield batch


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)


def clip_loss(image_emb: torch.Tensor, text_emb: torch.Tensor, temperature: float = 0.07):
    """
    Computes the contrastive CLIP loss for an input image_emb and text_emb tensor.

    :param image_emb: An input tensor of size (N, E) of latent image embeddings, one per batch obs. This is
        usually the CLS token from the set of patch embedding vectors.
    :param text_emb: An input tensor of size (N, E) of latent text embeddings, one per batch obs. This is
        usually the latent embedding vector of the last input token that's not a padding token.
    :returns: The contrastive loss value.
    """
    # Apply L2 normalization to each embedding vector
    image_emb = F.normalize(image_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)
    logits = image_emb @ text_emb.T  # Compute cosine similarity (N, N)
    logits = logits / temperature
    targets = torch.arange(len(logits), device=logits.device)
    loss_i2t = F.cross_entropy(logits, targets)
    loss_t2i = F.cross_entropy(logits.T, targets)
    return (loss_i2t + loss_t2i) / 2


################################
### MAE Pre-Training Trainer ###
################################
# TODO: Section marker

class TrainerMAE:
    def __init__(self, encoder: ImageEncoder, decoder: MAEdecoder, dataloader_train: DataLoader,
                 dataloader_val: DataLoader, lr_start: float = 1e-4, lr_end: float = 1e-6,
                 weight_decay: float = 0.01, train_num_steps: int = 100000, warm_up_pct: float = 0.1,
                 adam_betas: Tuple[float] = (0.9, 0.98), grad_clip: float = 1.0,
                 sample_every: int = 1000, save_every: int = 5000, results_folder: str = None,
                 use_amp: bool = False, use_latest_checkpoint: bool = True, *args, **kwargs):
        """
        A framework for pre-training a vision-transformer (ViT) encoder model on a MAE objective. This class
        wrapper has methods for loading a model from a recent checkpoint, saving a model periodically during
        training, and running a training loop to train from scratch or to continue from the last checkpoint.

        :param encoder: A vision-transformer encoder model to create deep latent image patch representations.
        :param decoder: A MAE decoder transformer model to create patch re-constructions.
        :param dataloader_train: A data loader object that will yield the required training batches.
        :param dataloader_val: A data loader object that will yield the required validation batches.
        :param lr_start: The initial learning rate.
        :param lr_end: The terminal training learning rate.
        :param weight_decay: The weight_decay to provide to the Adam optimizer for L2 regularization.
        :param train_num_steps: The number of training steps to run in total.
        :param warm_up_pct: The percentage of train_num_steps over which the learning rate warm up period
            will be run i.e. ramps from very low to a peak of lr_start.
        :param adam_betas: Beta parameters for the adam optimizer.
        :param grad_clip: The amount of gradient clipping to use during training.
        :param sample_every: An int denoting how often to sample and save outputs from the model.
        :param save_every: An int denoting how often to save the model weights and losses.
        :param results_folder: A location to save the results of training.
        :param use_amp: Whether to use automatic mixed-precision type casting during training.
        :param use_latest_checkpoint: If set to True, then the latest checkpoint detected in the results
            directory will be loaded in before training begins to pick up from where it was last left off.
        """
        super().__init__()

        # 1). Create directories to save results
        assert results_folder is not None, "You must specify results folder to save the outputs"
        self.results_folder = results_folder  # A directory where the checkpoints will be saved
        self.checkpoints_folder = os.path.join(self.results_folder, "checkpoints/")
        self.losses_folder = os.path.join(self.results_folder, "losses/")
        self.samples_folder = os.path.join(self.results_folder, "samples/")
        for directory in [self.results_folder, self.checkpoints_folder,
                          self.losses_folder, self.samples_folder]:
            os.makedirs(directory, exist_ok=True)  # Create the directory if not already there

        # 2). Set up logging during training
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:  # Prevent duplicate handlers
            file_handler = logging.FileHandler(os.path.join(self.results_folder, "pretrain_mae.log"),
                                               encoding="utf-8")
            file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            # file_handler.stream = sys.stdout  # Ensure UTF-8 capable stream
            self.logger.addHandler(file_handler)

            tqdm_handler = TqdmLoggingHandler()
            tqdm_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            # tqdm_handler.stream = sys.stdout  # Ensure UTF-8 capable stream
            self.logger.addHandler(tqdm_handler)
        self.logger.propagate = False

        # 3). Record input parameters
        self.encoder = encoder  # The vision-transformer model
        self.decoder = decoder  # The small transformer used for decoding and re-constructing masked patches

        self.device = get_device()  # Auto-detect what device to use for training
        self.grad_clip = grad_clip  # The amount of gradient clipping to use during training
        self.amp_dtype = get_amp_dtype(self.device.type) if use_amp else None
        self.save_every = save_every  # The frequency of saving model weights
        self.sample_every = sample_every  # How often to generate samples
        self.num_sample = 5  # Sets how many obs to randomly sample from the validation set for sampling
        self.train_num_steps = train_num_steps  # The total number of training steps
        self.warm_up_pct = warm_up_pct  # The percentage of training steps to run as LR warm-up

        # Save a pointer to the train and validation dataloaders
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val

        # 4). Configure the optimizer for training - segment the encoder training parameters from the decoder
        self.logger.info(f"Encoder model parameters: {sum(p.numel() for p in encoder.parameters())}")
        self.logger.info(f"Decoder model parameters: {sum(p.numel() for p in decoder.parameters())}")

        self.all_params = list(encoder.parameters()) + list(decoder.parameters())

        enc_param_groups = get_param_groups(self.encoder, weight_decay=weight_decay, lr=lr_start)
        dec_param_groups = get_param_groups(self.decoder, weight_decay=weight_decay, lr=lr_start)

        # Configure the optimizer for training both models
        self.opt = AdamW(enc_param_groups + dec_param_groups, betas=adam_betas)

        # 5). Configure a learning rate scheduler for training with warm-up and cosine annealing
        warmup_steps = int(train_num_steps * warm_up_pct)  # Slowly ramp up the LR from very low to peak
        warmup = LinearLR(self.opt, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        # Cosine annealing of the learning rate during the rest of training
        decay = CosineAnnealingLR(self.opt, T_max=train_num_steps - warmup_steps, eta_min=lr_end)
        # Stack both the learning rate warm up and the gradual linear decay into 1 scheduler
        self.scheduler = SequentialLR(self.opt, schedulers=[warmup, decay], milestones=[warmup_steps])

        # 6). Keep track of the training step and losses along the way
        self.step = 0  # Training step counter
        self.all_losses = []  # Aggregate loss values during training

        # 7). Load in the latest checkpoint weights to continue from where the models were last saved
        if use_latest_checkpoint:
            checkpoints = os.listdir(self.checkpoints_folder)
            if len(checkpoints) > 0:
                last_checkpoint = max([int(x.replace("model-", "").replace(".pt", "")) for x in checkpoints])
                self.load(last_checkpoint)  # Load in the most recent milestone to continue training

    def save(self, milestone: int) -> None:
        """
        Saves the weights of the model for the current milestone.

        :param milestone: An integer denoting the training timestep at which the model weights were saved.
        :returns: None. Writes the weights and losses to disk.
        """
        checkpoint_path = os.path.join(self.checkpoints_folder, f"model-{milestone}.pt")
        self.logger.info(f"Saving model to {checkpoint_path}.")
        data = {"step": self.step,
                "model_encoder": self.encoder.state_dict(),
                "model_decoder": self.decoder.state_dict(),
                "opt": self.opt.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                }
        torch.save(data, checkpoint_path)
        # Save down all the loss values produced by model training since the last caching
        pd.Series(self.all_losses).to_csv(os.path.join(self.losses_folder, f"losses-{milestone}.csv"))

    def load(self, milestone: int) -> None:
        """
        Loads in the cached weights from disk for a particular milestone.

        :param milestone: An integer denoting the training timestep at which the model weights were saved.
        :returns: None. Weights and other trainer state parameter values are loaded into memory.
        """
        checkpoint_path = os.path.join(self.checkpoints_folder, f"model-{milestone}.pt")
        self.logger.info(f"Loading model from {checkpoint_path}.")
        checkpoint_data = torch.load(checkpoint_path, map_location=self.device)

        # Re-instate the training step counter, model weights, and optimizer state from the checkpoint data
        # read in from disk
        self.step = checkpoint_data["step"]
        self.encoder.load_state_dict(checkpoint_data["model_encoder"])
        self.decoder.load_state_dict(checkpoint_data["model_decoder"])
        self.opt.load_state_dict(checkpoint_data["opt"])
        self.scheduler.load_state_dict(checkpoint_data["scheduler"])
        # Losses are not loaded in, they are saved to disk periodically with the model weights and are not
        # needed to continue training. The losses obtained by training will be cached again at the next save

        # Move the model and the optimizer to the same device to continue training or for inference
        for state in self.opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

    def report_lr_wd(self):
        """
        Reports the learning rates and weight decay parameter values of the vlm model.
        """
        self.logger.info(f"Reporting learning rates and weight decay at step={self.step}")
        labels = ["Encoder Decay", "Encoder No Decay", "Decoder Decay", "Decoder No Decay"]
        for i, group in enumerate(self.opt.param_groups):  # Report all learning rates
            self.logger.info((f"{(labels[i] + ':').ljust(17)} lr = {group['lr']:.2e}, wd = "
                              f"{group['weight_decay']:.2e}, count = {len(group['params'])}"))

    def train(self, mask_ratio: float = 0.75) -> None:
        """
        Runs the MAE pre-training of the VLM model until completion for self.train_num_steps total training
        iterations.

        :param mask_ratio: The ratio of the image patches to randomly mask out during training.
        :returns: None. Caches the results to disk.
        """
        self.logger.info(f"Starting MAE Training, device={self.device}, amp_dtype={self.amp_dtype}")
        self.report_lr_wd()

        self.encoder.to(self.device)  # Move the encoder model to the correct device
        self.encoder.train()  # Make sure to set the encoder model to train mode for training

        self.decoder.to(self.device)  # Move the decoder model to the correct device
        self.decoder.train()  # Make sure to set the decoder model to train mode for training

        # These data-loaders do not cache batches which makes them more memory efficient
        inf_dataloader_train = infinite_loader(self.dataloader_train)
        inf_dataloader_val = infinite_loader(self.dataloader_val)

        if self.amp_dtype is not None:
            if self.device.type != 'cuda':
                self.logger.info("AMP with FP16 requires CUDA")
                self.amp_dtype = None
            else:
                scaler = torch.amp.GradScaler('cuda')

        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:

            while self.step < self.train_num_steps:  # Run until all training iterations are complete
                # Get the next training batch and move it to the same device as the models
                batch = next(inf_dataloader_train)
                imgs = batch["images"].to(self.device, non_blocking=True)  # (N, C, H, W)

                # Zero the grads of the opt before computing the loss
                self.opt.zero_grad(set_to_none=True)

                # Compute the forward pass through the vlm encoder and the MAE decoder models
                if self.amp_dtype is not None:
                    with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                        x_vis_latent, mask, ids_restore = self.encoder.encode_mae(imgs, mask_ratio)
                        pred_img, mask = self.decoder(x_vis_latent, mask, ids_restore)
                        # Convert to image patches and normalize for the MSE obj (N, num_patches, patch_dim)
                        norm_patches = normalize_patches(self.encoder.patch_embed.patchify(imgs))
                        loss = ((pred_img - norm_patches) ** 2).mean(dim=-1)  # Avg per pixel squared loss
                        loss = (loss * mask).sum() / max(mask.sum(), 1)  # Compute the MSE on the masked
                else:
                    x_vis_latent, mask, ids_restore = self.encoder.encode_mae(imgs, mask_ratio)
                    pred_img, mask = self.decoder(x_vis_latent, mask, ids_restore)
                    # Convert to image patches and normalize for the MSE obj (N, num_patches, patch_dim)
                    norm_patches = normalize_patches(self.encoder.patch_embed.patchify(imgs))
                    loss = ((pred_img - norm_patches) ** 2).mean(dim=-1)  # Avg per pixel squared loss
                    loss = (loss * mask).sum() / max(mask.sum(), 1)  # Compute the MSE on the masked

                if self.amp_dtype == torch.float16:
                    scaler.scale(loss).backward()
                    if self.grad_clip is not None:
                        scaler.unscale_(self.opt)  # Unscale before clipping
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.all_params, self.grad_clip)
                    scaler.step(self.opt)  # Update the model parameters by taking a gradient step
                    scaler.update()
                else:
                    loss.backward()
                    if self.grad_clip is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.all_params, self.grad_clip)
                    # Update the model parameters by taking a gradient step
                    self.opt.step()

                pbar.set_postfix(loss=f"{loss.item():.4f}", grad_norm=f"{grad_norm:.3f}")

                self.scheduler.step()  # Update the learning rate scheduler

                self.all_losses.append(loss.item())  # Aggregate all the loss values for each timestep
                self.step += 1

                # Periodically save the model weights to disk
                if self.step % self.save_every == 0 or self.step == self.train_num_steps:
                    self.save(self.step)
                    plot_and_save_loss(self.losses_folder)  # Generate a new plot of the training losses
                    self.all_losses = []  # Clear the list of losses after each save, store only the ones
                    # from the last save to the next save
                    torch.cuda.empty_cache()

                # Periodically generate samples from the model and save them to disk
                if self.step % self.sample_every == 0 or self.step == self.train_num_steps:
                    # Periodically log the loss and other training metrics
                    self.logger.info((f"loss={loss.item():.4f}, grad_norm={grad_norm:.3f}"))
                    self.report_lr_wd()
                    gpu_mem_used = torch.cuda.memory_allocated() / 1e9
                    cpu_mem_used = psutil.virtual_memory().used / 1e9
                    msg = f"[GPU, CPU] Memory Allocated: {gpu_mem_used:.2f}GB {cpu_mem_used:.2f}GB"
                    self.logger.info(msg)

                    self.logger.info("\n")
                    self.logger.info(f"Generating samples at step={self.step}")

                    batch = next(inf_dataloader_val)  # Get the next batch of data from the validation set
                    indices = torch.randperm(batch["images"].size(0))[:self.num_sample]
                    imgs = batch["images"][indices].to(self.device, non_blocking=True)
                    # Switch the models over to eval mode for decoding a few validation set samples
                    self.encoder.eval()
                    self.decoder.eval()

                    # Pass the val set images through the MAE encoder (N, num_patches_vis, patch_dim)
                    x_vis_latent, mask, ids_restore = self.encoder.encode_mae(imgs, mask_ratio)
                    # Generate decoded image reconstructions of size (N, num_patches, patch_dim)
                    pred_imgs, mask = self.decoder(x_vis_latent, mask, ids_restore)
                    # Patchify the original images so that we can use them for filling
                    img_patches = self.encoder.patch_embed.patchify(imgs)  # (N, num_patches, patch_dim)
                    # Create a masked version of the original images, fill zeros for the masked patches
                    img_patches_masked = torch.where(mask.unsqueeze(-1) == 0, img_patches, 0)
                    # The decoder is trained to predict the whitened pixel values, reverse the normalization
                    pred_imgs = denormalize_patches(img_patches, pred_imgs.detach())
                    # Fill the unmasked image patches with the actual image patch data to blend with the true
                    pred_imgs_filled = torch.where(mask.unsqueeze(-1) == 0, img_patches, pred_imgs)
                    # Combine all the image tensors into 1 big tensor so that they can be saved to disk
                    # Original + Original Masked + Pred + Pred Blended with Original
                    N, num_patches, patch_dim = img_patches.shape
                    tensors = [img_patches, img_patches_masked, pred_imgs, pred_imgs_filled]
                    x = torch.stack(tensors, dim=1).reshape(N * len(tensors), num_patches, patch_dim)
                    # Convert from (N*4, num_patches, patch_dim) -> (N, C, H, W)
                    img_grid = denormalize_imagenet(self.encoder.patch_embed.unpatchify(x))

                    # Save the values to an image grid in the samples folder
                    output_filepath = f"{os.path.join(self.samples_folder, str(self.step))}.png"
                    save_image(img_grid, output_filepath, nrow=4)
                    self.logger.info(f"Saved sampled image grid to {output_filepath}")

                    # Switch the models back over to continue training
                    self.encoder.train()
                    self.decoder.train()

                del imgs, x_vis_latent, mask, ids_restore, pred_img, loss

                pbar.update(1)


#######################################
### CLIP-Style Pre-Training Trainer ###
#######################################
# TODO: Section marker

class TrainerCLIP:
    def __init__(self, img_encoder: ImageEncoder, text_encoder: CLIPtextEncoder,
                 sp_model: spm.SentencePieceProcessor, dataloader_train: DataLoader,
                 dataloader_val: DataLoader, lr_start: float = 1e-4, lr_end: float = 1e-6,
                 weight_decay: float = 0.01, train_num_steps: int = 100000, warm_up_pct: float = 0.1,
                 adam_betas: Tuple[float] = (0.9, 0.98), grad_clip: float = 1.0,
                 sample_every: int = 1000, save_every: int = 5000, results_folder: str = None,
                 use_amp: bool = False, use_latest_checkpoint: bool = True, *args, **kwargs):
        """
        A framework for pre-training a vision-transformer (ViT) encoder model on a CLIP-style vision-language
        objective. This step of pre-training generally follows MAE pre-training and trains a
        vision-transformer (ViT) (ImageEncoder) to associate images with textual descriptions, which teaches
        the image encoder to learn semantic image understanding by maximizing the cosine similarity of images
        and their captions in a shared latent space, while minimizing the cosine similarity of off-diagonal
        (image, caption) pairs.

        This class wrapper has methods for loading a model from a recent checkpoint, saving a model
        periodically during training, and running a training loop to train from scratch or to continue from
        the last checkpoint.

        :param img_encoder: A vision-transformer encoder model to create deep latent image patch tensors.
        :param text_encoder: A frozen CLIP text encoder used to generate rich latent text representations.
        :param sp_model: A sentencepiece.SentencePieceProcessor model used to convert between tokens and
            strings.
        :param dataloader_train: A data loader object that will yield the required training batches.
        :param dataloader_val: A data loader object that will yield the required validation batches.
        :param lr_start: The initial learning rate.
        :param lr_end: The terminal training learning rate.
        :param weight_decay: The weight_decay to provide to the Adam optimizer for L2 regularization.
        :param train_num_steps: The number of training steps to run in total.
        :param warm_up_pct: The percentage of train_num_steps over which the learning rate warm up period
            will be run i.e. ramps from very low to a peak of lr_start.
        :param adam_betas: Beta parameters for the adam optimizer.
        :param grad_clip: The amount of gradient clipping to use during training.
        :param sample_every: An int denoting how often to sample and save outputs from the model.
        :param save_every: An int denoting how often to save the model weights and losses.
        :param results_folder: A location to save the results of training.
        :param use_amp: Whether to use automatic mixed-precision type casting during training.
        :param use_latest_checkpoint: If set to True, then the latest checkpoint detected in the results
            directory will be loaded in before training begins to pick up from where it was last left off.
        """
        super().__init__()

        # 1). Create directories to save results
        assert results_folder is not None, "You must specify results folder to save the outputs"
        self.results_folder = results_folder  # A directory where the checkpoints will be saved
        self.checkpoints_folder = os.path.join(self.results_folder, "checkpoints/")
        self.losses_folder = os.path.join(self.results_folder, "losses/")
        for directory in [self.results_folder, self.checkpoints_folder, self.losses_folder]:
            os.makedirs(directory, exist_ok=True)  # Create the directory if not already there

        # 2). Set up logging during training
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:  # Prevent duplicate handlers
            file_handler = logging.FileHandler(os.path.join(self.results_folder, "pretrain_clip.log"),
                                               encoding="utf-8")
            file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            # file_handler.stream = sys.stdout  # Ensure UTF-8 capable stream
            self.logger.addHandler(file_handler)

            tqdm_handler = TqdmLoggingHandler()
            tqdm_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            # tqdm_handler.stream = sys.stdout  # Ensure UTF-8 capable stream
            self.logger.addHandler(tqdm_handler)
        self.logger.propagate = False

        # 3). Record input parameters
        self.img_encoder = img_encoder  # The vision-transformer encoder model
        self.text_encoder = text_encoder  # Pre-trained, frozen text-encoder model from CLIP
        self.sp_model = sp_model

        self.device = get_device()  # Auto-detect what device to use for training
        self.grad_clip = grad_clip  # The amount of gradient clipping to use during training
        self.amp_dtype = get_amp_dtype(self.device.type) if use_amp else None
        self.save_every = save_every  # The frequency of saving model weights
        self.sample_every = sample_every  # How often to generate samples
        self.num_sample = 5  # Sets how many obs to randomly sample from the validation set for sampling
        self.train_num_steps = train_num_steps  # The total number of training steps
        self.warm_up_pct = warm_up_pct  # The percentage of training steps to run as LR warm-up

        # Save a pointer to the train and validation dataloaders
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val

        # 4). Configure the optimizer for training - segment the image and text encoders
        self.logger.info(
            f"Image Encoder model parameters: {sum(p.numel() for p in img_encoder.parameters())}")
        self.logger.info(
            f"Text Encoder model parameters: {sum(p.numel() for p in text_encoder.parameters())}")

        # Add a linear mapping between the image encoder embed_dim and the text encoder embed_dim
        self.vision_proj = nn.Linear(img_encoder.embed_dim, text_encoder.embed_dim)
        self.all_params = list(img_encoder.parameters()) + list(text_encoder.parameters())
        self.all_params += list(self.vision_proj.parameters())

        img_enc_param_groups = get_param_groups(self.img_encoder, weight_decay=weight_decay, lr=lr_start)
        vision_proj_param_groups = get_param_groups(self.vision_proj, weight_decay=weight_decay, lr=lr_start)

        # Configure the optimizer for training the image encoder and vision projection
        self.opt = AdamW(img_enc_param_groups + vision_proj_param_groups, betas=adam_betas)

        # 5). Configure a learning rate scheduler for training with warm-up and cosine annealing
        warmup_steps = int(train_num_steps * warm_up_pct)  # Slowly ramp up the LR from very low to peak
        warmup = LinearLR(self.opt, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        # Cosine annealing of the learning rate during the rest of training
        decay = CosineAnnealingLR(self.opt, T_max=train_num_steps - warmup_steps, eta_min=lr_end)
        # Stack both the learning rate warm up and the gradual linear decay into 1 scheduler
        self.scheduler = SequentialLR(self.opt, schedulers=[warmup, decay], milestones=[warmup_steps])

        # 6). Keep track of the training step and losses along the way
        self.step = 0  # Training step counter
        self.all_losses = []  # Aggregate loss values during training

        # 7). Load in the latest checkpoint weights to continue from where the models were last saved
        if use_latest_checkpoint:
            checkpoints = os.listdir(self.checkpoints_folder)
            if len(checkpoints) > 0:
                last_checkpoint = max([int(x.replace("model-", "").replace(".pt", "")) for x in checkpoints])
                self.load(last_checkpoint)  # Load in the most recent milestone to continue training

        # 7). Load in the latest checkpoint weights to continue from where the models were last saved
        if use_latest_checkpoint > 0:  # If set to True, then use the most recent checkpoint available
            checkpoints = os.listdir(self.checkpoints_folder)
            if len(checkpoints) > 0:  # If there is a milestone saved, load in the weights
                last_checkpoint = max([int(x.replace("model-", "").replace(".pt", "")) for x in checkpoints])
                # Load in the most recent milestone to continue training
                weights_only = (use_latest_checkpoint == 2)  # Load in only the weights if set to 2
                self.load(last_checkpoint, weights_only)
            else:  # Otherwise, check if there are any MAE pre-trained weights to use as a starting point
                max_milestone = None  # Look for checkpoints in the pre-trained weights folder instead
                pretrained_wts_dir = os.path.join(self.results_folder, "../pretrain_mae/checkpoints")
                if os.path.exists(pretrained_wts_dir):
                    milestones = [int(x.replace("model-", "").replace(".pt", ""))
                                  for x in os.listdir(pretrained_wts_dir)]
                    if len(milestones) > 0:
                        max_milestone = max(milestones)
                if max_milestone is not None:  # Load in the latest pre-trained weights
                    self.load_pretrained(max_milestone)

    def save(self, milestone: int) -> None:
        """
        Saves the weights of the model for the current milestone.

        :param milestone: An integer denoting the training timestep at which the model weights were saved.
        :returns: None. Writes the weights and losses to disk.
        """
        checkpoint_path = os.path.join(self.checkpoints_folder, f"model-{milestone}.pt")
        self.logger.info(f"Saving model to {checkpoint_path}.")
        data = {"step": self.step,
                "img_encoder": self.img_encoder.state_dict(),
                "vision_proj": self.vision_proj.state_dict(),
                "opt": self.opt.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                }
        torch.save(data, checkpoint_path)
        # Save down all the loss values produced by model training since the last caching
        pd.Series(self.all_losses).to_csv(os.path.join(self.losses_folder, f"losses-{milestone}.csv"))

    def load(self, milestone: int) -> None:
        """
        Loads in the cached weights from disk for a particular milestone.

        :param milestone: An integer denoting the training timestep at which the model weights were saved.
        :returns: None. Weights and other trainer state parameter values are loaded into memory.
        """
        checkpoint_path = os.path.join(self.checkpoints_folder, f"model-{milestone}.pt")
        self.logger.info(f"Loading model from {checkpoint_path}.")
        checkpoint_data = torch.load(checkpoint_path, map_location=self.device)

        # Re-instate the training step counter, model weights, and optimizer state from the checkpoint data
        # read in from disk
        self.step = checkpoint_data["step"]
        self.img_encoder.load_state_dict(checkpoint_data["img_encoder"])
        self.vision_proj.load_state_dict(checkpoint_data["vision_proj"])
        self.opt.load_state_dict(checkpoint_data["opt"])
        self.scheduler.load_state_dict(checkpoint_data["scheduler"])
        # Losses are not loaded in, they are saved to disk periodically with the model weights and are not
        # needed to continue training. The losses obtained by training will be cached again at the next save

        # Move the model and the optimizer to the same device to continue training or for inference
        for state in self.opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

    def load_pretrained(self, milestone: int) -> None:
        """
        Loads in MAE pre-trained model weights form disk for a particular milestone.

        :param milestone: An integer denoting the training timestep at which the model weights were saved.
        :returns: None. Weights are loaded into the model from disk.
        """
        file_path = f"../pretrain_mae/checkpoints/model-{milestone}.pt"
        checkpoint_path = os.path.join(self.results_folder, file_path)
        self.logger.info(f"Loading MAE pretrained encoder model weights from {checkpoint_path}.")
        checkpoint_data = torch.load(checkpoint_path, map_location=self.device)
        # Re-instate the model weights from the checkpoint data read in from disk
        self.img_encoder.load_state_dict(checkpoint_data["model_encoder"])

    def report_lr_wd(self):
        """
        Reports the learning rates and weight decay parameter values of the vlm model.
        """
        self.logger.info(f"Reporting learning rates and weight decay at step={self.step}")
        labels = ["Encoder Decay", "Encoder No Decay", "Vision Proj Decay", "Vision Proj No Decay"]
        for i, group in enumerate(self.opt.param_groups):  # Report all learning rates
            self.logger.info((f"{(labels[i] + ':').ljust(21)} lr = {group['lr']:.2e}, wd = "
                              f"{group['weight_decay']:.2e}, count = {len(group['params'])}"))

    def train(self) -> None:
        """
        Runs the CLIP-style pre-training of the VLM image encoder until  completion for self.train_num_steps
        total training iterations.

        :returns: None. Caches the results to disk.
        """
        self.logger.info(f"Starting CLIP Training, device={self.device}, amp_dtype={self.amp_dtype}")
        self.report_lr_wd()

        # Move the modesl to the right device and make sure they're set to training mode
        for m in [self.img_encoder, self.vision_proj]:
            m.to(self.device)
            m.train()

        inf_dataloader_train = infinite_loader(self.dataloader_train)

        if self.amp_dtype is not None:
            if self.device.type != 'cuda':
                self.logger.info("AMP with FP16 requires CUDA")
                self.amp_dtype = None
            else:
                scaler = torch.amp.GradScaler('cuda')

        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:

            while self.step < self.train_num_steps:  # Run until all training iterations are complete
                # Get the next training batch and move it to the same device as the models
                batch = next(inf_dataloader_train)
                imgs = batch["images"].to(self.device, non_blocking=True)  # (N, C, H, W)
                # Decode the captions to convert from int -> string captions
                captions_str = [self.sp_model.decode(c.tolist()) for c in batch["captions"]]
                # Apply the CLIP tokenizer to tokenize the captions in the way expected by the text_encoder
                tokens = self.text_encoder.tokenizer(captions_str).to(self.device, non_blocking=True)

                # Zero the grads of the opt before computing the loss
                self.opt.zero_grad(set_to_none=True)

                # Compute the forward pass through the image and text encoders
                if self.amp_dtype is not None:
                    with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                        img_emb = self.vision_proj(self.img_encoder(imgs)[:, -1, :])  # CLS token only
                        txt_emb = self.text_encoder(tokens)  # Normalization takes place within the loss func
                        loss = clip_loss(img_emb, txt_emb)  # Compute the CLIP cosine loss
                else:
                    img_emb = self.vision_proj(self.img_encoder(imgs)[:, -1, :])  # CLS token only
                    txt_emb = self.text_encoder(tokens)  # Normalization takes place within the loss func
                    loss = clip_loss(img_emb, txt_emb)  # Compute the CLIP cosine loss

                if self.amp_dtype == torch.float16:
                    scaler.scale(loss).backward()
                    if self.grad_clip is not None:
                        scaler.unscale_(self.opt)  # Unscale before clipping
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.all_params, self.grad_clip)
                    scaler.step(self.opt)  # Update the model parameters by taking a gradient step
                    scaler.update()
                else:
                    loss.backward()
                    if self.grad_clip is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.all_params, self.grad_clip)
                    # Update the model parameters by taking a gradient step
                    self.opt.step()

                pbar.set_postfix(loss=f"{loss.item():.4f}", grad_norm=f"{grad_norm:.3f}")

                self.scheduler.step()  # Update the learning rate scheduler

                self.all_losses.append(loss.item())  # Aggregate all the loss values for each timestep
                self.step += 1

                # Periodically save the model weights to disk
                if self.step % self.save_every == 0 or self.step == self.train_num_steps:
                    self.save(self.step)
                    plot_and_save_loss(self.losses_folder)  # Generate a new plot of the training losses
                    self.all_losses = []  # Clear the list of losses after each save, store only the ones
                    # from the last save to the next save
                    torch.cuda.empty_cache()

                # Periodically run some evaluations on the validation data set to check that the embeddings
                # of val set images are close to the embeddings of val set captions
                if self.step % self.sample_every == 0 or self.step == self.train_num_steps:
                    # Periodically log the loss and other training metrics
                    self.logger.info((f"loss={loss.item():.4f}, grad_norm={grad_norm:.3f}, step={self.step}"))
                    self.report_lr_wd()
                    gpu_mem_used = torch.cuda.memory_allocated() / 1e9
                    cpu_mem_used = psutil.virtual_memory().used / 1e9
                    msg = f"[GPU, CPU] Memory Allocated: {gpu_mem_used:.2f}GB {cpu_mem_used:.2f}GB"
                    self.logger.info(msg)

                    self.logger.info("\n")
                    self.logger.info(f"Evaluating CLIP loss on validation set at step={self.step}")

                    for m in [self.img_encoder, self.vision_proj]:
                        m.eval()

                    with torch.no_grad():  # No gradient tracking needed during evaluation
                        eval_losses, eval_n = 0.0, 0
                        for batch in self.dataloader_train:
                            imgs = batch["images"].to(self.device, non_blocking=True)  # (N, C, H, W)
                            # Decode the captions to convert from int -> string captions
                            captions_str = [self.sp_model.decode(c.tolist()) for c in batch["captions"]]
                            # Apply the CLIP tokenizer to tokenize the captions in the way expected
                            tokens = self.text_encoder.tokenizer(captions_str).to(self.device,
                                                                                  non_blocking=True)
                            # Compute the forward pass through the image and text encoders
                            if self.amp_dtype is not None:
                                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                                    img_emb = self.vision_proj(self.img_encoder(imgs)[:, -1, :])  # CLS token
                                    txt_emb = self.text_encoder(tokens)
                                    loss = clip_loss(img_emb, txt_emb)  # Compute the CLIP cosine loss
                            else:
                                img_emb = self.vision_proj(self.img_encoder(imgs)[:, -1, :])  # CLS token
                                txt_emb = self.text_encoder(tokens)
                                loss = clip_loss(img_emb, txt_emb)  # Compute the CLIP cosine loss

                            eval_losses += loss.item()  # Sum the losses over the validation set
                            eval_n += 1  # Count up how batches were used

                    eval_loss = eval_losses / eval_n
                    self.logger.info(f"Validation set loss={eval_loss:.4f}")
                    update_cache_and_plot(self.step, eval_loss, self.results_folder, "eval_loss")

                    # Switch the models back over to continue training
                    for m in [self.img_encoder, self.vision_proj]:
                        m.train()

                del batch, imgs, captions_str, tokens, img_emb, txt_emb, loss

                pbar.update(1)


################################
### Image Captioning Trainer ###
################################
# TODO: Section marker

class TrainerCaptioning:
    def __init__(self, vlm: VisionLanguageModel, dataloader_train: DataLoader,
                 dataloader_val: DataLoader, gts_train: Dict = None, gts_val: Dict = None,
                 lr_start: float = 1e-4, lr_end: float = 1e-6, wd_encoder: float = 0.05,
                 wd_decoder: float = 0.01, train_num_steps: int = 30000, warm_up_pct: float = 0.10,
                 frozen_enc_pct: float = 0.30, adam_betas: Tuple[float] = (0.9, 0.98), grad_clip: float = 1.0,
                 sample_every: int = 500, save_every: int = 5000, eval_every: int = 1000,
                 results_folder: str = None, use_amp: bool = False, use_latest_checkpoint: int = 1,
                 scst: bool = False, *args, **kwargs):
        """
        A framework for training a Vision-Language Model (VLT). This class wrapper has methods for loading
        a model from a recent checkpoint, saving a model periodically during training, and running a training
        loop to train from scratch or to continue from the last checkpoint.

        :param vlm: A vision-language model for captioning implemented in pytorch.
        :param dataloader_train: A data loader object that will yield the required training batches.
        :param dataloader_val: A data loader object that will yield the required validation batches.
        :param gts_train: A dictionary of train set ground-truth captions organized in a dictionary with
            image_id as the key (as an int) and lists of strings (i.e. GT captions) as the values.
        :param gts_val: A dictionary of validation set ground-truth captions organized in a dictionary with
            image_id as the key (as an int) and lists of strings (i.e. GT captions) as the values.
        :param lr_start: The initial learning rate.
        :param lr_end: The terminal training learning rate.
        :param wd_encoder: The weight_decay to provide to the Adam optimizer for L2 regularization of the
            encoder model parameters (certain params are excluded).
        :param wd_decoder: The weight_decay to provide to the Adam optimizer for L2 regularization of the
            decoder model parameters (certain params are excluded).
        :param train_num_steps: The number of training steps to run in total.
        :param warm_up_pct: The percentage of train_num_steps over which the learning rate warm up period
            will be run i.e. ramps from very low to a peak of lr_start.
        :param frozen_enc_pct: The percentage of the initial train_num_steps to freeze the encoder parameters
            during training so that the random init of the decoder does not undo the pre-training.
        :param adam_betas: Beta parameters for the adam optimizer.
        :param grad_clip: The amount of gradient clipping to use during training.
        :param sample_every: An int denoting how often to sample and save outputs from the model.
        :param save_every: An int denoting how often to save the model weights and losses.
        :param eval_every: An int denoting how often to run the evaluation scoring on the validation set.
        :param results_folder: A location to save the results of training.
        :param use_amp: Whether to use automatic mixed-precision type casting during training.
        :param use_latest_checkpoint: If set to 0, then no loading is done from disk automatically. If set to
            1 then the model weights, opt, and scheduler will be loaded from the checkpoint directory before
            training begins to pick up from where it was last left off. If set to 2, then only the weights
            are loaded, but not the optimizer or scheduler.
        :param scst: A bool indicating if this is SCST fine-tuning, otherwise if False, the trainer will be
            set up for supervised captioning training using teacher-forcing.
        """
        super().__init__()

        # 1). Create directories to save results
        assert results_folder is not None, "You must specify results folder to save the outputs"
        self.results_folder = results_folder  # A directory where the checkpoints will be saved
        self.checkpoints_folder = os.path.join(self.results_folder, "checkpoints/")
        self.losses_folder = os.path.join(self.results_folder, "losses/")
        for directory in [self.results_folder, self.checkpoints_folder, self.losses_folder]:
            os.makedirs(directory, exist_ok=True)  # Create the directory if not already there

        # 2). Set up logging during training
        self.logger = logging.getLogger(self.__class__.__name__ + "scst" if scst else "")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:  # Prevent duplicate handlers
            log_file = "train.log" if scst is False else "scst_train.log"
            file_handler = logging.FileHandler(os.path.join(self.results_folder, log_file),
                                               encoding="utf-8")
            file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            # file_handler.stream = sys.stdout  # Ensure UTF-8 capable stream
            self.logger.addHandler(file_handler)

            tqdm_handler = TqdmLoggingHandler()
            tqdm_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            # tqdm_handler.stream = sys.stdout  # Ensure UTF-8 capable stream
            self.logger.addHandler(tqdm_handler)
        self.logger.propagate = False
        self.logger.info("Logger successfully initialized")

        # 3). Record input parameters
        self.vlm = vlm  # The vision-language model to be trained
        self.device = get_device()  # Auto-detect what device to use for training
        self.grad_clip = grad_clip  # The amount of gradient clipping to use during training
        self.amp_dtype = get_amp_dtype(self.device.type) if use_amp else None
        self.num_sample = 5  # Sets how many obs to randomly sample from the validation set for sampling
        self.save_every = save_every  # The frequency of saving model weights
        self.sample_every = sample_every  # How often to generate samples
        self.eval_every = eval_every  # How often to run the evaluation metrics on the validation set
        self.train_num_steps = train_num_steps  # The total number of training steps to run
        self.warm_up_pct = warm_up_pct  # The percentage of training steps to run as LR warm-up
        self.frozen_enc_pct = frozen_enc_pct  # The number of steps for which the encoder will be frozen
        self.freeze_steps = int(train_num_steps * frozen_enc_pct)  # Freeze the encoder at first
        # since it will be pre-trained on the MAE objective and should generally be in a good spot
        self.scst = scst  # Record whether this is SCST fine-tuning or not

        # Save a pointer to the train and validation dataloaders
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        # A dictionary of image captions for periodic eval vs the validation set or training
        self.gts_train = {int(k): v for k, v in gts_train.items()} if gts_train else None
        self.gts_val = {int(k): v for k, v in gts_val.items()} if gts_val else None
        self.cider_scorer = Cider()  # Used for computing CIDEr scores

        # 4). Configure the optimizer for training - segment the encoder training parameters from the decoder
        self.encoder_params = list(self.vlm.encoder.parameters())
        self.decoder_params = list(self.vlm.decoder.parameters())

        params = sum(p.numel() for p in self.encoder_params)
        trainable_params = sum(p.numel() for p in self.encoder_params if p.requires_grad)
        self.logger.info(f"Encoder model parameters: {params}, {trainable_params} (trainable)")

        params = sum(p.numel() for p in self.decoder_params)
        trainable_params = sum(p.numel() for p in self.decoder_params if p.requires_grad)
        self.logger.info(f"Decoder model parameters: {params}, {trainable_params} (trainable)")

        params = sum(p.numel() for p in vlm.parameters())
        trainable_params = sum(p.numel() for p in vlm.parameters() if p.requires_grad)
        self.logger.info(f"Total model parameters: {params}, {trainable_params} (trainable)")

        # Configure the optimizer, use 1/8 the learning rate for all encoder parameters and different weight
        # decays for the encoder vs decoder parameters
        dec_groups = get_param_groups(self.vlm.decoder, wd_decoder, lr_start)
        if isinstance(vlm.encoder, ImageEncoder):  # If the encoder is the trainable ImageEncoder then add
            # its parameters to the optimizer for gradient updates during training
            enc_groups = get_param_groups(self.vlm.encoder, wd_encoder, lr_start * 0.125)
            self.opt = AdamW(enc_groups + dec_groups, betas=adam_betas)
        else:  # Otherwise the encoder will be the frozen CLIP model, don't add its params to the optimizer
            # since we will not want it being updated at all during training
            self.opt = AdamW(dec_groups, betas=adam_betas)

        # 5). Configure a learning rate scheduler for training with warm-up and cosine annealing
        warmup_steps = int(train_num_steps * warm_up_pct)  # Slowly ramp up the LR from very low to peak
        warmup = LinearLR(self.opt, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        # Cosine annealing of the learning rate during the rest of training
        decay = CosineAnnealingLR(self.opt, T_max=train_num_steps - warmup_steps, eta_min=lr_end)
        # Stack both the learning rate warm up and the gradual linear decay into 1 scheduler
        self.scheduler = SequentialLR(self.opt, schedulers=[warmup, decay], milestones=[warmup_steps])

        # 6). Keep track of the training step and losses along the way
        self.step = 0  # Training step counter for the current training run
        self.all_losses = []  # Aggregate loss values during training

        # 7). Load in the latest checkpoint weights to continue from where the models were last saved
        if use_latest_checkpoint > 0:  # If set to True, then use the most recent checkpoint available
            checkpoints = os.listdir(self.checkpoints_folder)
            if len(checkpoints) > 0:  # If there is a milestone saved, load in the weights
                last_checkpoint = max([int(x.replace("model-", "").replace(".pt", "")) for x in checkpoints])
                # Load in the most recent milestone to continue training
                weights_only = (use_latest_checkpoint == 2)  # Load in only the weights if set to 2
                self.load(last_checkpoint, weights_only)
            else:  # Otherwise, check if there are any pre-trained weights to use as a starting point
                max_milestone = None  # Look for checkpoints in the pre-trained weights folder instead
                if scst is False:  # If doing stage 2 supervised training, look to the MAE pre-training dir
                    pretrained_wts_dir = os.path.join(self.results_folder, "../pretrain_clip/checkpoints")
                else:  # Otherwise if doing stage 2 SCST training, look to the captioning dir
                    pretrained_wts_dir = os.path.join(self.results_folder, "../captioning/checkpoints")
                if os.path.exists(pretrained_wts_dir):
                    milestones = [int(x.replace("model-", "").replace(".pt", ""))
                                  for x in os.listdir(pretrained_wts_dir)]
                    if len(milestones) > 0:
                        max_milestone = max(milestones)
                if max_milestone is not None:  # Load in the latest pre-trained weights
                    self.load_pretrained(max_milestone)

    def save(self, milestone: int) -> None:
        """
        Saves the weights of the model for the current milestone.

        :param milestone: An integer denoting the training timestep at which the model weights were saved.
        :returns: None. Writes the weights and losses to disk.
        """
        checkpoint_path = os.path.join(self.checkpoints_folder, f"model-{milestone}.pt")
        self.logger.info(f"Saving model to {checkpoint_path}.")
        data = {"step": self.step,
                "model": self.vlm.state_dict(),
                "opt": self.opt.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                }
        torch.save(data, checkpoint_path)
        # Save down all the loss values produced by model training since the last caching
        pd.Series(self.all_losses).to_csv(os.path.join(self.losses_folder, f"losses-{milestone}.csv"))

    def load(self, milestone: int, weights_only: bool = False) -> None:
        """
        Loads in the cached weights from disk for a particular milestone.

        :param milestone: An integer denoting the training timestep at which the model weights were saved.
        :param weights_only: If True, then only the model weights are loaded from disk.
        :returns: None. Weights and other trainer state parameter values are loaded into memory.
        """
        checkpoint_path = os.path.join(self.checkpoints_folder, f"model-{milestone}.pt")
        self.logger.info(f"Loading model from {checkpoint_path}.")
        checkpoint_data = torch.load(checkpoint_path, map_location=self.device)

        # Re-instate the training step counter, model weights, and optimizer state from the checkpoint data
        # read in from disk
        self.vlm.load_state_dict(checkpoint_data["model"])
        if not weights_only:  # Load in more than just the weights
            self.step = checkpoint_data["step"]
            self.opt.load_state_dict(checkpoint_data["opt"])
            self.scheduler.load_state_dict(checkpoint_data["scheduler"])
        else:
            self.logger.info("Optimizer and scheduler not loaded")
        # Losses are not loaded in, they are saved to disk periodically with the model weights and are not
        # needed to continue training. The losses obtained by training will be cached again at the next save

        # Move the model and the optimizer to the same device to continue training or for inference
        for state in self.opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

    def load_pretrained(self, milestone: int) -> None:
        """
        Loads in pre-trained model weights form disk for a particular milestone. If self.scst is False, then
        the trainer is configured for supervised training so pre-trained weights will be located in the
        pretrain directory corresponding to the CLIP-style pre-training. If self.scst is True, then the
        trainer is configured for SCST fine-tuning so the pre-trained model weights will be located in the
        captioning directory corresponding to the VLM model.

        :param milestone: An integer denoting the training timestep at which the model weights were saved.
        :returns: None. Weights are loaded into the model from disk.
        """
        if self.scst is False:  # Stage 2 supervised training with a cross-entropy loss
            if isinstance(self.vlm.encoder,
                          ImageEncoder):  # Only if the encoder is the trainable ImageEncoder
                file_path = f"../pretrain_clip/checkpoints/model-{milestone}.pt"
                checkpoint_path = os.path.join(self.results_folder, file_path)
                self.logger.info(f"Loading pretrained encoder model weights from {checkpoint_path}.")
                checkpoint_data = torch.load(checkpoint_path, map_location=self.device)
                # Re-instate the model weights from the checkpoint data read in from disk
                self.vlm.encoder.load_state_dict(checkpoint_data["img_encoder"])
        else:  # Stage 3 SCST fine-tuning, load the pre-trained model from the captioning folder
            file_path = f"../captioning/checkpoints/model-{milestone}.pt"
            checkpoint_path = os.path.join(self.results_folder, file_path)
            self.logger.info(f"Loading pretrained VLM model weights from {checkpoint_path}.")
            checkpoint_data = torch.load(checkpoint_path, map_location=self.device)
            # Re-instate the model weights from the checkpoint data read in from disk
            self.vlm.load_state_dict(checkpoint_data["model"])

    def compute_eval_scores(self, max_len: int = 50, max_batches: int = 10, eps: float = 0.1) -> Tuple[float]:
        """
        This method computes the negative log-likelihood, perplexity, and CIDEr score on the validation data
        set using the current model weights. Calling this method periodically during training is helpful for
        tracking progress.

        :param max_len: Sets the max length of the sampled captions for CIDEr calculations.
        :param max_batches: Set the max number of batches to use from the validation dataloader.
        :returns: The average per token negative log-likelihood, perplexity, and CIDEr score evaluated on
            the validation data set.
        """
        res = {}  # Compute CIDEr using the COCO eval toolkit, record captions by image ID
        losses = []  # Compute the out-of-sample loss & perplexity as well

        self.vlm.eval()  # Set to eval mode during evaluation

        batch_count = 0
        for batch in self.dataloader_val:
            images = batch["images"].to(self.device, non_blocking=True)  # (N, C, H, W)
            captions = batch["captions"].to(self.device, non_blocking=True)  # (N, tgt_max_len)

            with torch.no_grad():  # No longer training, no gradient tracking needed
                if self.amp_dtype is not None:
                    with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                        outputs = self.vlm(images, captions)  # (N, T, V)
                        loss = self.vlm.compute_loss(outputs, captions, eps)
                        pred_captions, _ = self.vlm.sample(images, max_len, True, False, 0.0)
                else:
                    outputs = self.vlm(images, captions)  # (N, T, V)
                    loss = self.vlm.compute_loss(outputs, captions, eps)
                    pred_captions, _ = self.vlm.sample(images, max_len, True, False, 0.0)

                # Record the average per token negative log likelihood and how many obs are in this batch
                losses.append((loss.item(), images.shape[0]))

                # Record the prediction captions in a dictionary to match up against gts_val
                for pred_caption, image_name in zip(pred_captions, batch["image_names"]):
                    res[int(image_name)] = [pred_caption.lower().strip()]

            batch_count += 1
            if batch_count >= max_batches:
                break

        # Compute the weighted average loss using the batch size of each loss
        overall_loss, total_n = 0, 0
        for loss, n in losses:
            overall_loss += loss * n
            total_n += n
        val_loss = overall_loss / total_n  # Compute the avg per token negative log likelihood
        val_ppl = np.exp(val_loss)  # Compute the perplexity

        # Compute the CIDEr score using the COCO utils
        gts_val = {img_id: self.gts_val[img_id] for img_id in res.keys()}
        val_cider, _ = self.cider_scorer.compute_score(gts_val, res)

        self.vlm.train()  # Set back to train mode to re-enable dropout etc.
        return val_loss, val_ppl, val_cider

    def report_lr_wd(self):
        """
        Reports the learning rates and weight decay parameter values of the vlm model.
        """
        self.logger.info(f"Reporting learning rates and weight decay at step={self.step}")
        labels = ["Encoder Decay", "Encoder No Decay", "Decoder Decay", "Decoder No Decay"]
        for i, group in enumerate(self.opt.param_groups):  # Report all learning rates
            self.logger.info((f"{(labels[i] + ':').ljust(17)} lr = {group['lr']:.2e}, wd = "
                              f"{group['weight_decay']:.2e}, count = {len(group['params'])}"))

    def train(self, eps: float = 0.1, max_len: int = 50) -> None:
        """
        Runs the training of the model until completion for self.train_num_steps total training iterations.

        :param eps: A smoothing parameter used during decoding to smooth the probability distribution
            predicted by the model over the vocabulary space.
        :param max_len: Sets the max length of the sampled captions during training which are periodically
            printed to show the model's progress and also for the periodic eval runs on the validation set.
        :returns: None. Caches the results to disk.
        """
        if self.step < self.freeze_steps:  # Freeze the encoder params for the first initial training steps
            for p in self.encoder_params:
                p.requires_grad = False
            self.logger.info(f"Image encoder parameters are frozen at step={self.step}")
        assert all(p.requires_grad for p in self.decoder_params)  # Should be always trainable

        self.logger.info(f"Starting Captions Training, device={self.device}, amp_dtype={self.amp_dtype}")
        self.report_lr_wd()

        self.vlm.to(self.device)  # Move the model to the correct device
        self.vlm.train()  # Make sure to set the model to train mode for training

        # These data-loaders do not cache batches which makes them more memory efficient
        inf_dataloader_train = infinite_loader(self.dataloader_train)
        inf_dataloader_val = infinite_loader(self.dataloader_val)

        if self.amp_dtype is not None:
            if self.device.type != 'cuda':
                self.logger.info("AMP with FP16 requires CUDA")
                self.amp_dtype = None
            else:
                scaler = torch.amp.GradScaler('cuda')

        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:

            while self.step < self.train_num_steps:  # Run until all training iterations are complete
                # Get the next training batch and move it to the same device as the model
                batch = next(inf_dataloader_train)
                images = batch["images"].to(self.device, non_blocking=True)  # (N, C, H, W)
                captions = batch["captions"].to(self.device, non_blocking=True)  # (N, tgt_max_len)

                self.opt.zero_grad(set_to_none=True)  # Zero the grads of the opt before computing the loss
                # Compute the forward-pass through the model and compute a tensor that is the same shape
                # along the first 2 dims as captions but also gives the prob dist across the vocab
                if self.amp_dtype is not None:
                    with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                        outputs = self.vlm(images, captions)  # (N, T, V)
                        loss = self.vlm.compute_loss(outputs, captions, eps)
                else:
                    outputs = self.vlm(images, captions)  # (N, T, V)
                    loss = self.vlm.compute_loss(outputs, captions, eps)

                if self.amp_dtype == torch.float16:
                    scaler.scale(loss).backward()
                    if self.grad_clip is not None:
                        scaler.unscale_(self.opt)  # Unscale before clipping
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.vlm.parameters(), self.grad_clip)
                    scaler.step(self.opt)  # Update the model parameters by taking a gradient step
                    scaler.update()
                else:
                    loss.backward()
                    if self.grad_clip is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.vlm.parameters(), self.grad_clip)
                    self.opt.step()  # Update the model parameters by taking a gradient step

                pbar.set_postfix(loss=f"{loss.item():.4f}", ppl=f"{np.exp(loss.item()):.2f}",
                                 grad=f"{grad_norm:.3f}", logit_std=f"{outputs.std(dim=-1).mean():.3f}")

                self.scheduler.step()  # Update the learning rate scheduler

                # pbar.set_description(f"loss: {loss.item():.4f}, perplexity: {np.exp(loss.item()):.2f}")
                self.all_losses.append(loss.item())  # Aggregate all the loss values for each timestep

                self.step += 1

                # Periodically save the model weights to disk
                if self.step % self.save_every == 0 or self.step == self.train_num_steps:
                    self.save(self.step)
                    plot_and_save_loss(self.losses_folder)  # Generate a new plot of the training losses
                    self.all_losses = []  # Clear the list of losses after each save, store only the ones
                    # from the last save to the next save
                    torch.cuda.empty_cache()

                # Periodically compute performance metrics on the eval dataset
                if self.step % self.eval_every == 0 or self.step == self.train_num_steps:
                    val_loss, val_ppl, val_cider = self.compute_eval_scores(max_len)
                    msg = f"Validation Set NLL: {val_loss:.3f} PPL: {val_ppl:.3f} CIDEr: {val_cider:.3f}"
                    self.logger.info(msg)
                    update_cache_and_plot(self.step, val_ppl, self.results_folder, "val_ppl")
                    update_cache_and_plot(self.step, val_cider, self.results_folder, "val_cider")

                # Periodically generate samples from the model
                if self.step % self.sample_every == 0 or self.step == self.train_num_steps:
                    # Periodically log the loss and other training metrics
                    self.logger.info((f"loss={loss.item():.4f}, ppl={np.exp(loss.item()):.2f}, "
                                      f"grad={grad_norm:.3f}, logit_std={outputs.std(dim=-1).mean():.3f}"))
                    self.report_lr_wd()
                    gpu_mem_used = torch.cuda.memory_allocated() / 1e9
                    cpu_mem_used = psutil.virtual_memory().used / 1e9
                    msg = f"[GPU, CPU] Memory Allocated: {gpu_mem_used:.2f}GB {cpu_mem_used:.2f}GB"
                    self.logger.info(msg)

                    self.logger.info("\n")
                    self.logger.info(f"Generating samples at step={self.step}")
                    batch = next(inf_dataloader_val)  # Get the next batch of data from the validation set
                    indices = torch.randperm(batch["images"].size(0))[:self.num_sample]
                    images = batch["images"][indices].to(self.device)
                    captions = batch["captions"][indices].to(self.device)
                    image_names = [batch["image_names"][int(idx)] for idx in indices]
                    if self.amp_dtype is not None:
                        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                            pred_captions, _ = self.vlm.sample(images, max_len, True, False, 0.0)  # (N, T)
                    else:
                        pred_captions, _ = self.vlm.sample(images, max_len, True, False, 0.0)  # (N, T)
                    actual_captions = [decode_caption(x, self.vlm.sp_model) for x in captions]
                    # Print some side-by-side comparisons of the predicted vs actual captions
                    for yhat, y, img_name in zip(pred_captions, actual_captions, image_names):
                        self.logger.info(f"Image:     {img_name}")
                        self.logger.info(f"Predicted: {yhat}")
                        self.logger.info(f"Actual:    {y}")
                    self.logger.info("\n")
                    self.vlm.train()  # Switch back to training mode once finished

                # Check if beyond the initial freeze steps period to unfreeze the encoder params
                if self.step == self.freeze_steps:  # Unfreeze the parameters of the image encoder
                    for p in self.encoder_params:
                        p.requires_grad = True
                    self.logger.info(f"Image encoder parameters unfrozen at step={self.step}")
                del outputs, loss, images, captions
                torch.cuda.empty_cache()

                pbar.update(1)

    def train_scst(self, eps: float = 0.1, max_len: int = 50, lambda_xe: float = 0.1) -> None:
        """
        Runs the training of the model until completion for self.train_num_steps total training iterations
        using a Self-Critical Sequence Training (SCST) approach to improve CIDEr scores.

        :param eps: A smoothing parameter used during decoding to smooth the probability distribution
            predicted by the model over the vocabulary space.
        :param max_len: Sets the max length of the sampled captions during training which are periodically
            printed to show the model's progress and also for the periodic eval runs on the validation set.
        :param lambda_xe: Allows the SCST loss to be augmented with lambda_xe * xe_loss for stability. Set to
            0.0 to skip entirely. This parameter sets the max value which is annealed to zero during training.
        :returns: None. Caches the results to disk.
        """
        if self.step < self.freeze_steps:  # Freeze the encoder params for the first initial training steps
            for p in self.encoder_params:
                p.requires_grad = False
            self.logger.info(f"Image encoder parameters are frozen at step={self.step}")
        assert all(p.requires_grad for p in self.decoder_params)  # Should be always trainable

        # Turn off dropout during SCST training in both the encoder and decoder networks
        for module in self.vlm.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0

        self.logger.info(f"Starting SCST Training, device={self.device}, amp_dtype={self.amp_dtype}")
        self.report_lr_wd()

        self.vlm.to(self.device)  # Move the model to the correct device
        self.vlm.train()  # Make sure to set the model to train mode for training

        # These data-loaders do not cache batches which makes them more memory efficient
        inf_dataloader_train = infinite_loader(self.dataloader_train)
        inf_dataloader_val = infinite_loader(self.dataloader_val)

        if self.amp_dtype is not None:
            if self.device.type != 'cuda':
                self.logger.info("AMP with FP16 requires CUDA")
                self.amp_dtype = None
            else:
                scaler = torch.amp.GradScaler('cuda')

        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:

            while self.step < self.train_num_steps:  # Run until all training iterations are complete
                # Get the next training batch and move it to the same device as the model
                batch = next(inf_dataloader_train)
                captions_gt = {int(img_id): self.gts_train[int(img_id)] for img_id in batch["image_names"]}
                images = batch["images"].to(self.device, non_blocking=True)  # (N, C, H, W)
                captions = batch["captions"].to(self.device, non_blocking=True)  # (N, tgt_max_len)

                self.opt.zero_grad(set_to_none=True)  # Zero the grads of the opt before computing the loss
                # Compute the forward-pass through the model and compute a tensor that is the same shape
                # along the first 2 dims as captions but also gives the prob dist across the vocab

                # 1). Generate baseline image captions using greedy decoding, produce a list of strings
                with torch.no_grad():
                    if self.amp_dtype is not None:
                        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                            greedy_captions, logprobs_g = self.vlm.sample(images, max_len=50,
                                                                          return_strings=True,
                                                                          track_gradients=False, temp=0.0)
                    else:
                        greedy_captions, logprobs_g = self.vlm.sample(images, max_len=50, return_strings=True,
                                                                      track_gradients=False, temp=0.0)
                # Convert to a dict with structure: {image_id: [string caption]} for CIDEr eval
                greedy_captions = {int(img_id): [c.lower().strip()] for img_id, c
                                   in zip(batch["image_names"], greedy_captions)}

                # 2). Sample exploratory captions i.e. the policy rollout - track gradients here, gives us
                # sequences sampled from the distribution of next predicted words, not just the greedy
                # selection of the top most probably each step
                if self.amp_dtype is not None:
                    with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                        sampled_captions, logprobs_sum = self.vlm.sample(images, max_len=50,
                                                                         return_strings=True,
                                                                         track_gradients=True, temp=0.0)
                else:
                    sampled_captions, logprobs_sum = self.vlm.sample(images, max_len=50, return_strings=True,
                                                                     track_gradients=True, temp=0.0)
                # sampled_seqs is a list of strings of length N and sampled_logprobs is a torch.Tensor of
                # float values of length N which has gradient tracking for computing the loss below
                # Convert to a dict with structure: {image_id: [string caption]} for CIDEr eval
                sampled_captions = {int(img_id): [c.lower().strip()] for img_id, c
                                    in zip(batch["image_names"], sampled_captions)}

                # 3). Compute rewards, the CIDEr difference between the greedy and exploratory captions
                # The CIDEr scorer expects a dictionary of format {image_id: [string1, string2 ...], ...}
                greedy_rewards = self.cider_scorer.compute_score(captions_gt, greedy_captions)[1]
                sampled_rewards = self.cider_scorer.compute_score(captions_gt, sampled_captions)[1]

                # Advantage = sampled - baseline (hence self-critical)
                advantages = sampled_rewards - greedy_rewards  # Size N = batch_size
                # Normalize the advantages to reduce variance
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                advantages = np.clip(advantages, -2.0, 2.0)  # Add extreme value clipping
                advantages = torch.tensor(advantages, device=logprobs_sum.device)  # Move to torch tensor

                # 4). Compute the SCST loss = (-1) * mean( advantage * log_prob )
                loss = (-1) * (advantages * logprobs_sum).mean()

                # 5). Compute a small cross-entropy loss as well for stability (a hybrid loss)
                if lambda_xe > 0:  # If zero, then no weight so skip computing
                    if self.amp_dtype is not None:
                        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                            outputs = self.vlm(images, captions)  # (N, T, V)
                            xe_loss = self.vlm.compute_loss(outputs, captions, eps)
                    else:
                        outputs = self.vlm(images, captions)  # (N, T, V)
                        xe_loss = self.vlm.compute_loss(outputs, captions, eps)

                    # Combine the SCST loss with the cross-entropy loss, anneal towards 0 during training
                    loss = loss + (lambda_xe * 1 - (self.step + 1) / self.train_num_steps) * xe_loss
                    del outputs, xe_loss  # Free up memory when finished

                if abs(loss.item()) > 1e-3: # Trigger early stopping if the loss is too small
                    if self.amp_dtype == torch.float16:
                        scaler.scale(loss).backward()
                        if self.grad_clip is not None:
                            scaler.unscale_(self.opt)  # Unscale before clipping
                            grad_norm = torch.nn.utils.clip_grad_norm_(self.vlm.parameters(), self.grad_clip)
                        scaler.step(self.opt)  # Update the model parameters by taking a gradient step
                        scaler.update()
                    else:
                        loss.backward()
                        if self.grad_clip is not None:
                            grad_norm = torch.nn.utils.clip_grad_norm_(self.vlm.parameters(), self.grad_clip)
                    self.opt.step()  # Update the model parameters by taking a gradient step
                    early_stop = False
                else:
                    early_stop = True

                pbar.set_postfix(loss=f"{loss.item():.4f}", grad=f"{grad_norm:.3f}",
                                 mean_greedy_CIDEr=f"{greedy_rewards.mean():.3f}",
                                 mean_sampled_CIDEr=f"{sampled_rewards.mean():.3f}")

                self.scheduler.step()  # Update the learning rate scheduler
                self.all_losses.append(loss.item())  # Aggregate all the loss values for each timestep

                self.step += 1

                # Periodically save the model weights to disk
                if self.step % self.save_every == 0 or self.step == self.train_num_steps or early_stop:
                    self.save(self.step)
                    plot_and_save_loss(self.losses_folder)  # Generate a new plot of the training losses
                    self.all_losses = []  # Clear the list of losses after each save, store only the ones
                    # from the last save to the next save
                    torch.cuda.empty_cache()

                # Periodically compute performance metrics on the eval dataset
                if self.step % self.eval_every == 0 or self.step == self.train_num_steps or early_stop:
                    val_loss, val_ppl, val_cider = self.compute_eval_scores(max_len)
                    msg = f"Validation Set NLL: {val_loss:.3f} PPL: {val_ppl:.3f} CIDEr: {val_cider:.3f}"
                    self.logger.info(msg)

                # Periodically generate samples from the model
                if self.step % self.sample_every == 0 or self.step == self.train_num_steps or early_stop:
                    # Periodically log the loss and other training metrics
                    self.logger.info((f"loss={loss.item():.4f}, grad={grad_norm:.3f}, "
                                      f"mean_greedy_cider={greedy_rewards.mean():.3f}"))
                    self.report_lr_wd()
                    gpu_mem_used = torch.cuda.memory_allocated() / 1e9
                    cpu_mem_used = psutil.virtual_memory().used / 1e9
                    msg = f"[GPU, CPU] Memory Allocated: {gpu_mem_used:.2f}GB {cpu_mem_used:.2f}GB"
                    self.logger.info(msg)

                    self.logger.info("\n")
                    self.logger.info(f"Generating samples at step={self.step}")
                    batch = next(inf_dataloader_val)  # Get the next batch of data from the validation set
                    indices = torch.randperm(batch["images"].size(0))[:self.num_sample]
                    images = batch["images"][indices].to(self.device)
                    captions = batch["captions"][indices].to(self.device)
                    image_names = [batch["image_names"][int(idx)] for idx in indices]
                    if self.amp_dtype is not None:
                        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                            pred_captions, _ = self.vlm.sample(images, max_len, True, False, 0.0)
                    else:
                        pred_captions, _ = self.vlm.sample(images, max_len, True, False, 0.0)
                    actual_captions = [decode_caption(x, self.vlm.sp_model) for x in captions]
                    # Print some side-by-side comparisons of the predicted vs actual captions
                    for yhat, y, img_name in zip(pred_captions, actual_captions, image_names):
                        self.logger.info(f"Image:     {img_name}")
                        self.logger.info(f"Predicted: {yhat}")
                        self.logger.info(f"Actual:    {y}")
                    self.logger.info("\n")
                    self.vlm.train()  # Switch back to training mode once finished

                # Check if beyond the initial freeze steps period to unfreeze the encoder params
                if self.step == self.freeze_steps:  # Unfreeze the parameters of the image encoder
                    for p in self.encoder_params:
                        p.requires_grad = True
                    self.logger.info(f"Image encoder parameters unfrozen at step={self.step}")

                if early_stop: # Trigger early stopping, end the training loop
                    return None

                del batch, captions_gt, images, captions, greedy_captions, logprobs_g, sampled_captions
                del logprobs_sum, greedy_rewards, sampled_rewards, advantages, loss

                pbar.update(1)
