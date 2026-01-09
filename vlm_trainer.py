import os, itertools, sys
import numpy as np
import torch
from tqdm.auto import tqdm
from torch.optim import AdamW
from typing import Tuple
from utils import get_device, get_amp_dtype, decode_caption
import logging
import psutil
from pathlib import Path
from torch.utils.data import DataLoader
from torch_models import VisionLanguageTransformer

def infinite_loader(dl):
    while True:
        for batch in dl:
            yield batch

class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)


class Trainer:
    def __init__(self, vlm: VisionLanguageTransformer, dataloader_train: DataLoader,
                 dataloader_val: DataLoader, lr: float = 1e-3, weight_decay: float = 1e-3,
                 train_num_steps: int = 100000, adam_betas: Tuple[float] = (0.9, 0.99),
                 grad_clip: float = 1.0, sample_every: int = 1000, save_every: int = 5000,
                 results_folder: str = None, use_amp: bool = False, use_latest_checkpoint: bool = True):
        """
        A frame work for training a Vision-Language Model (VLT). This class wrapper has methods for loading
        a model from a recent checkpoint, saving a model periodically during training, and running a training
        loop to train from scratch or to continue from the last checkpoint.

        :param vlm: A vision-language model for captioning implemented in pytorch.
        :param dataloader_train: A data loader object that will yield the required training batches.
        :param dataloader_val: A data loader object that will yield the required validation batches.
        :param lr: The training learning rate.
        :param weight_decay: The weight_decay to provide to the Adam optimizer for L2 regularization.
        :param train_num_steps: The number of training steps to run in total.
        :param adam_betas: Beta parameters for the adam optimizer.
        :param grad_clip: The amount of gradient clipping to use during training.
        :param sample_every: An int denoting how often to sample and save outputs from the model.
        :param save_every: An int denoting how often to save the model weights.
        :param results_folder: A location to save the result of the training.
        :param use_amp: Whether to use automatic mixed-precision type casting during training.
        :param use_latest_checkpoint: If set to True, then the latest checkpoint detected in the results
            directory will be loaded in before training begins to pick up from where it was last left off.
        """
        super().__init__()

        assert results_folder is not None, "You must specify results folder to save the outputs"

        self.results_folder = results_folder  # A directory where the checkpoints will be saved
        os.makedirs(self.results_folder, exist_ok=True)  # Create the directory if not already there

        self.checkpoints_folder = os.path.join(self.results_folder, "checkpoints/")
        os.makedirs(self.checkpoints_folder, exist_ok=True)  # Create the directory if not already there

        # Set up logging during training
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:  # Prevent duplicate handlers
            file_handler = logging.FileHandler(os.path.join(self.results_folder, "train.log"),
                                               encoding="utf-8")
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            )
            # file_handler.stream = sys.stdout  # Ensure UTF-8 capable stream
            self.logger.addHandler(file_handler)

            tqdm_handler = TqdmLoggingHandler()
            tqdm_handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            )
            # tqdm_handler.stream = sys.stdout  # Ensure UTF-8 capable stream
            self.logger.addHandler(tqdm_handler)
        self.logger.propagate = False

        self.vlm = vlm  # The vision-language model
        self.logger.info(f"Number of model parameters: {sum(p.numel() for p in vlm.parameters())}")
        self.device = get_device()  # Auto-detect what device to use for training
        self.grad_clip = grad_clip  # The amount of gradient clipping to use during training
        self.amp_dtype = get_amp_dtype(self.device.type) if use_amp else None
        self.save_every = save_every  # The frequency of saving model weights
        self.num_samples = 5  # The number of samples to generate periodically print
        self.sample_every = sample_every  # How often to generate samples
        self.train_num_steps = train_num_steps  # The total number of training steps

        # Set the dataset and dataloader
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val

        # Configure the optimizer for training
        self.opt = AdamW(self.vlm.parameters(), lr=lr, betas=adam_betas, weight_decay=weight_decay)

        self.step = 0  # Training step counter
        self.all_losses = []  # Aggregate loss values during training

        if use_latest_checkpoint:
            checkpoints = os.listdir(self.checkpoints_folder)
            if len(checkpoints) > 0:
                last_checkpoint = max([int(x.replace("model-", "").replace(".pt", "")) for x in checkpoints])
                self.load(last_checkpoint)  # Load in the most recent milestone to continue training

    def save(self, milestone: int) -> None:
        """
        Saves the weights of the model for the current milestone.

        :param milestone: An integer denoting the training timestep at which the model weights were saved.
        :returns: None.
        """
        checkpoint_path = os.path.join(self.checkpoints_folder, f"model-{milestone}.pt")
        self.logger.info(f"Saving model to {checkpoint_path}.")
        data = {"step": self.step,
                "all_losses": self.all_losses, # This saves only the losses computed between save() calls
                "model": self.vlm.state_dict(),
                "opt": self.opt.state_dict(),
                }
        torch.save(data, checkpoint_path)

    def load(self, milestone: int) -> None:
        """
        Loads in the cached weights from disk for a particular milestone.

        :param milestone: An integer denoting the training timestep at which the model weights were saved.
        :returns: None.
        """
        checkpoint_path = os.path.join(self.checkpoints_folder, f"model-{milestone}.pt")
        self.logger.info(f"Loading model from {checkpoint_path}.")
        checkpoint_data = torch.load(checkpoint_path, map_location=self.device)

        # Re-instate the training step counter, loss values, model weights and optimizer state from the
        # checkpoint data read in from disk
        self.step = checkpoint_data["step"]
        self.vlm.load_state_dict(checkpoint_data["model"])
        self.opt.load_state_dict(checkpoint_data["opt"])
        # We don't load the all_losses part of the saved model state, we will begin a new all_losses list
        # and save whatever is generated with the next checkpoint so that each checkpoint only contains the
        # losses since the prior checkpoint save

        # Move the model and the optimizer to the same device to continue training or for inference
        for state in self.opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

    def train(self, eps: float = 0.0, max_len: int = 75) -> None:
        """
        Runs the training of the model until completion for self.train_num_steps total training iterations.

        :param eps: A smoothing parameter used during decoding to smooth the probability distribution
            predicted by the model over the vocabulary space.
        :param max_len: Sets the max length of the sampled captions during training which are periodically
            printed to show the model's progress.
        :returns: None. Caches the results to disk.
        """
        self.logger.info(f"Starting Training, device={self.device}, amp_dtype={self.amp_dtype}")
        self.vlm.to(self.device)  # Move the model to the correct device
        self.vlm.train()  # Make sure to set the model to train mode for training

        inf_dataloader_train = infinite_loader(self.dataloader_train) # This does not cache batches
        inf_dataloader_val = infinite_loader(self.dataloader_val) # This does not cache batches

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
                images = batch["images"].to(self.device, non_blocking=True)
                captions = batch["captions"].to(self.device, non_blocking=True)

                self.opt.zero_grad(set_to_none=True)  # Zero the gradients of the optimizer before computing the loss
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
                        torch.nn.utils.clip_grad_norm_(self.vlm.parameters(), self.grad_clip)
                    scaler.step(self.opt)  # Update the model parameters by taking a gradient step
                    scaler.update()
                else:
                    loss.backward()
                    if self.grad_clip is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.vlm.parameters(), self.grad_clip)
                    self.opt.step()  # Update the model parameters by taking a gradient step

                pbar.set_postfix(loss=f"{loss.item():.4f}", ppl=f"{np.exp(loss.item()):.2f}",
                                 grad=f"{grad_norm:.3f}", logit_std=f"{outputs.std(dim=-1).mean():.3f}")

                if self.step % 500 == 0: # Periodically log the loss and other training metrics
                    self.logger.info((f"loss={loss.item():.4f}, ppl={np.exp(loss.item()):.2f}, "
                                      f"grad={grad_norm:.3f}, logit_std={outputs.std(dim=-1).mean():.3f}"))
                    gpu_mem_used = torch.cuda.memory_allocated() / 1e9
                    cpu_mem_used = psutil.virtual_memory().used / 1e9
                    msg = f"[GPU, CPU] Memory Allocated: {gpu_mem_used:.2f}GB {cpu_mem_used:.2f}GB"
                    self.logger.info(msg)

                # pbar.set_description(f"loss: {loss.item():.4f}, perplexity: {np.exp(loss.item()):.2f}")
                self.all_losses.append(loss.item())  # Aggregate all the loss values for each timestep

                self.step += 1

                if self.step % self.save_every == 0:  # Periodically save the model weights to disk
                    self.save(self.step)
                    torch.cuda.empty_cache()

                if self.step % self.sample_every == 0:  # Periodically generate samples from the model
                    self.logger.info("\n")
                    self.logger.info(f"Generating samples at step={self.step}")
                    batch = next(inf_dataloader_val)  # Get the next batch of data from the validation set
                    images = batch["images"][:self.num_samples].to(self.device)
                    captions = batch["captions"][:self.num_samples].to(self.device)
                    image_names = batch["image_names"]  # So that the corresponding images can be located
                    pred_captions = self.vlm.sample(images, max_len, True)  # (N, T)
                    actual_captions = [decode_caption(x, self.vlm.sp_model) for x in captions]
                    # Print some side-by-side comparisons of the predicted vs actual captions
                    for yhat, y, img_name in zip(pred_captions, actual_captions, image_names):
                        self.logger.info(f"Image:     {img_name}")
                        self.logger.info(f"Predicted: {yhat}")
                        self.logger.info(f"Actual:    {y}")
                    self.logger.info("\n")
                    self.vlm.train()  # Switch back to training mode once finished

                del outputs, loss, images, captions

                # if self.step % 50 == 0: # For debugging GPU RAM usage during training
                #     print(torch.cuda.memory_allocated() / 1e9, "GB")

                pbar.update(1)
