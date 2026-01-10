"""
This script is used to train the Vision-Language Model for caption generation.
"""
import sys, os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, CURRENT_DIR)

import sentencepiece as spm
from torch_models import VisionLanguageTransformer
from dataset_utils import get_dataloader
from vlm_trainer import Trainer
from utils import get_device
from typing import Dict
import argparse


def train_model(config: Dict) -> None:
    """
    Runs training for the model using the configurations specified in the config file which can contain
    configurations for the Vision-Language model and the Trainer objects.

    :param config: A config dictionary containing parameters for various aspects of the training loop.
    :returns: None. Results are saved to disk.
    """
    # 1). Read in the sub-word sentence piece vocab model derived from the training captions
    sp_model = spm.SentencePieceProcessor()
    sp_model.load(os.path.join(CURRENT_DIR, "dataset/preprocessed/vocab.model"))

    # 2).Init the Vision-Language transformer model
    vlm = VisionLanguageTransformer(sp_model=sp_model, **config.get("VisionLanguageTransformer", {}))
    # print("Number of parameters:", sum(p.numel() for p in vlm.parameters()))

    # 3). Construct the COCO training dataset loader and validation dataset loader
    dataloader_train = get_dataloader(split='train', **config.get("DataLoader", {}))
    dataloader_val = get_dataloader(split='val', **config.get("DataLoader", {}))

    # 4). Configure the training pipeline with the trainer object
    trainer = Trainer(vlm, dataloader_train, dataloader_val, **config.get("Trainer", {}))

    # 5). Train the model to completion
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Pipeline Module")
    parser.add_argument("--debug", help="Set to True to run in debug mode")
    args = parser.parse_args()
    debug = args.debug.lower() == "true"

    if debug:  # Set up a debug config file to test the training pipeline locally
        config = {
            "VisionLanguageTransformer": {
                "img_size": 224,
                "patch_size": 28,
                "in_channels": 3,
                "embed_dim": 128,
                "num_layers": 1,
                "num_heads": 4,
                "ffn_dim": 256,
                "dropout": 0.1,
            },
            "DataLoader": {
                "batch_size": 32,
                "device": get_device().type,
                "dataset_dir": f"{CURRENT_DIR}/dataset/preprocessed/",
            },
            "Trainer": {
                "lr": 5e-4,
                "weight_decay": 1e-3,
                "train_num_steps": 300000,
                "grad_clip": 1.0,
                "sample_every": 500,
                "save_every": 5000,
                "results_folder": f"{CURRENT_DIR}/results",
                "use_amp": True,
                "use_latest_checkpoint": True,
            }
        }
    else:  # Set up a config for prod training, set parameters for each component
        config = {
            "VisionLanguageTransformer": {
                "img_size": 224,
                "patch_size": 14,
                "in_channels": 3,
                "embed_dim": 512,
                "num_layers": 3,
                "num_heads": 8,
                "ffn_dim": 1024,
                "dropout": 0.1,
            },
            "DataLoader": {
                "batch_size": 512,
                "device": get_device().type,
                "dataset_dir": f"{CURRENT_DIR}/dataset/preprocessed/",
            },
            "Trainer": {
                "lr": 5e-4,
                "weight_decay": 1e-3,
                "train_num_steps": 300000,
                "grad_clip": 1.0,
                "sample_every": 500,
                "save_every": 5000,
                "results_folder": f"{CURRENT_DIR}/results",
                "use_amp": True,
                "use_latest_checkpoint": True,
            }
        }

    train_model(config)  # Run model training
