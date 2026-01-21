"""
This script is used to train the Vision-Language Model for caption generation.
"""
import sys, os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, CURRENT_DIR)

import sentencepiece as spm
from torch_models import VisionLanguageTransformer, MAEdecoder
from dataset_utils import get_dataloader
from vlm_trainer import TrainerCaptioning, TrainerMAE
from utils import get_device
from typing import Dict
import argparse


def pre_train_mae(config: dict) -> None:
    """
    Runs MAE pre-training a vision-language model using the VisionLanguageTransformer and TrainerMAE classes
    with the configurations specified in the config file.

    :param config: A config dictionary containing parameters for various aspects of the training loop and
        model paramters for how to configure the model parameters.
    :returns: None. Results are saved to disk.
    """
    # 1). Read in the sub-word sentence piece vocab model derived from the training captions
    sp_model = spm.SentencePieceProcessor()
    sp_model.load(os.path.join(CURRENT_DIR, "dataset/preprocessed/vocab.model"))

    # 2).Init the Vision-Language transformer model and decoder model
    vlm = VisionLanguageTransformer(sp_model=sp_model, **config.get("VisionLanguageTransformer", {}))
    decoder = MAEdecoder(**config.get("MAEdecoder", {}))

    # 3). Construct the COCO training dataset loader and validation dataset loader
    dataloader_train = get_dataloader(split='train', include_captions=True,
                                      **config.get("DataLoaderTrain", {}))
    dataloader_val = get_dataloader(split='val', include_captions=False,
                                    **config.get("DataLoaderVal", {}))

    # 4). Configure the training pipeline with the trainer object
    trainer = TrainerMAE(vlm, decoder, dataloader_train, dataloader_val, **config.get("TrainerMAE", {}))

    # 5). Train the model to completion
    trainer.train()


def train_captioning_model(config: Dict) -> None:
    """
    Runs end-to-end image captioning training for a vision-language model using the VisionLanguageTransformer
    and TrainerCaptioning classes with the configurations specified in the config file.

    :param config: A config dictionary containing parameters for various aspects of the training loop and
        model paramters for how to configure the model parameters.
    :returns: None. Results are saved to disk.
    """
    # 1). Read in the sub-word sentence piece vocab model derived from the training captions
    sp_model = spm.SentencePieceProcessor()
    sp_model.load(os.path.join(CURRENT_DIR, "dataset/preprocessed/vocab.model"))

    # 2).Init the Vision-Language transformer model
    vlm = VisionLanguageTransformer(sp_model=sp_model, **config.get("VisionLanguageTransformer", {}))

    # 3). Construct the COCO training dataset loader and validation dataset loader
    dataloader_train = get_dataloader(split='train', include_captions=True,
                                      **config.get("DataLoaderTrain", {}))
    dataloader_val = get_dataloader(split='val', include_captions=True,
                                    **config.get("DataLoaderVal", {}))

    # 4). Configure the training pipeline with the trainer object
    trainer = TrainerCaptioning(vlm, dataloader_train, dataloader_val, **config.get("TrainerCaptioning", {}))

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
                "patch_size": 16,
                "in_channels": 3,
                "embed_dim": 64,
                "num_layers": 2,
                "num_heads": 2,
                "ffn_dim": 64 * 2,
                "dropout": 0.1,
            },
            "MAEdecoder": {
                "img_size": 224,
                "patch_size": 16,
                "in_channels": 3,
                "embed_dim": 64,
                "num_layers": 1,
                "num_heads": 2,
                "ffn_dim": 64 * 2,
                "dropout": 0.1,
            },
            "DataLoaderTrain": {
                "batch_size": 32,
                "device": get_device().type,
                "dataset_dir": f"{CURRENT_DIR}/dataset/preprocessed/",
                "add_augmentation": True,
            },
            "DataLoaderVal": {
                "batch_size": 4,
                "device": get_device().type,
                "dataset_dir": f"{CURRENT_DIR}/dataset/preprocessed/",
                "add_augmentation": False,
            },
            "TrainerMAE": {
                "lr_start": 1e-4,
                "lr_end": 1e-5,
                "weight_decay": 5e-3,
                "train_num_steps": 100,
                "grad_clip": 1.0,
                "sample_every": 5,
                "save_every": 10,
                "results_folder": f"{CURRENT_DIR}/debug/pretrain",
                "use_amp": True,
                "use_latest_checkpoint": True,
            },
            "TrainerCaptioning": {
                "lr_start": 1e-4,
                "lr_end": 1e-5,
                "weight_decay": 5e-3,
                "train_num_steps": 50,
                "grad_clip": 1.0,
                "sample_every": 25,
                "save_every": 25,
                "results_folder": f"{CURRENT_DIR}/debug/captioning",
                "use_amp": True,
                "use_latest_checkpoint": True,
            },
        }

    else:  # Set up a config for prod training, set parameters for each component
        config = {
            "VisionLanguageTransformer": {
                "img_size": 224,
                "patch_size": 16,
                "in_channels": 3,
                "embed_dim": 768,
                "num_layers": 8,
                "num_heads": 8,
                "ffn_dim": 768 * 2,
                "dropout": 0.1,
            },
            "MAEdecoder": {
                "img_size": 224,
                "patch_size": 16,
                "in_channels": 3,
                "embed_dim": 768,
                "num_layers": 4,
                "num_heads": 8,
                "ffn_dim": 768 * 2,
                "dropout": 0.1,
            },
            "DataLoaderTrain": {
                "batch_size": 32,
                "device": get_device().type,
                "dataset_dir": f"{CURRENT_DIR}/dataset/preprocessed/",
                "add_augmentation": True,
            },
            "DataLoaderVal": {
                "batch_size": 4,
                "device": get_device().type,
                "dataset_dir": f"{CURRENT_DIR}/dataset/preprocessed/",
                "add_augmentation": False,
            },
            "TrainerMAE": {
                "lr_start": 1e-4,
                "lr_end": 1e-5,
                "weight_decay": 5e-3,
                "train_num_steps": 200000,
                "grad_clip": 1.0,
                "sample_every": 500,
                "save_every": 5000,
                "results_folder": f"{CURRENT_DIR}/results/pretrain",
                "use_amp": True,
                "use_latest_checkpoint": True,
            },
            "TrainerCaptioning": {
                "lr_start": 1e-4,
                "lr_end": 1e-5,
                "weight_decay": 5e-3,
                "train_num_steps": 200000,
                "grad_clip": 1.0,
                "sample_every": 500,
                "save_every": 5000,
                "results_folder": f"{CURRENT_DIR}/results/captioning",
                "use_amp": True,
                "use_latest_checkpoint": True,
            },
        }

    # pre_train_mae(config)  # Run model pre-training
    train_captioning_model(config)  # Run model training
