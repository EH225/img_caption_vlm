"""
This script is used to train the Vision-Language Model for caption generation.
"""
import sys, os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, CURRENT_DIR)

import sentencepiece as spm
from torch_models import ImageEncoder, MAEdecoder, LanguageDecoder, VisionLanguageModel
from dataset_utils import get_dataloader
from vlm_trainer import TrainerCaptioning, TrainerMAE
from utils import get_device
from typing import Dict
import argparse, shutil, json


def pre_train_mae(config: dict) -> None:
    """
    Runs MAE pre-training on vision-transformer encoder model using the ImageEncoder, MAEdecoder, and
    TrainerMAE classes with the configurations specified in the config file.

    :param config: A config dictionary containing parameters for various aspects of the training loop and
        model parameters for how to configure the model parameters.
    :returns: None. Results are saved to disk.
    """
    # 1). Read in the sub-word sentence piece vocab model derived from the training captions
    sp_model = spm.SentencePieceProcessor()
    sp_model.load(os.path.join(config["dataset_dir"], "vocab.model"))

    # 2).Init the encoder and decoder model for training
    encoder = ImageEncoder(**config.get("ImageEncoder", {}))
    decoder = MAEdecoder(**config.get("MAEdecoder", {}))

    # 3). Construct the COCO training dataset loader and validation dataset loader
    dataloader_train = get_dataloader(split='train', include_captions=True,
                                      **config.get("DataLoaderTrain", {}))
    dataloader_val = get_dataloader(split='val', include_captions=False,
                                    **config.get("DataLoaderVal", {}))

    # 4). Configure the training pipeline with the TrainerMAE object
    trainer = TrainerMAE(encoder, decoder, dataloader_train, dataloader_val, **config.get("TrainerMAE", {}))

    # 5). Train the model to completion
    trainer.train(mask_ratio=config["TrainerMAE"].get("mask_ratio", 0.75))


def train_captioning_model(config: Dict) -> None:
    """
    Runs end-to-end image captioning training for a vision-language model using the VisionLanguageModel
    and TrainerCaptioning classes with the configurations specified in the config file.

    :param config: A config dictionary containing parameters for various aspects of the training loop and
        model parameters for how to configure the model parameters.
    :returns: None. Results are saved to disk.
    """
    # 1). Read in the sub-word sentence piece vocab model derived from the training captions
    sp_model = spm.SentencePieceProcessor()
    sp_model.load(os.path.join(config["dataset_dir"], "vocab.model"))

    # 2).Init the encoder and decoder model for training and use them to create a vision-language model
    vlm = VisionLanguageModel(encoder=ImageEncoder(**config.get("ImageEncoder", {})),
                              decoder=LanguageDecoder(sp_model=sp_model, **config.get("LanguageDecoder", {})))

    # 3). Construct the COCO training dataset loader and validation dataset loader
    dataloader_train = get_dataloader(split='train', include_captions=True,
                                      **config.get("DataLoaderTrain", {}))
    dataloader_val = get_dataloader(split='val', include_captions=True,
                                    **config.get("DataLoaderVal", {}))
    with open(os.path.join(config["dataset_dir"], 'gts_val.json'), 'r') as f:
        gts_val = json.load(f)  # Load in the dictionary of ground-truth val set captions

    # 4). Configure the training pipeline with the trainer object
    trainer = TrainerCaptioning(vlm, dataloader_train, dataloader_val, gts_val,
                                **config.get("TrainerCaptioning", {}))

    # 5). Train the model to completion
    trainer.train(eps=config["TrainerCaptioning"].get("eps", 0.1),
                  max_len=config["TrainerCaptioning"].get("max_len", 50))


DATASET_DIR = f"{CURRENT_DIR}/dataset/preprocessed/"

#### Config for local debugging
DEBUG_CONFIG = {
    "clear_dir": False,
    "dataset_dir": DATASET_DIR,
    "ImageEncoder": {
        "img_size": 224,
        "patch_size": 16,
        "in_channels": 3,
        "embed_dim": 64,
        "num_layers": 2,
        "num_heads": 8,
        "ffn_dim": 64 * 4,
        "dropout": 0.1,
        "ffn_dropout": 0.2,
    },
    "MAEdecoder": {
        "img_size": 224,
        "patch_size": 16,
        "in_channels": 3,
        "embed_dim": 64,
        "num_layers": 1,
        "num_heads": 8,
        "ffn_dim": 64 * 4,
        "dropout": 0.1,
        "ffn_dropout": 0.2,
    },
    "LanguageDecoder": {
        "img_feature_shape": (196, 64),
        "embed_dim": 32,
        "num_layers": 2,
        "num_heads": 8,
        "ffn_dim": 32 * 4,
        "dropout": 0.1,
        "ffn_dropout": 0.2,
    },
    "DataLoaderTrain": {
        "batch_size": 16,
        "device": get_device().type,
        "dataset_dir": DATASET_DIR,
        "add_augmentation": True,
    },
    "DataLoaderVal": {
        "batch_size": 16,
        "device": get_device().type,
        "dataset_dir": DATASET_DIR,
        "add_augmentation": False,
    },
    "TrainerMAE": {
        "lr_start": 1e-4,
        "lr_end": 1e-6,
        "weight_decay": 0.05,
        "train_num_steps": 80,
        "grad_clip": 1.0,
        "sample_every": 10,
        "save_every": 40,
        "results_folder": f"{CURRENT_DIR}/debug/pretrain",
        "use_amp": True,
        "use_latest_checkpoint": True,
        "mask_ratio": 0.75,
    },
    "TrainerCaptioning": {
        "lr_start": 1.5e-4,
        "lr_end": 1e-6,
        "wd_encoder": 0.05,
        "wd_decoder": 0.01,
        "train_num_steps": 25,
        "warm_up_pct": 0.1,
        "frozen_enc_pct": 0.3,
        "grad_clip": 1.0,
        "sample_every": 5,
        "save_every": 10,
        "eval_every": 5,
        "results_folder": f"{CURRENT_DIR}/debug/captioning",
        "use_amp": True,
        "use_latest_checkpoint": True,
        "eps": 0.10,
        "max_len": 5,
    },
}

#### Config for final prod training
PROD_CONFIG = {
    "dataset_dir": DATASET_DIR,
    "ImageEncoder": {
        "img_size": 224,
        "patch_size": 16,
        "in_channels": 3,
        "embed_dim": 768,
        "num_layers": 8,
        "num_heads": 12,
        "ffn_dim": 768 * 4,
        "dropout": 0.1,
        "ffn_dropout": 0.2,
    },
    "MAEdecoder": {
        "img_size": 224,
        "patch_size": 16,
        "in_channels": 3,
        "embed_dim": 768,
        "num_layers": 4,
        "num_heads": 8,
        "ffn_dim": 768 * 4,
        "dropout": 0.1,
        "ffn_dropout": 0.2,
    },
    "LanguageDecoder": {
        "img_feature_shape": (196, 768),
        "embed_dim": 512,
        "num_layers": 4,
        "num_heads": 8,
        "ffn_dim": 512 * 4,
        "dropout": 0.1,
        "ffn_dropout": 0.2,
    },
    "DataLoaderTrain": {
        "batch_size": 256,
        "device": get_device().type,
        "dataset_dir": DATASET_DIR,
        "add_augmentation": True,
    },
    "DataLoaderVal": {
        "batch_size": 256,
        "device": get_device().type,
        "dataset_dir": DATASET_DIR,
        "add_augmentation": False,
    },
    "TrainerMAE": {
        "lr_start": 1e-4,
        "lr_end": 1e-6,
        "weight_decay": 0.05,
        "train_num_steps": 80000,
        "grad_clip": 1.0,
        "sample_every": 1000,
        "save_every": 5000,
        "results_folder": f"{CURRENT_DIR}/results/pretrain",
        "use_amp": True,
        "use_latest_checkpoint": True,
        "mask_ratio": 0.75,
    },
    "TrainerCaptioning": {
        "lr_start": 1.5e-4,
        "lr_end": 1e-6,
        "wd_encoder": 0.05,
        "wd_decoder": 0.01,
        "train_num_steps": 25000,
        "warm_up_pct": 0.1,
        "frozen_enc_pct": 0.3,
        "grad_clip": 1.0,
        "sample_every": 500,
        "save_every": 1000,
        "eval_every": 1000,
        "results_folder": f"{CURRENT_DIR}/results/captioning",
        "use_amp": True,
        "use_latest_checkpoint": True,
        "eps": 0.10,
        "max_len": 50,
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Pipeline Module")
    parser.add_argument("--debug", help="Set to True to run in debug mode")
    args = parser.parse_args()
    debug = args.debug.lower() == "true"
    config = DEBUG_CONFIG if debug else PROD_CONFIG

    if debug and config.get("clear_dir", False):
        debug_results_dir = os.path.join(CURRENT_DIR, "debug")
        if os.path.exists(debug_results_dir):  # Check if the output results directory exists
            shutil.rmtree(debug_results_dir)  # Remove entire results directory

    pre_train_mae(config)  # Run model pre-training
    train_captioning_model(config)  # Run model training
