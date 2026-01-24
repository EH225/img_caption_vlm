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
from utils import get_device, read_config
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Pipeline Module")
    parser.add_argument("--config", help="The name of the config file to use for training")
    args = parser.parse_args()
    debug = args.config.lower() == "debug"

    config = read_config(args.config, dataset_dir=f"{CURRENT_DIR}/dataset/preprocessed/")

    if debug and config.get("clear_dir", False):
        debug_results_dir = os.path.join(CURRENT_DIR, "debug")
        if os.path.exists(debug_results_dir):  # Check if the output results directory exists
            shutil.rmtree(debug_results_dir)  # Remove entire results directory

    pre_train_mae(config)  # Run model pre-training
    train_captioning_model(config)  # Run model training
