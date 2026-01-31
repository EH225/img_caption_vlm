"""
This script is used to train the Vision-Language Model for caption generation.
"""
import sys, os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, CURRENT_DIR)

import sentencepiece as spm
from torch_models import ImageEncoder, MAEdecoder, LanguageDecoder, VisionLanguageModel
from torch_models import CLIPimgEncoder, CLIPtextEncoder
from dataset_utils import get_dataloader
from vlm_trainer import TrainerMAE, TrainerCLIP, TrainerCaptioning
from utils import read_config, get_device
from typing import Dict
import argparse, shutil, json


def pre_train_mae(config: dict) -> None:
    """
    Runs MAE pre-training on the vision-transformer encoder model using the ImageEncoder, MAEdecoder, and
    TrainerMAE classes with the configurations specified in the config file. This pre-training loop trains
    a vision-transformer (ViT) model from scratch (ImageEncoder) on an image patch reconstrucion loss
    objective, which teaches the image encoder to learn low-level feature structure.

    :param config: A config dictionary containing parameters for various aspects of the training loop and
        model parameters for how to configure the model parameters.
    :returns: None. Results are saved to disk.
    """
    # 1).Init the encoder and decoder model for training
    encoder = ImageEncoder(**config.get("ImageEncoder", {}))
    decoder = MAEdecoder(**config.get("MAEdecoder", {}))

    # 2). Construct the COCO training dataset loader and validation dataset loader
    dataloader_train = get_dataloader(split='train test', include_captions=False,
                                      **config.get("DataLoaderTrain", {}))
    dataloader_val = get_dataloader(split='val', include_captions=False,
                                    **config.get("DataLoaderVal", {}))

    # 3). Configure the training pipeline with the TrainerMAE object
    trainer = TrainerMAE(encoder, decoder, dataloader_train, dataloader_val, **config.get("TrainerMAE", {}))

    # 4). Train the model to completion
    trainer.train(mask_ratio=config["TrainerMAE"].get("mask_ratio", 0.75))


def pre_train_clip(config: dict) -> None:
    """
    Runs CLIP style pre-training on the vision-transformer encoder model using the ImageEncoder,
    CLIPtextEncoder, and TrainerCLIP classes with the configurations specified in the config file. This step
    of pre-training generally follows MAE pre-training and trains a vision-transformer (ViT) (ImageEncoder)
    to associate images with textual descriptions, which teaches the image encoder to learn semantic image
    understanding by maximizing the cosine similarity of images and their captions in a shared latent space
    while minimzing the cosine similarity of off-diagonal (image, caption) pairs.

    This training loop uses the pre-trained CLIP text encoder for stability and transfer learning purposes.

    :param config: A config dictionary containing parameters for various aspects of the training loop and
        model parameters for how to configure the model parameters.
    :returns: None. Results are saved to disk.
    """
    # 1). Read in the sub-word sentence piece vocab model derived from the training captions
    sp_model = spm.SentencePieceProcessor()
    sp_model.load(os.path.join(config["dataset_dir"], "vocab.model"))

    # 2).Init the image encoder and text encoder model for training
    img_encoder = ImageEncoder(**config.get("ImageEncoder", {}))  # Init the trainable image encoder
    text_encoder = CLIPtextEncoder(get_device())  # Load the pre-trained, frozen CLIP text encoder.

    # 3). Construct the COCO training dataset loader and validation dataset loader
    dataloader_train = get_dataloader(split='train', include_captions=True,
                                      **config.get("DataLoaderTrain", {}))
    dataloader_val = get_dataloader(split='val', include_captions=True,
                                    **config.get("DataLoaderVal", {}))

    # 4). Configure the training pipeline with the TrainerCLIP object
    trainer = TrainerCLIP(img_encoder, text_encoder, sp_model, dataloader_train, dataloader_val,
                          **config.get("TrainerCLIP", {}))

    # 5). Train the model to completion on the CLIP pre-training task
    trainer.train()


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
    if config.get("use_clip_encoder", True):
        print("Using the CLIP encoder")
        encoder = CLIPimgEncoder(device=get_device())
    else:
        encoder = ImageEncoder(**config.get("ImageEncoder", {}))
    vlm = VisionLanguageModel(encoder=encoder,
                              decoder=LanguageDecoder(sp_model=sp_model, **config.get("LanguageDecoder", {})))

    # 3). Construct the COCO training dataset loader and validation dataset loader
    dataloader_train = get_dataloader(split='train', include_captions=True,
                                      **config.get("DataLoaderTrain", {}))
    dataloader_val = get_dataloader(split='val', include_captions=True,
                                    **config.get("DataLoaderVal", {}))
    with open(os.path.join(config["dataset_dir"], 'gts_val.json'), 'r') as f:
        gts_val = json.load(f)  # Load in the dictionary of ground-truth val set captions

    # 4). Configure the training pipeline with the trainer object
    trainer = TrainerCaptioning(vlm, dataloader_train, dataloader_val, gts_val=gts_val,
                                **config.get("TrainerCaptioning", {}))

    # 5). Train the model to completion
    trainer.train(eps=config["TrainerCaptioning"].get("eps", 0.1),
                  max_len=config["TrainerCaptioning"].get("max_len", 50))


def train_scst(config: Dict) -> None:
    """
    Runs Self-Critical Sequence Training (SCST) to improve CIDEr scores of a pre-trained model.

    :param config: A config dictionary containing parameters for various aspects of the training loop and
        model parameters for how to configure the model parameters.
    :returns: None. Results are saved to disk.
    """
    # 1). Read in the sub-word sentence piece vocab model derived from the training captions
    sp_model = spm.SentencePieceProcessor()
    sp_model.load(os.path.join(config["dataset_dir"], "vocab.model"))

    # 2).Init the encoder and decoder model for training and use them to create a vision-language model
    if config.get("use_clip_encoder", True):
        print("Using the CLIP encoder")
        encoder = CLIPimgEncoder(device=get_device())
    else:
        encoder = ImageEncoder(**config.get("ImageEncoder", {}))
    vlm = VisionLanguageModel(encoder=encoder,
                              decoder=LanguageDecoder(sp_model=sp_model, **config.get("LanguageDecoder", {})))

    # 3). Construct the COCO training dataset loader and validation dataset loader + read in caption dicts
    config["DataLoaderTrain"]["batch_size"] = min(config["DataLoaderTrain"]["batch_size"], 128)
    config["DataLoaderVal"]["batch_size"] = min(config["DataLoaderVal"]["batch_size"], 128)
    dataloader_train = get_dataloader(split='train', include_captions=True,
                                      **config.get("DataLoaderTrain", {}))
    dataloader_val = get_dataloader(split='val', include_captions=True,
                                    **config.get("DataLoaderVal", {}))
    with open(os.path.join(config["dataset_dir"], 'gts_val.json'), 'r') as f:
        gts_val = json.load(f)  # Load in the dictionary of ground-truth val set captions
    with open(os.path.join(config["dataset_dir"], 'gts_train.json'), 'r') as f:
        gts_train = json.load(f)  # Load in the dictionary of ground-truth val set captions

    # 4). Configure the training pipeline with the trainer object
    trainer = TrainerCaptioning(vlm, dataloader_train, dataloader_val, gts_train=gts_train, gts_val=gts_val,
                                **config.get("TrainerSCST", {}))

    # 5). Train the model to completion
    trainer.train_scst(eps=config["TrainerSCST"].get("eps", 0.1),
                       max_len=config["TrainerSCST"].get("max_len", 50),
                       lambda_xe=config["TrainerSCST"].get("lambda_xe", 0.1))


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

    if not config.get("use_clip_encoder", False):  # Run pre-training on the image encoder if not using the
        # pre-trained, frozen CLIP image encoder
        pre_train_mae(config)  # Run MAE pre-training to train the image encoder
        pre_train_clip(config)  # Run CLIP-style image-language pre-training to train the image encoder
    train_captioning_model(config)  # Run supervised teacher-forcing training to train a caption decoder
    train_scst(config)  # Run SCST fine-tuning on the VLM to optimize CIDEr score
