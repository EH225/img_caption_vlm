"""
This module contains functions for pre-processing the images and captions of the dataset and helpful functions
for constructing and loading a dataloader object to access the pre-processed data.
"""
import sys, os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

import sentencepiece as spm
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import random, json
import torch
from collections import defaultdict
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List
from pycocotools.coco import COCO
from utils import get_device


############################
### Image Pre-Processing ###
############################

def resize_with_padding(img: Image, target_size: int = 224,
                        padding_pixels: Tuple[int] = (128, 128, 128)) -> Image:
    """
    Takes an input PIL.image and applies resizing & padding to it. Images are first re-sized along the
    largest dimension to target_size and then padding is added to the shorter size to get an overall
    square shape of [target_size x target_size x 3]. padding_pixels determines the color of the pixels added
    for padding, the default is a light gray (128, 128, 128).

    :param img: An input PIL.image object to be processed.
    :param target_size: The desired size, height and width, images are resized to be square.
    :param padding_pixels: The color of the padding pixels to be added.
    :returns: A processed image that has been resized and padded to be square.
    """
    w, h = img.size  # Get the current height and width
    scale = target_size / max(w, h)  # Compute how much to scale the largest dim to get target_size
    new_w, new_h = int(w * scale), int(h * scale)  # Scale to reach the desired height and width
    img = img.resize((new_w, new_h), Image.BICUBIC)  # Resize the image

    pad_w = target_size - new_w  # Compute how much padding to add horizontally
    pad_h = target_size - new_h  # Compute how much padding to add vertically

    # Compute the amount of padding needed along each edge
    padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)

    img = F.pad(img, padding, fill=padding_pixels)  # Add padding to make the image square
    return img


def preprocess_images(original_img_dir: str, output_image_dir: str, target_size: int = 224) -> None:
    """
    This function applies pre-processing to all images (ending in .jpg) found in original_img_dir and
    saves the output to output_image_dir. Each image will be re-sized to be square with target_size pixels
    along each side with padding added to fill in the rest.

    :param original_img_dir: A folder path to a directory containing images to be processed.
    :param output_image_dir: A folder path to a directory where the processed images are to be saved.
    :param target_size: The desired size to make all images.
    :returns: None. Images are read from and written to disk.
    """
    os.makedirs(output_image_dir, exist_ok=True)  # Make this directory if it doesn't yet exist

    # Get all image filenames in the original_img_dir directory
    image_files = [f for f in os.listdir(original_img_dir) if f.endswith(".jpg")]

    print(f"Processing images in: {original_img_dir}")
    for image_file in tqdm(image_files, ncols=75):
        image_path = os.path.join(original_img_dir, image_file)
        image = Image.open(image_path).convert("RGB")  # Open in RGB mode
        image = resize_with_padding(image)  # Resize and add padding
        # Save the processed image to the output directory
        image.save(os.path.join(output_image_dir, image_file))
    print(f"Processing complete! Images saved to: {output_image_dir}")


##############################
### Caption Pre-Processing ###
##############################


def clean_caption(caption: str) -> str:
    """
    Performs data cleaning on an input caption string provided and returns the cleaned caption string.

    Makes all letters lower case, removes excess whitespace at the start and end, and adds an ending period
    if no end punctuation is present. Removes other special characters as well that are non-standard.

    :param caption: A string caption for an image.
    :return: A cleaned version of the caption string.
    """
    caption = caption.lower().strip().rstrip("!?.") # Lower case, remove excess white space, remove end punct
    if len(caption) > 0 and caption[-1] not in ".?!": # Add a period to the end for consistent punctuation
        caption += "."
    return caption


def create_vocab(train_captions_path: str, output_dir: str, vocab_size: int = 10000) -> None:
    """
    Creates a vocab.model and vocab.vocab file by training a sub-piece sentence tokenizer on the corpus of
    training captions with a given vocab size limit.

    :param train_captions_path: The file path of where the training set captions are located.
    :param output_dir: A directory to save the trained vocab model out to.
    :param vocab_size: The desired number of unique sub-word tokens in the vocab.
    :returns: None. Saves the trained model to disk.
    """
    os.makedirs(output_dir, exist_ok=True)  # Make this directory if it doesn't yet exist

    coco_train = COCO(train_captions_path)
    image_ids_train = coco_train.getImgIds()
    # Create a list of dictionaries, one for each training image in the dataset
    captions_train = [coco_train.loadAnns(coco_train.getAnnIds(imgIds=img_id)) for img_id in image_ids_train]
    all_captions = []  # Collect all training set captions
    for item in tqdm(captions_train, ncols=75):  # Extract out the captions for each image
        for d in item:  # Extract all the captions, each image comes with multiple captions
            all_captions.append(d["caption"])

    all_captions = pd.DataFrame(all_captions)  # Convert to a dataframe and write to CSV
    all_captions[0] = all_captions[0].apply(clean_caption)  # Clean the captions
    all_captions_train_path = os.path.join(output_dir, "all_captions_train.csv")
    all_captions.to_csv(all_captions_train_path)

    # Create a vocab using the training captions as a word bank, use the sub-word tokenizer
    # This sets <pad> as the padding token and assigns it to a token_id of 0, it also adds the <unk>
    # token at index 1, <s> at index 2 and </s> at index 3
    spm.SentencePieceTrainer.train(input=all_captions_train_path,
                                   model_prefix=os.path.join(output_dir, "vocab"),
                                   vocab_size=vocab_size, model_type="unigram", character_coverage=0.9995,
                                   pad_id=0, unk_id=1, bos_id=2, eos_id=3,
                                   pad_piece="<PAD>", unk_piece="<UNK>", bos_piece="<s>", eos_piece="</s>",
                                   num_threads=8)
    os.remove(all_captions_train_path)  # Delete this csv file when finished


def tokenize_captions(captions_path: str, vocab_model_path: str, output_path: str,
                      max_tokens: int = 50) -> None:
    """
    This function reads in all the captions saved to a particular captions_path and tokenizes them using a
    saved sentence piece tokenizer trained and saved to disk. This method saves the results as a .pt file
    in output_path. max_tokens determines the max number of tokens allowed per caption, which includes the
    special start <s> and end <s/> tokens.

    :param captions_path: A file path pointing to the captions .json file to process.
    :param vocab_model_path: A file path pointing to a saved sp model.
    :param output_path: A directory to save the tokenized captions to (a list of lists of ints).
    :param max_tokens: The max number of tokens allowed per caption including the start and end tokens.
    :returns: None. Saves the results to disk.
    """
    # 1). Read in the caption data to be tokenized
    coco_data = COCO(captions_path)
    image_ids = coco_data.getImgIds()
    # Create a list of dictionaries, one for each training image in the dataset
    caption_dicts = [coco_data.loadAnns(coco_data.getAnnIds(imgIds=img_id)) for img_id in image_ids]

    # 2). Load in the vocab.model cached to disk and use it to tokenize the captions
    sp = spm.SentencePieceProcessor()
    sp.load(vocab_model_path)

    # 3). Tokenize all the captions and aggregate them as a list of lists of ints for each img in the dataset
    output_dict = {}
    bos, eos = sp.bos_id(), sp.eos_id()
    for d in caption_dicts:  # Tokenize each of the captions into integers, truncate, and pad if needed
        tokens = [
            [bos] + sp.encode(clean_caption(x["caption"]), out_type=int)[:(max_tokens - 2)] + [eos]
            for x in d]  # Add <s> and </s> tokens to either size, cap at max_tokens in total
        # Record image_id: List[List[int]] to associate the caption token sequences with each image
        output_dict[d[0]["image_id"]] = tokens

    # 4). Save the results to disk to complete the captions pre-processing
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Make this directory if it doesn't yet exist
    torch.save(output_dict, output_path)

def save_gts_dict(captions_path: str, output_path: str) -> None:
    """
    Extracts the captions from a given captions file and saves them to disk after cleaning them for CIDEr
    evaluation. Saves a dictionary of captions in the format {image_id: [caption_str1, caption_str2, ...]}
    """
    # Read in the caption data to be tokenized
    coco_data = COCO(captions_path)
    image_ids = coco_data.getImgIds()
    # Create a list of dictionaries, one for each training image in the dataset
    caption_dicts = [coco_data.loadAnns(coco_data.getAnnIds(imgIds=img_id)) for img_id in image_ids]

    # Create an output dictionary in json format
    output = defaultdict(list)
    for d in caption_dicts:
        for x in d:
            output[x["image_id"]].append(clean_caption(x["caption"]))
    json.dump(output, open(output_path, "w", encoding="utf-8"), indent=4)


####################
### Data Loaders ###
####################

def collate_fn(batch: List[Dict], pad_token_id: int) -> Dict:
    """
    This function is used to aggregate multiple entries from the dataset into a batch by collating the caption
    token list of lists into a padded torch.Tensor so that they're all the same length, i.e. the longest
    caption in the batch.

    :param batch: A batch as a list of dictionaries.
    :param pad_token_id: An integer denoting the index of the padding token.
    :returns: A dictionary with keys "images": torch.Tensor (N, C, H, W) and "captions": torch.Tensor (N, T).
    """
    images = torch.stack([b["image"] for b in batch])  # Concatenate into 1 large tensor of size (N, C, H, W)
    captions = [b["caption"] for b in batch]  # Collect caption tensors into a list
    # Combine them into 1 large tensor of size (N, T) where T = longest caption token seq length, add padding
    # to the others where needed to make them all of length T
    captions = pad_sequence(captions, batch_first=True, padding_value=pad_token_id)
    image_names = [b["image_name"] for b in batch]  # Create a list of image names e.g. ["000000001234", ...]
    return {"images": images, "captions": captions, "image_names": image_names}


class CocoCaptionDataset(Dataset):
    """
    Dataset object for COCO image captioning data.
    """

    def __init__(self, dataset_dir: str, split: str, load_captions: bool, transform: transforms.Compose):
        """
        Initializes a dataset object for the COCO image-captioning dataset.

        The dataset structure is assumed to follow the pre-defined structure of the preprocessed directory
        denoted by dataset_dir.

        :param dataset_dir: A directory containing all the data (images, captions etc.)
        :param split: The image ids to include from image_dir which holds all images from all splits. This
            can be 1 string e.g. "val" or a space separated string e.g. "train test" to include multiple
            splits. This determines which image _ids are part of this data-loader.
        :param load_captions: If set to True, then the captions data associated with the split param are
            also loaded. If False, then no captions data is loaded and a tensor of all zeros of size (N, 1)
            is returned in each batch.
        :param transform: A composition of torch vision transforms to apply to each image before being added
            to a batch.
        """
        self.image_dir = os.path.join(dataset_dir, "images") # The directory containing all the images
        self.image_ids = []
        for s in split.split(): # Separate on white space, read in all the image IDs for the splits defined
            self.image_ids.extend(pd.read_csv(os.path.join(dataset_dir, f"image_ids/{s}.csv"),
                                              header=None).iloc[:, 0].astype(int).tolist())

        self.caption_dir = os.path.join(dataset_dir, "captions")
        self.captions = None
        if load_captions:
            for s in split.split(): # Separate on white space, read in all captions for each split
                captions_path = os.path.join(self.caption_dir, f"{s}_captions.pt")
                captions = torch.load(captions_path, map_location="cpu", weights_only=False)
                if self.captions is None:
                    self.captions = captions
                else: # Combine together the captions loaded from each split
                    self.captions = self.captions_1 | captions
            assert self.captions.keys() == set(self.image_ids), "caption ids and image ids don't match"

        self.transform = transform  # Used to transform the images before they are returned in the batch

    def __len__(self):
        """
        Returns the total number of image-caption pairs in the dataset.
        """
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns a dictionary containing keys "image" and "caption" for a particular index in the dataset.
        """
        image_id = self.image_ids[idx]  # Get the image id associated with this int index
        image_name = "0" * (12 - len(str(image_id))) + str(image_id)  # Convert to a str and pad with 0s

        # Load the image from disk, images names have 0s left padding e.g. 000000001234, the total length
        # is 12 for all image names
        img_path = os.path.join(self.image_dir, f"{image_name}.jpg")
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)  # Convert to Tensor and normalize with the saved transforms
        if self.captions is None:  # If no captions in this data loader, still return a 0 for each
            caption = torch.zeros(1)
        else:  # Sample a random caption associated with this image and return it as a torch tensor
            caption = torch.tensor(random.choice(self.captions[image_id]), dtype=torch.long)

        # Return the data as a dict, torch.FloatTensor [3,224,224], torch.IntTensor(len(caption))
        return {"image": image, "caption": caption, "image_name": image_name}


def get_dataloader(split: str = "train", batch_size: int = 128, device: str = None,
                   include_captions: bool = True, add_augmentation: bool = False,
                   dataset_dir: str = None) -> DataLoader:
    """
    This method returns a DataLoader object by loading in the pre-processed dataset from disk.

    :param split: The data split e.g. "train", "val", or "test.
    :param batch_size: The batch size for the data loader.
    :param device: A string denoting the device.
    :param include_captions: If True, then the dataloader is constructed with captions.
    :param add_augmentation: If True, then the dataloader applies modest data augmentation operations to the
        batched images to increase image diversity and reduce overfitting.
    :param dataset_dir: A directory where the data set is saved. If not provided, it is assumed to be
        'dataset/preprocessed/' relative to this file's location.
    :returns: A DataLoader object with the dataset split specified loaded.
    """
    device = device if device is not None else get_device().type  # Auto-detect the available hardware
    assert isinstance(device, str), "device expected to be a string"
    dataset_dir = os.path.join(CURRENT_DIR, "dataset/preprocessed/") if dataset_dir is None else dataset_dir

    if add_augmentation is True:  # Add in data-augmentations
        image_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)],
                p=0.2),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ])

    else:  # No data-augmentation, just convert to tensor and normalize
        image_transforms = transforms.Compose([
            transforms.ToTensor(),  # PIL → FloatTensor [0,1]
            transforms.Normalize(  # From ImageNet sample statistics, standard normalizations
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    # Load the vocab model to get the padding token id, should be 0 but it is safer to check
    vocab_model_path = os.path.join(dataset_dir, "vocab.model")
    sp = spm.SentencePieceProcessor()
    sp.load(vocab_model_path)
    dataset = CocoCaptionDataset(dataset_dir, split, include_captions, image_transforms)
    print(f"{split} dataloader device: {device}")
    if device == "cuda":
        num_workers, pin_memory, persistent_workers, prefetch_factor = 4, True, True, 16
    else:
        num_workers, pin_memory, persistent_workers, prefetch_factor = 0, False, False, None
    print("num_workers, pin_memory, persistent_workers, prefetch_factor, batch_size, len(dataset):",
          num_workers, pin_memory, persistent_workers, prefetch_factor, batch_size, len(dataset))
    # num_workers, pin_memory, persistent_workers = 0, False, False
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                      pin_memory=pin_memory, persistent_workers=persistent_workers,
                      prefetch_factor=prefetch_factor,
                      collate_fn=lambda b: collate_fn(b, pad_token_id=sp.pad_id()))


if __name__ == "__main__":
    # 0).  The steps below run a full data-set preprocessing pipeline of steps
    img_size = 224  # Process all the images to be a set size with H==W
    vocab_size = 6000  # How many sub-word tokens we will create in our vocab
    max_tokens_per_caption = 50  # Limit how many total tokens we can have for each caption, this includes
    # the special start and end tokens appended to the front and back
    dataset_dir = os.path.join(CURRENT_DIR, "dataset/")

    # 1). Pre-process the training, validation, and test images, down-size and pad to [224 x 224 x 3]
    for split in ["train", "val", "test"]:
        original_img_dir = os.path.join(dataset_dir, f"coco_original/images/{split}2017")
        output_image_dir = os.path.join(dataset_dir, "preprocessed/images/") # Save all into 1 folder
        preprocess_images(original_img_dir, output_image_dir, img_size)
        image_ids = [x.replace(".jpg", "") for x in os.listdir(original_img_dir) if x.endswith(".jpg")]
        pd.DataFrame(image_ids).to_csv(os.path.join(dataset_dir, f"preprocessed/image_ids/{split}.csv"),
                                                    index=False, header=None)

    # 2). Using all the training set captions, create a sub-word tokenizer
    train_captions_path = os.path.join(dataset_dir, "coco_original/annotations/captions_train2017.json")
    output_dir = os.path.join(dataset_dir, "preprocessed")
    create_vocab(train_captions_path, output_dir, vocab_size)

    # 3). Preprocess the train and validation image captions and tokenize them into integers ahead of time
    # There are no captions provided for the test set image
    for split in ["train", "val"]:
        captions_path = os.path.join(dataset_dir, f"coco_original/annotations/captions_{split}2017.json")
        vocab_model_path = os.path.join(dataset_dir, "preprocessed/vocab.model")
        output_path = os.path.join(dataset_dir, f"preprocessed/captions/{split}_captions.pt")
        tokenize_captions(captions_path, vocab_model_path, output_path, max_tokens=max_tokens_per_caption)
        save_gts_dict(captions_path, os.path.join(dataset_dir, f"preprocessed/gts_{split}.json"))
