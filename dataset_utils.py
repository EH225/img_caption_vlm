import sys, os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

import sentencepiece as spm
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import random
import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List
from pycocotools.coco import COCO


############################
### Image Pre-Processing ###
############################

def resize_with_padding(img: Image, target_size: int = 224,
                        padding_pixels: Tuple[int] = (128, 128, 128)) -> Image:
    """
    Takes an input PIL.image and applies resizing & padding to it. Images are first re-sized along the
    largest dimension to target_size and then padding is added to the shorter dimension to get an overall
    square shape of [target_size x target_size]. padding_pixels determines the color of the pixels added
    for padding.

    :param img: An input PIL.image object to be processed.
    :param target_size: The desired size, height and width.
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
    This function will apply pre-processing to all images (ending in .jpg) found in original_img_dir and
    save the output to output_image_dir. Each image will be re-sized to be square with target_size pixels
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
    print("Processing complete! Images saved to: {output_image_dir}")


##############################
### Caption Pre-Processing ###
##############################

def create_vocab(train_captions_path: str, train_imgs_path: str, output_dir: str,
                 vocab_size: int = 10000) -> None:
    """
    Creates a vocab.model and vocab.vocab file by training a sub-piece sentence tokenizer on the corpus of
    training captions with a given vocab size limit.

    :param train_captions_path: The file path of where the training captions set is located.
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
    all_captions_train_path = os.path.join(output_dir, "all_captions_train.csv")
    all_captions.to_csv(all_captions_train_path)

    # Create a vocab using the training captions as a word bank, use the sub-word tokenizer
    # This sets <pad> as the padding token and assigns it to a token_id of 0, it also adds the <unk>
    # token at index 1, <s> at index 2 and </s> at index 3
    spm.SentencePieceTrainer.train(input=all_captions_train_path,
                                   model_prefix=os.path.join(output_dir, "vocab"),
                                   vocab_size=vocab_size, pad_id=0, pad_piece="<pad>", unk_id=1,
                                   bos_id=2, eos_id=3)
    os.remove(all_captions_train_path)  # Delete this csv file when finished


def tokenize_captions(captions_path: str, vocab_model_path: str, output_path: str,
                      max_tokens: int = 100) -> None:
    """
    This function reads in all the captions saved to a particular captions_path and tokenizes them using a
    saved sentence piece tokenizer trained and saved to disk. This method saves the results as a .pt file
    in output_path. max_tokens determines the max number of tokens allowed per caption.

    :param captions_path: A file path pointing to the captions .json file to process.
    :param vocab_model_path: A file path pointing to a save sp model.
    :param output_path: A directory to save the tokenized captions to (a list of lists of ints).
    :param max_tokens: The max number of tokens allowed per caption.
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

    # 3). Tokenize all the captions and aggregate them into a np.array for each image in the dataset
    output_dict = {}
    for d in caption_dicts:  # Tokenize each of the captions into integers, truncate, and pad if needed
        tokens = [[sp.bos_id()] + sp.encode(x["caption"], out_type=int)[:(max_tokens - 2)] + [sp.eos_id()]
                  for x in d]  # Add <s> and </s> tokens to either size, cap at max_tokens in total
        # tokens = [x + [sp.pad_id()] * ((max_tokens - 1) - len(x)) for x in tokens] # Pad the rest
        # Record image_id: List[List[int]] to associate the caption token sequences with each image
        output_dict[d[0]["image_id"]] = tokens

    # 4). Save the results to disk to complete the pre-processing
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Make this directory if it doesn't yet exist
    torch.save(output_dict, output_path)
    # np.savez_compressed(output_path, **{str(k): v for k, v in output_dict.items()})


####################
### Data Loaders ###
####################

def collate_fn(batch: List[Dict], pad_token_id: int) -> Dict:
    """
    This function is used aggregate multiple entries from the dataset into a batch by collating the caption
    token list of lists into a padded torch.Tensor so that they're all the same length, i.e. the longest
    caption in the batch.

    :param batch: A batch as a list of dictionaries.
    """
    images = torch.stack([b["image"] for b in batch])
    captions = [b["caption"] for b in batch]
    captions = pad_sequence(captions, batch_first=True, padding_value=pad_token_id)
    return {"images": images, "captions": captions}


class CocoCaptionDataset(Dataset):
    """
    Dataset object for COOC image captioning data.
    """

    def __init__(self, image_dir: str, caption_path: str, transform: transforms.Compose):
        self.image_dir = image_dir  # The directory containing all of the images
        self.captions = torch.load(caption_path, map_location="cpu", weights_only=False)
        self.image_ids = list(self.captions.keys())  # The unique int image ids of all images
        self.transform = transform  # Used to transform the images before they are returned in the batch

    def __len__(self):
        """
        Returns the total number of images in the dataset.
        """
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns a dictionary containing keys "image" and "caption" for a particular index in the dataset.
        """
        image_id = self.image_ids[idx]  # Get the image id associated with this int index
        image_name = "0" * (12 - len(str(image_id))) + str(image_id)  # Convert to a str and pad with 0s

        # Load the image from disk, images names have zero left padding e.g. 0000048
        img_path = os.path.join(self.image_dir, f"{image_name}.jpg")
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)  # Convert to Tensor and normalize with the saved transforms
        caption = random.choice(self.captions[image_id])  # Sample a random caption associated with this image

        # Return the data as a dict, torch.FloatTensor [3,224,224], torch.IntTensor(len(caption))
        return {"image": image, "caption": torch.tensor(caption, dtype=torch.long)}


def get_dataloader(split: str = "train", batch_size: int = 128, device: str = "cuda") -> DataLoader:
    """
    This method returns a DataLoader object by loading in the pre-processed dataset from disk.

    :param split: The data split e.g. "train" or "val".
    :param batch_size: The batch size for the data loader.
    :param device: A string denoting the device.
    :returns: A DataLoader object with the dataset split specified loaded.
    """
    image_dir = os.path.join(CURRENT_DIR, f"dataset/preprocessed/images/{split}2017")
    caption_path = os.path.join(CURRENT_DIR, f"dataset/preprocessed/captions/{split}_captions.pt")
    image_transforms = transforms.Compose([
        transforms.ToTensor(),  # PIL → FloatTensor [0,1]
        transforms.Normalize(  # From ImageNet sample statistics, standard normalizations
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load the vocab model to get the padding token id, should be 0 but it is safer to check
    vocab_model_path = os.path.join(CURRENT_DIR, "dataset/preprocessed/vocab.model")
    sp = spm.SentencePieceProcessor()
    sp.load(vocab_model_path)
    dataset = CocoCaptionDataset(image_dir, caption_path, image_transforms)
    # if device == "cuda":
    #     num_workers, pin_memory, persistent_workers = 4, True, True
    # else:
    #     num_workers, pin_memory, persistent_workers = 0, False, False
    num_workers, pin_memory, persistent_workers = 0, False, False
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                      pin_memory=pin_memory, persistent_workers=persistent_workers,
                      collate_fn=lambda b: collate_fn(b, pad_token_id=sp.pad_id()))


if __name__ == "__main__":
    # The steps below run a full data-set preprocessing pipeline of steps
    img_size = 224  # Process all the images to be a set size with H==W
    vocab_size = 10000  # How many sub-word tokens we will create in our vocab
    max_tokens_per_caption = 100  # Limit how many total tokens we can have for each caption, this includes
    # the special start and end tokens appended to the front and back
    dataset_dir = os.path.join(CURRENT_DIR, "dataset/")

    # 1). Pre-process the training images, down-size and pad to [224 x 224 x 3]
    original_img_dir = os.path.join(dataset_dir, "coco_original/images/train2017")
    output_image_dir = os.path.join(dataset_dir, "preprocessed/images/train2017")
    preprocess_images(original_img_dir, output_image_dir, img_size)

    # 2). Pre-process the validation images, down-size and pad to [224 x 224 x 3]
    original_img_dir = os.path.join(dataset_dir, "coco_original/images/val2017")
    output_image_dir = os.path.join(dataset_dir, "preprocessed/images/val2017")
    preprocess_images(original_img_dir, output_image_dir, img_size)

    # 3). Using all the training set captions, create a sub-word tokenizer
    train_captions_path = os.path.join(dataset_dir, "coco_original/annotations/captions_train2017.json")
    train_imgs_path = os.path.join(dataset_dir, "coco_original/images/train2017/")
    output_dir = os.path.join(dataset_dir, "preprocessed")
    create_vocab(train_captions_path, train_imgs_path, output_dir, vocab_size)

    # 4). Preprocess the train image captions and tokenize them into integers ahead of time
    captions_path = os.path.join(dataset_dir, "coco_original/annotations/captions_train2017.json")
    vocab_model_path = os.path.join(dataset_dir, "preprocessed/vocab.model")
    output_path = os.path.join(dataset_dir, "preprocessed/captions/train_captions.pt")
    tokenize_captions(captions_path, vocab_model_path, output_path, max_tokens=max_tokens_per_caption)

    # 5). Preprocess the validation image captions and tokenize them into integers ahead of time
    captions_path = os.path.join(dataset_dir, "coco_original/annotations/captions_val2017.json")
    vocab_model_path = os.path.join(dataset_dir, "preprocessed/vocab.model")
    output_path = os.path.join(dataset_dir, "preprocessed/captions/val_captions.pt")
    tokenize_captions(captions_path, vocab_model_path, output_path, max_tokens=max_tokens_per_caption)
