# Vision-Language Transformer Model for Image Caption Generation Project
This repository contains code for training a Vision-Language Transformer model for the task of producing high-quality, descriptive image captions using the [COCO](https://cocodataset.org/#home) dataset. Below is a brief outline of the contents of this repo:
- `utils.py`: This module contains low-level helper functions used throughout the project.
- `dataset_utils.py`: This module contains utility functions for pre-processing the COCO dataset and for constructing dataloaders used in training and validation.
- `torch_models.py`: This module contains all the components used to construct the Vision-Language Model (VLM) in pytorch.
- `vlm_trainer.py`: This module contains a classes that are used to run MAE pre-training, CLIP-style pre-training, teacher-forcing supervised training, and Self-Critical Sequence Training (SCST) fine tuning.
- `run.py`: This module is used for putting together all the components mentioned above to run model training.
- `env_req`: This folder contains `environment.yml` which outlines the virtural environment configuration used to develop and run this project.

This project leveraged materials from Stanford University's Deep Learning for Computer Vision ([XCS231N](https://cs231n.stanford.edu/)) course, with many modifications.
