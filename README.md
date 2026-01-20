# Vision-Language Transformer Model for Image Caption Generation Project
This repository contains code for the Vision-Language Transformer Model for Image Caption Generation Project which utilizes a transformers to process input images and output text captions describing what is pictured. The objective of the model is to produce high-quality, descriptive image captions using the [COCO](https://cocodataset.org/#home) dataset. Below is a brief outline of the contents of this repo:
- `utils.py`: This module contains low-level helper functions used throughout the project.
- `dataset_utils.py`: This module contains utility functions for pre-processing the COCO dataset and for constructing dataloaders.
- `torch_models.py`: This module contains all the components used to construct the Vision-Language Model (VLM) in pytorch.
- `vlm_trainer.py`: This module contains a class that is used to run a training loop for the VLM and periodically cache results.
- `run.py`: This module is used for putting together all the components mentioned above to run model training.
- `env_req`: This folder contains `environment.yml` which outlines the virtural environment configuration used to develop and run this project.

This project leveraged materials from Stanford University's Deep Learning for Computer Vision ([XCS231N](https://cs231n.stanford.edu/)) course, with many modifications.