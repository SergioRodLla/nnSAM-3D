# Leveraging General-purpose models for improved H&N tumor segmentation

This repository contains the code used in [...]

Code was designed to be integrated within the [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) framework (v2.4.1). The folder *nnUNet_changes* contains the files used to perform the experiments implemented in this work.

- `get_network_from_plans_custom.py`: Contains the function to build the specified network architecture.

- `nnUNetTrainer_custom_models.py`: Contains new trainer classes built on top of the nnU-Net default trainer, defining the specified model.

- `STUNetTrainer.py`: Contains the code to build the STU-Net-B and STU-Net-H network architectures.

- `run_finetuning_stunet.py`: Contains training adjustments to fine-tune STU-Net-B and STU-Net-H models.

- `unet.py` and `unet_decoder.py`: Contain the code to build the nnSAM-3D network architecture.
  - `unet.py` defines both network encoder blocks (nnU-Net and 3D SAM ViT) and includes the custom forwarding of CT and PET to leverage their embeddings as described in the manuscript.
  - `unet_decoder.py` defines the custom decoder structure, adjusted to handle the dimensions of the concatenated embeddings.


### Run training / fine-tuning:

To run training use the original `nnUNetv2_train` command specifying the desired trainer class indicated with the `-tr` argument. For example:

`nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD -tr nnSAM3D_Trainer`

IMPORTANT: To use nnSAM-3D or nnSAM (2D) models require setting enviromental variable like:

`export MODEL_NAME=nnsam_3d`

`export MODEL_NAME=nnsam_2d`


To fine-tune STU-Net models, run. For example to fine-tune STU-Net-H (huge):

`run_finetuning_stunet.py DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD -tr STUNetTrainer_huge_ft -pretrained_weights path_to_stunet_weights/huge_ep4k.model`


