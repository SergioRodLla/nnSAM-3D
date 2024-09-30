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

<br />

### Run training / fine-tuning:

#### Baseline (vanilla) nnUNet

To run training use the original `nnUNetv2_train` command:

`nnUNetv2_train <DATASET_NAME_OR_ID> <UNET_CONFIGURATION> <FOLD>`

Example training baseline model using first fold of cross-validation:

`nnUNetv2_train 500 3d_fullres 0`

The dataset ID is arbitatry and can be set when preprocessing it. Alternatively the dataset name can also be used.
Change the fold to train other CV folds (0,1,2,3,4).
There are other useful arugments in `nnUNetv2_train`, like resumuning an unfinished training (`--c`). 

<br />

#### nnSAM-3D

**IMPORTANT**: To use nnSAM-3D model require setting enviromental variable:

`export MODEL_NAME=nnsam_3d`

This will specify the model architecture when calling `get_network_from_plans_custom`, inside the custom trainer class when loading the model. Also the SAM-Med3D model weights are assumed to be at "nnUNet_raw" path (environmental variable), alternatively change the path in `unet.py` line 235.

Train nnSAM-3D change the nnUNet trainer using `-tr`:

`nnUNetv2_train 500 3d_fullres 0 -tr nnSAM3D_Trainer`

<br />

#### STU-Net models

To fine-tune STU-Net models, run `run_finetuning_stunet.py`:

`run_finetuning_stunet.py <DATASET_NAME_OR_ID> <UNET_CONFIGURATION> <FOLD> -tr <STUNET_MODEL_TYPE> -pretrained_weights <PATH_TO_STUNET_WEIGHTS>`

For example to fine-tune STU-Net-H (huge):

`run_finetuning_stunet.py 500 3d_fullres 0 -tr STUNetTrainer_huge_ft -pretrained_weights stunet_weights/huge_ep4k.model`
