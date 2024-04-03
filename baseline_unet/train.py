import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET3D
# from threeD_unet import BaselineUNet
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy_patches,
    save_predictions_as_patches,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2 # Test this
NUM_EPOCHS = 10
NUM_WORKERS = 2
IMAGE_HEIGHT = 176
IMAGE_WIDTH = 176
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "/content/gdrive/MyDrive/HECKTOR_dataset/train_images_small"
TRAIN_MASK_DIR = "/content/gdrive/MyDrive/HECKTOR_dataset/train_masks_small"
VAL_IMG_DIR = "/content/gdrive/MyDrive/HECKTOR_dataset/val_images_small"
VAL_MASK_DIR = "/content/gdrive/MyDrive/HECKTOR_dataset/val_masks_small"
NUM_CLASSES = 2 # 3 -> For multi-class (background, GTV-T, GTV-N)
PRED_DIR = "/content/gdrive/MyDrive/HECKTOR_dataset/saved_preds"
#PATCH_SIZE = [44, 44, 44] # Define small patches
PATCH_SIZE = [88, 88, 88] # Do 8 patches for image (176 / 2 = 88)

# def train_fn(loader, model, optimizer, loss_fn, scaler):
#     loop = tqdm(loader)

#     for batch_idx, (data, targets) in enumerate(loop):
#         data = data.to(device=DEVICE)
#         targets = targets.float().unsqueeze(1).to(device=DEVICE)

#         # forward
#         with torch.cuda.amp.autocast():
#             predictions = model(data)
#             loss = loss_fn(predictions, targets)

#         # backward
#         optimizer.zero_grad()
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()

#         # update tqdm loop
#         loop.set_postfix(loss=loss.item())

def train_patches_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = [patch.to(device=DEVICE) for patch in data]
        targets = [mask.float().unsqueeze(1).to(device=DEVICE) for mask in targets]

        optimizer.zero_grad()

        # Forward pass and compute loss for each patch
        total_loss = 0
        for patch, mask in zip(data, targets):
            # Use Mixed Precision Training to reduce memory usage (hopefully)
            with torch.cuda.amp.autocast():
                patch = patch.unsqueeze(1)
                predictions = model(patch)
                loss = loss_fn(predictions, mask)
                total_loss += loss

        # Backward pass
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update tqdm loop
        loop.set_postfix(loss=total_loss.item())
    
def main():
    train_transform = A.Compose(
        [
            #A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.5],
                std=[0.5],
                max_pixel_value=1.0
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            #A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.5],
                std=[0.5],
                max_pixel_value=1.0
            ),
            ToTensorV2(),
        ],
    )

    # in_channels changed to 1 since images are in grayscale
    if NUM_CLASSES == 2: # For binary segmentation
        model = UNET3D(in_channels=1, out_channels=1).to(DEVICE)
        loss_fn = nn.BCEWithLogitsLoss()
    # For binary segmentation use: nn.BCEWithLogitsLoss() The nn.CrossEntropyLoss() function in PyTorch combines
    # nn.LogSoftmax() and nn.NLLLoss() (negative log likelihood loss) in one single class.
    else: # For multi-class segmentation
        model = UNET3D(in_channels=1, out_channels=NUM_CLASSES).to(DEVICE)
        loss_fn = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        PATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)


    check_accuracy_patches(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_patches_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy_patches(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_patches(
            val_loader, model, folder=PRED_DIR, device=DEVICE
        )


if __name__ == "__main__":
    main()