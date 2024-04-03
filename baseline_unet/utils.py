import torch
import torchvision
from dataset import HecktorDataset3D
from torch.utils.data import DataLoader
import os
import nibabel as nib
import numpy as np

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    check_dir = "/content/gdrive/MyDrive/HECKTOR_dataset/checkpoints"
    torch.save(state, os.path.join(check_dir, filename))

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    patch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = HecktorDataset3D(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        patch_size=patch_size,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = HecktorDataset3D(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        patch_size=patch_size,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy_patches(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for data, targets in loader:
            data = [patch.to(device) for patch in data]
            targets = [mask.float().unsqueeze(1).to(device) for mask in targets]

            # Forward pass and calculate accuracy metrics for each patch
            for patch, mask in zip(data, targets):
                patch = patch.unsqueeze(1)
                #print(f"Input shape: {patch.shape}")
                preds = torch.sigmoid(model(patch))
                preds = (preds > 0.5).float()
                num_correct += (preds == mask).sum()
                num_pixels += torch.numel(preds)
                dice_score += (2 * (preds * mask).sum()) / ((preds + mask).sum() + 1e-8)

    accuracy = num_correct / num_pixels * 100
    dice_score /= len(loader)

    print(f"Got {num_correct}/{num_pixels} with acc {accuracy:.2f}")
    print(f"Dice score: {dice_score:.4f}")
    model.train()

def save_predictions_as_patches(loader, model, folder="saved_images/", device="cuda"):
    model.eval()

    if not os.path.exists(folder):
        os.makedirs(folder)

    for idx, (data, targets) in enumerate(loader):
        data = [patch.to(device=device) for patch in data]
        targets = [mask.float().unsqueeze(1).to(device) for mask in targets]

        # Forward pass and save predictions for each patch
        for i, patch in enumerate(data, targets):
            with torch.no_grad():
                patch = patch.unsqueeze(1)
                preds = torch.sigmoid(model(patch))
                preds = (preds > 0.5).float()

            nifti_img = nib.Nifti1Image(patch, np.eye(4))
            nib.save(nifti_img, os.path.join(folder, f"pred_{idx}_patch_{i}.png"))

            mask_nifti = nib.Nifti1Image(targets[i], np.eye(4))
            nib.save(mask_nifti, os.path.join(folder, f"gt_{idx}_patch_{i}.png"))

            '''
            torchvision.utils.save_image(preds, os.path.join(folder, f"pred_{idx}_patch_{i}.png"))
            torchvision.utils.save_image(targets[i], os.path.join(folder, f"gt_{idx}_patch_{i}.png"))
            '''
    model.train()

# def check_accuracy(loader, model, device="cuda"):
#     num_correct = 0
#     num_pixels = 0
#     dice_score = 0
#     model.eval()

#     with torch.no_grad():
#         for x, y in loader:
#             x = x.to(device)
#             y = y.to(device)
#             #y = y.to(device).unsqueeze(1)
#             preds = torch.sigmoid(model(x))
#             preds = (preds > 0.5).float()
#             num_correct += (preds == y).sum()
#             num_pixels += torch.numel(preds)
#             dice_score += (2 * (preds * y).sum()) / (
#                 (preds + y).sum() + 1e-8
#             )

#     print(
#         f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
#     )
#     print(f"Dice score: {dice_score/len(loader)}")
#     model.train()

# def save_predictions_as_imgs(
#     loader, model, folder="saved_images/", device="cuda"
# ):
#     model.eval()
#     for idx, (x, y) in enumerate(loader):
#         x = x.to(device=device)
#         with torch.no_grad():
#             preds = torch.sigmoid(model(x))
#             preds = (preds > 0.5).float()
#         torchvision.utils.save_image(
#             preds, f"{folder}/pred_{idx}.png"
#         )
#         torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

#     model.train()




