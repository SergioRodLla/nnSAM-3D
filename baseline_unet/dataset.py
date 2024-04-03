import os
# from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib


class HecktorDataset3D(Dataset):
    def __init__(self, image_dir, mask_dir, patch_size, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.patch_size = patch_size # maybe (16, 16, 16)?
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    # return a list of patches and masks instead of an image
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace("WF","gtv"))
        image = np.array(nib.load(img_path).get_fdata(), dtype=np.float32)
        mask = np.array(nib.load(mask_path).get_fdata(), dtype=np.float32)
        mask[mask == 2.0] = 1.0  # Treat it like a binary segmentation

        patches_image = []
        patches_mask = []

        z, y, x = image.shape
        patch_z, patch_y, patch_x = self.patch_size

        # Divide each dimension into two equal parts and extract a patch from each part to get 8 equal 3D patches
        half_z = z // 2
        half_y = y // 2
        half_x = x // 2

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    start_z = i * half_z
                    start_y = j * half_y
                    start_x = k * half_x

                    patch_image = image[start_z:start_z + patch_z,
                                        start_y:start_y + patch_y,
                                        start_x:start_x + patch_x]
                    patch_mask = mask[start_z:start_z + patch_z,
                                      start_y:start_y + patch_y,
                                      start_x:start_x + patch_x]

                    patches_image.append(patch_image)
                    patches_mask.append(patch_mask)

        return patches_image, patches_mask


# For training using single images (2D slices)
class HecktorDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace("WF", "gtv"))
        image = np.array(nib.load(img_path).get_fdata(dtype=np.float32))
        mask = np.array(nib.load(mask_path).get_fdata(dtype=np.float32))
        # print("Img =", image.shape)
        # image = np.array(Image.open(img_path), dtype=np.float32)
        # mask = np.array(Image.open(mask_path), dtype=np.float32)
        mask[mask == 2.0] = 1.0  # Treat it like a binary segmentation
        # print("Mask =", mask.shape)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
