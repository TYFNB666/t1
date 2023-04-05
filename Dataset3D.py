# import os
# import nibabel as nib
# import numpy as np
# import torch
# import torch.utils.data as data
#
#
# class CreateNiiDataset(data.Dataset):
#     def __init__(self, images_path, images_class):
#         self.images_path = images_path
#         self.images_class = images_class
#         """self.image_filenames = [os.path.join(image_dir, x) for x in os.listdir(image_dir)]
#         self.label_filenames = [os.path.join(label_dir, x) for x in os.listdir(label_dir)]"""
#
#     def __len__(self):
#         return len(self.images_path)
#
#     def __getitem__(self, index):
#         image = nib.load(self.images_path[index])
#         label = self.images_class[index]
#
#         image_arr = image.get_fdata().astype(np.double)
#
#         ImageTensor = torch.from_numpy(image_arr).unsqueeze(0)  # .unsqueeze(0) to add a channal
#
#         return ImageTensor.float(), label

import os
import nibabel as nib
import numpy as np
import torch
import torch.utils.data as data
import random
import scipy

class RandomRotation3D:
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, img):
        angle = random.uniform(-self.degrees, self.degrees)
        img = scipy.ndimage.rotate(img, angle, axes=(1, 2), reshape=False)
        return img

class CreateNiiDataset(data.Dataset):
    def __init__(self, images_path, images_class, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        image = nib.load(self.images_path[index])
        label = self.images_class[index]
        image_arr = image.get_fdata().astype(np.double)

        if self.transform:
            image_arr = self.transform(image_arr)

        return image_arr, label

    @staticmethod
    def collate_fn(batch):
        images, labels = zip(*batch)
        images = torch.stack(images, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)
        return images, labels
