# from PIL import Image
# import torch
# from torch.utils.data import Dataset
#
#
# class MyDataSet(Dataset):
#     """自定义数据集"""
#
#     def __init__(self, images_path: list, images_class: list, transform=None):
#         self.images_path = images_path
#         self.images_class = images_class
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.images_path)
#
#     def __getitem__(self, item):
#         img = Image.open(self.images_path[item])
#         # RGB为彩色图片，L为灰度图片
#         if img.mode != 'L':
#             raise ValueError("image: {} isn't gray mode.".format(self.images_path[item]))
#         label = self.images_class[item]
#
#         if self.transform is not None:
#             img = self.transform(img)
#
#         return img, label
#
#     @staticmethod
#     def collate_fn(batch):
#         # 官方实现的default_collate可以参考
#         # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
#         images, labels = tuple(zip(*batch))
#
#         images = torch.stack(images, dim=0)
#         labels = torch.as_tensor(labels)
#         return images, labels

# import os
# import re
# import numpy as np
# from PIL import Image
# from torch.utils.data import Dataset
# import torch
# from torchvision import transforms
#
#
# class MyDataSet(Dataset):
#     def __init__(self, images_path, images_class, transform=None):
#         self.images_path = images_path
#         self.images_class = images_class
#         self.transform = transform
#         self.grouped_images = self.group_images_by_patient()
#
#     def __len__(self):
#         return len(self.images_class)
#
#     def group_images_by_patient(self):
#         groups = {}
#         for img_path, img_class in zip(self.images_path, self.images_class):
#             patient_id = re.match(r'(P\d+)_.*', os.path.basename(img_path)).group(1)
#             if patient_id not in groups:
#                 groups[patient_id] = []
#             groups[patient_id].append(img_path)
#         return groups
#
#     def get_patient_image_count(self, index):
#         patient_id = re.match(r'(P\d+)_.*', os.path.basename(self.images_path[index])).group(1)
#         return len(self.grouped_images[patient_id])
#
#     def __getitem__(self, idx):
#         class_name = self.images_class[idx]
#         patient_id = re.match(r'(P\d+)_.*', os.path.basename(self.images_path[idx])).group(1)
#         image_filenames = self.grouped_images[patient_id]
#         image_list = []
#
#         for img_path in image_filenames:
#             img = Image.open(img_path).convert('L')  # Assuming grayscale images
#             img_np = np.array(img)
#             image_list.append(img_np)
#
#         volume = np.stack(image_list, axis=0)
#
#         if self.transform:
#             volume = self.transform(volume)
#
#         volume_tensor = torch.from_numpy(volume)
#
#         return volume_tensor, class_name

