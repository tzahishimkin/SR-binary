import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import ImageTransforms


class SRDataset(Dataset):
    """
    A PyTorch Dataset to be used by a PyTorch DataLoader.
    """

    def __init__(self, data_folder, mode, crop_size, scaling_factor, lr_img_type, hr_img_type, test_data_name=None):
        """
        :param data_folder: image folder
        :param mode: 'train' or 'test'
        :param crop_size: crop size of target HR images
        :param scaling_factor: the input LR images will be downsampled from the target HR images by this factor; the scaling done in the super-resolution
        :param lr_img_type: the format for the LR image supplied to the model; see convert_image() in utils.py for available formats
        :param hr_img_type: the format for the HR image supplied to the model; see convert_image() in utils.py for available formats
        :param test_data_name: if this is the 'test' split, which test dataset? (for example, "Set14")
        """

        self.data_folder = data_folder
        self.crop_size = int(crop_size)
        self.scaling_factor = int(scaling_factor)
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type
        self.test_data_name = test_data_name
        self.mode = mode

        if self.mode == 'train':
            assert self.crop_size % self.scaling_factor == 0, "In SR model, crop dimensions are not perfectly divisible by scaling factor!"

        files = os.listdir(data_folder)
        self.images = [os.path.join(data_folder, file) for file in files]
        self.transform = ImageTransforms(split=self.mode,
                                         crop_size=self.crop_size,
                                         scaling_factor=self.scaling_factor,
                                         lr_img_type=self.lr_img_type,
                                         hr_img_type=self.hr_img_type)

    def __getitem__(self, i):
        """
        :param i: index to retrieve
        :return: the 'i'th pair LR and HR images to be fed into the model
        """
        # Read image
        img = Image.open(self.images[i], mode='r')
        # if self.lr_img_type == '[0, 1]':
        #     img = img.convert('L')
        # else:
        img = img.convert('RGB')

        if img.width <= 96 or img.height <= 96:
            print(self.images[i], img.width, img.height)
        lr_img, hr_img = self.transform(img)

        return lr_img, hr_img

    def __len__(self):
        """
        This method is required to be defined for use in the PyTorch DataLoader.

        :return: size of this data (in number of images)
        """
        return len(self.images)
