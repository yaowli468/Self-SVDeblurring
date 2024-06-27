import os.path
import pdb

import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import random
import cv2


class SingleDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.stage=opt.stage

        self.dir = opt.dataroot
        self.AB_paths = sorted(make_dataset(self.dir))

        if self.stage=='train':
            transform_list = [transforms.ToTensor()]
        else:
            transform_list = [transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5),
                                                   (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        A_path = self.AB_paths[index]

        A_img=Image.open(A_path).convert("RGB")

        A_img = self.transform(A_img)


        return {'A': A_img, 'image_path': A_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'SingleImageDataset'

