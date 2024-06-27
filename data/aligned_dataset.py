import os.path
import pdb
import random
import imageio

import numpy
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import util.util as util

from skimage.transform import pyramid_gaussian
import cv2
import numpy as np
from scipy import signal
from scipy import misc

import matplotlib.pyplot as plt


class AlignedDataset(BaseDataset):
    def __init__(self, opt):
        # super(AlignedDataset,self).__init__(opt)
        self.opt = opt
        self.root = opt.dataroot

        self.dir = opt.dataroot
        self.AB_paths = sorted(make_dataset(self.dir))

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')

        # aug = random.randint(0, 2)
        # if aug == 1:
        #     sat_factor = 1 + (0.2 - 0.4*np.random.rand())
        #     AB = torchvision.transforms.functional.adjust_saturation(AB, sat_factor)

        AB = self.transform(AB)

        w_total = AB.size(2)
        w = int(w_total / 2)
        h = AB.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineWidthSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineheightSize - 1))

        A = AB[:, h_offset:h_offset + self.opt.fineheightSize,   # Blurred
            w_offset:w_offset + self.opt.fineWidthSize]
        B = AB[:, h_offset:h_offset + self.opt.fineheightSize,    # Ground Truth
            w + w_offset:w + w_offset + self.opt.fineWidthSize]

        aug = random.randint(0, 8)
        if aug==1:
            A = A.flip(1)
            B = B.flip(1)
        elif aug==2:
            A = A.flip(2)
            B = B.flip(2)
        elif aug==3:
            A = torch.rot90(A,dims=(1,2))
            B = torch.rot90(B,dims=(1,2))
        elif aug==4:
            A = torch.rot90(A,dims=(1,2), k=2)
            B = torch.rot90(B,dims=(1,2), k=2)
        elif aug==5:
            A = torch.rot90(A,dims=(1,2), k=3)
            B = torch.rot90(B,dims=(1,2), k=3)
        elif aug==6:
            A = torch.rot90(A.flip(1),dims=(1,2))
            B = torch.rot90(B.flip(1),dims=(1,2))
        elif aug==7:
            A = torch.rot90(A.flip(2),dims=(1,2))
            B = torch.rot90(B.flip(2),dims=(1,2))

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}


    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'

class AlignedValDataset(BaseDataset):
    def __init__(self, opt):
        self.opt = opt
        self.valide_dir = opt.validateSet
        self.val_image_path=os.path.join(self.valide_dir, 'combined')

        self.val_AB_paths = sorted(make_dataset(self.val_image_path))

        transform_list_A = [transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5),
                                                 (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list_A)

    def __getitem__(self, index):

        AB_path = self.val_AB_paths[index]

        AB = Image.open(AB_path).convert("RGB")

        AB=self.transform(AB)

        w_total = AB.size(2)
        w = int(w_total / 2)
        h = AB.size(1)
        w_offset = random.randint(0, max(0, w - 256 - 1))
        h_offset = random.randint(0, max(0, h - 256 - 1))

        A = AB[:, h_offset:h_offset + 256, w_offset:w_offset + 256]     # Blurred
        B = AB[:, h_offset:h_offset + 256, w + w_offset:w + w_offset + 256]  # Ground Truth

        return {'val_blur': A, 'val_sharp': B}

    def __len__(self):
        return len(self.val_AB_paths)

