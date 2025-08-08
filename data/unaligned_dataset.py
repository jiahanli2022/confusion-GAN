import os
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import random
import torch
from pathlib import Path

def make_dataset(dir_path):
    return sorted([str(p) for p in Path(dir_path).rglob('*.png')])

class UnalignedDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        assert opt.phase == 'train'
        
        self.dir_A = opt.data_train_A
        self.dir_B = opt.data_train_B
        self.A_paths = make_dataset(self.dir_A) 
        self.B_paths = make_dataset(self.dir_B)

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        self.A_labels = torch.load(opt.A_lables) # opt.A_labels is a dictionary

        self.transform_A = get_transform(self.opt, grayscale=False)
        self.transform_B = get_transform(self.opt, grayscale=False)
        self.transform = get_transform(self.opt, grayscale=False)   

    def __getitem__(self, index):
        A_path = self.A_paths[index]  # make sure index is within then range
        index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        assert A_img.size[0] == 256 and A_img.size[0] == 256 and B_img.size[0] == 256 and B_img.size[0] == 256 
        
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        Rs_IHC, Rs_HE = [], []
        for _ in range(32):
            index_RA = random.randint(0, self.A_size - 1)
            index_RB = random.randint(0, self.B_size - 1)

            Rs_HE.append(self.transform(Image.open(self.A_paths[index_RA]).convert('RGB')))
            Rs_IHC.append(self.transform(Image.open(self.B_paths[index_RB]).convert('RGB')))
            
        if int(self.A_labels[A_path]) ==1: # label adapt from One-Class-Classifier
            A_label = torch.Tensor([1, 0])  # OCC positive or Custom
        else:
            A_label = torch.Tensor([0, 1])  # OCC negative or Custom

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'RS_IHC':Rs_IHC, 'RS_HE':Rs_HE, 'A_label':A_label}

    def __len__(self):
        return min(self.A_size, self.B_size)



