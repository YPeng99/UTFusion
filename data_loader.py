import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as tf



class CTMRIDataset(Dataset):
    def __init__(self, path="/home/data/SwitchFusion/Havard/MyDatasets/CT-MRI/test"):
        super().__init__()
        self.source_1 = os.path.join(path, 'CT')
        self.source_2 = os.path.join(path, 'MRI')
        self.file_list = os.listdir(self.source_1)
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, ], std=[0.5, ])
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        S1 = Image.open(os.path.join(self.source_1, self.file_list[index]))
        S2 = Image.open(os.path.join(self.source_2, self.file_list[index]))
        S1 = self.to_tensor(S1)
        S2 = self.to_tensor(S2)
        return S1, S2,0,self.file_list[index]


class PETMRIDataset(Dataset):
    def __init__(self, path="/home/data/SwitchFusion/Havard/MyDatasets/PET-MRI/test"):
        super().__init__()
        self.source_1 = os.path.join(path, 'PET')
        self.source_2 = os.path.join(path, 'MRI')
        self.file_list = os.listdir(self.source_1)
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, ], std=[0.5, ])
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        S1 = Image.open(os.path.join(self.source_1, self.file_list[index]))
        S2 = Image.open(os.path.join(self.source_2, self.file_list[index]))
        S1 = self.to_tensor(S1)
        S2 = self.to_tensor(S2)
        return S1, S2,0,self.file_list[index]


class LytroDataset(Dataset):
    def __init__(self, path="/home/data/SwitchFusion/Lytro"):
        super().__init__()
        self.source_1 = os.path.join(path, 'A_jpg')
        self.source_2 = os.path.join(path, 'B_jpg')
        self.file_list = os.listdir(self.source_1)

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, ], std=[0.5, ])
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        S1 = Image.open(os.path.join(self.source_1, self.file_list[index]))
        S2 = Image.open(os.path.join(self.source_2, self.file_list[index].replace('A', 'B')))
        S1 = self.to_tensor(S1)
        S2 = self.to_tensor(S2)
        return S1, S2,0,self.file_list[index].replace('-A','')


class MEFBDataset(Dataset):
    def __init__(self, path="/home/data/SwitchFusion/MEFB"):
        super().__init__()
        self.path = path
        self.file_list = os.listdir(path)
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, ], std=[0.5, ])
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file = self.file_list[index]
        image_files = os.listdir(os.path.join(self.path, file))
        S1, S2 = None, None

        for image in image_files:
            if '_A' in image or '_a' in image:
                S1 = Image.open(os.path.join(self.path, file, image))
                S1 = self.to_tensor(S1)
                continue
            if '_B' in image or '_b' in image:
                S2 = Image.open(os.path.join(self.path, file, image))
                S2 = self.to_tensor(S2)
                continue
        return S1, S2,1,file+'.jpg'

class MSRSDataset(Dataset):
    def __init__(self, path="/home/data/SwitchFusion/MSRS-main/test"):
        super().__init__()
        self.source_1 = os.path.join(path, 'ir')
        self.source_2 = os.path.join(path, 'vi')
        self.file_list = os.listdir(self.source_1)

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, ], std=[0.5, ])
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        S1 = Image.open(os.path.join(self.source_1, self.file_list[index]))
        S2 = Image.open(os.path.join(self.source_2, self.file_list[index].replace('A', 'B')))
        S1 = self.to_tensor(S1)
        S2 = self.to_tensor(S2)
        return S1, S2,0,self.file_list[index]


