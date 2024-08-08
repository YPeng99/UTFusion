import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as tf

class TrainDataset(Dataset):
    def __init__(self, path="/home/data/SwitchFusion/train"):
        super().__init__()
        self.img_size = 224
        self.correspond = {'CT': 'MRI', 'PET': 'MRI', 'UE': 'OE', 'FFI': 'NFI', 'VI': 'IR'}
        self.source_1 = os.path.join(path, 'source_1')
        self.source_2 = os.path.join(path, 'source_2')
        self.file_list = os.listdir(self.source_1)
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, ], std=[0.5, ])
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        S1 = self.file_list[index]
        fuse_scheme = 0
        type_1 = S1.split('_')[0]
        type_2 = self.correspond[type_1]
        S2 = type_2 + '_' + S1.split('_', maxsplit=1)[1]
        if type_1 == 'OE' or type_1 == 'UE':
            fuse_scheme = 1
            path_1 = os.path.join(self.source_1, S1)
            path_2 = os.path.join(self.source_2, S2)
            i_1 = random.randint(1, len(os.listdir(path_1)))
            i_2 = random.randint(2 + len(os.listdir(path_1)), 2 * len(os.listdir(path_1)) + 1)
            S1 = Image.open(os.path.join(path_1, str(i_1) + '.JPG')).convert('L')
            S2 = Image.open(os.path.join(path_2, str(i_2) + '.JPG')).convert('L')
        else:
            S1 = Image.open(os.path.join(self.source_1, S1)).convert('L')
            S2 = Image.open(os.path.join(self.source_2, S2)).convert('L')
        S1 = self.to_tensor(S1)
        S2 = self.to_tensor(S2)

        h, w = S1.shape[1:]
        if h <= self.img_size or w <= self.img_size:
            S1 = tf.resize(S1, [self.img_size, self.img_size])
            S2 = tf.resize(S2, [self.img_size, self.img_size])
        else:
            h = random.randint(0, h - self.img_size)
            w = random.randint(0, w - self.img_size)

            S1 = tf.crop(S1, h, w, self.img_size, self.img_size)
            S2 = tf.crop(S2, h, w, self.img_size, self.img_size)
        S1, S2 = self._data_augment(S1, S2)
        return S1, S2, fuse_scheme

    def _data_augment(self, S1, S2):
        if random.random() > 0.5:
            S1 = tf.hflip(S1)
            S2 = tf.hflip(S2)

        if random.random() > 0.5:
            S1 = tf.vflip(S1)
            S2 = tf.vflip(S2)

        if random.random() > 0.5:
            S1 = tf.rotate(S1, 90.0)
            S2 = tf.rotate(S2, 90.0)

        return S1, S2


class InferenceDataset(Dataset):
    def __init__(self, root="./datasets"):
        super().__init__()
        self.root = root
        self.file_path = os.listdir(self.root)
        self.file_list = os.listdir(os.path.join(self.root,self.file_path[0]))

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_name = self.file_list[index]
        img_list = []
        for p in self.file_path:
            img_path = os.path.join(self.root,p,img_name)
            img = Image.open(img_path)
            img = self.to_tensor(img)
            img_list.append(img)
        return img_list,img_name



if __name__ == '__main__':
    dataset = InferenceDataset()
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(data_loader.__len__())
    for img_list,img_name in data_loader:
        print(len(img_list))
        print(img_name)
        print(type(img_list))


