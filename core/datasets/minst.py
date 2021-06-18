'''
Author: lidong
Date: 2021-04-30 17:38:27
LastEditors: lidong
LastEditTime: 2021-06-04 16:18:41
Description: file content
'''
import torch
from torch.utils.data import Dataset
import torchvision
from core.datasets.base_dataset import BaseDataset

def get_minst_dataset(data_path, transforms, is_train=True):

    trainset = torchvision.datasets.MNIST(
        root=data_path,
        train=is_train,
        download=True,
        transform=transforms
    )
    return trainset

class MINSTDataset(BaseDataset):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.data = self.get_minst_dataset(**kwargs)
    def __getitem__(self, idx):
        return self.data[idx]
    def __len__(self):
        return len(self.data)
    
