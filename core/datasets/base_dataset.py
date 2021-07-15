'''
Author: lidong
Date: 2021-06-03 16:58:51
LastEditors: lidong
LastEditTime: 2021-07-02 09:44:00
Description: file content
'''

from abc import abstractclassmethod
import torch
from torch.utils.data import Dataset
from typing import List
import itertools
import cv2

import numpy as np

class BaseDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()

    @abstractclassmethod
    def pull_item(self, index) -> list:
        pass

    @abstractclassmethod
    def __len__(self) -> int:
        pass

    def read_image(self, img_path):
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    @staticmethod
    def to_chw(cv_image):
        return np.transpose(cv_image, [2, 0, 1])

class BaseProcessor(Dataset):

    def __init__(self) -> None:
        pass

    @abstractclassmethod
    def __getitem__(self, index):
        pass

    @abstractclassmethod
    def process(self, index):
        pass
    
    @abstractclassmethod
    def post_process(self, pred, kwargs):
        pass

class FullDatasets(Dataset):
    '''
    Multiple Dataset
    '''
    def __init__(self, datasets: List[BaseDataset]) -> None:
        super().__init__()

        self.datasets = datasets
        self.n = len(self.datasets)
        f = lambda a, b: a+len(b)
        self.sizes = [len(s) for s in datasets]
        self.accu_sizes = list(itertools.accumulate(self.sizes))
        assert self.n > 0 and self.accu_sizes[-1] > 0, 'Empty Datasets.'

    def pull_item(self, index) -> list:
        for i in range(self.n):
            if self.accu_sizes[i] > index:
                sub_index = -(self.accu_sizes[i] - index)
                return self.datasets[i][sub_index]
        return None

    def __len__(self) -> int:
        return self.accu_sizes[-1]
        
    def __getitem__(self, index):
        return self.pull_item(index)