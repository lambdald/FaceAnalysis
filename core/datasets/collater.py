'''
Author: lidong
Date: 2021-06-09 10:23:45
LastEditors: lidong
LastEditTime: 2021-06-11 16:10:07
Description: file content
'''

import torch

class DefaultCollater:

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def __call__(self, data):

        images = []
        targets = []

        for unit in data:
            if len(unit[1])==0:
                continue
            images.append(unit[0])
            targets.append(torch.FloatTensor(unit[1]))
        return torch.tensor(images), targets
