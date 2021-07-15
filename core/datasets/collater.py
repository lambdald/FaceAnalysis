'''
Author: lidong
Date: 2021-06-09 10:23:45
LastEditors: lidong
LastEditTime: 2021-07-06 15:36:10
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

class AllTensorCollater:

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
    def __call__(self, data):
        images = []
        targets = []

        for unit in data:
            if len(unit[1])==0:
                continue
            images.append(unit[0])
            targets.append(unit[1])
        return torch.tensor(images), torch.tensor(images)

class DictCollater:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
    def __call__(self, data):
        images = []
        targets = {}

        for unit in data:
            if len(unit[1])==0:
                continue
            images.append(unit[0])

            for k in unit[1]:
                if k in targets:
                    targets[k].append(unit[1][k])
                else:
                    targets[k] = [unit[1][k]]
        for k in targets:
            targets[k] = torch.tensor(targets[k], dtype=torch.float32)

        return torch.tensor(images), targets