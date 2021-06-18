'''
Author: lidong
Date: 2021-06-15 16:48:54
LastEditors: lidong
LastEditTime: 2021-06-15 16:53:05
Description: file content
'''
import random
import numpy as np
import torch


def set_random_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.random.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(s)