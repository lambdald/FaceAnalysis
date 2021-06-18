'''
Author: lidong
Date: 2021-06-04 17:13:18
LastEditors: lidong
LastEditTime: 2021-06-07 10:35:43
Description: file content
'''
from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        """Generate all prior box from unite center
            坐标和尺寸都是归一化的，方便转换成原图尺寸
        Returns:
            torch.Tensor: prior-box
        """
        mean = []
        # 遍历多尺度的 特征图: [38, 19, 10, 5, 3, 1]
        for k, f in enumerate(self.feature_maps):
            # 笛卡尔积，能够直接生成featureMap的所有xy坐标，相当于遍历featureMap
            for i, j in product(range(f), repeat=2):
                # 特征图的大小=图像尺寸/每个cell的尺寸
                #? 为什么不直接用f？
                f_k = self.image_size / self.steps[k]
                # unit center x,y 归一化坐标
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    # 面积相同的box
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # back to torch land 8732*4
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0) # 截断
        return output
