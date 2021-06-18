'''
Author: lidong
Date: 2021-06-07 20:06:45
LastEditors: lidong
LastEditTime: 2021-06-17 10:30:09
Description: file content
'''

from numpy.lib.arraysetops import isin
import torch

from core.models.head.BaseHead import BaseHead

class LossOnlyHead(BaseHead):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs
        self.in_channels = kwargs['in_channels']

    def get_output_shape(self):
        return {self.kwargs['name']: {'output':None}}   # 暂时无法得知输出的类型和shape

    def forward(self, input):
        # backbone需要把feature和output存到字典里
        x = input['output']
        
        loss = self.kwargs['criterion']
        input['head'][self.kwargs['name']] = {
            'output': x,
        }

        if self.kwargs['isTrain']:
            loss = self.kwargs['criterion'](x, input['target'])

            if isinstance(loss, tuple) or isinstance(loss, list):
                loss = sum(loss)
            input['head'][self.kwargs['name']].update({
                    'loss': loss
                })

        return input