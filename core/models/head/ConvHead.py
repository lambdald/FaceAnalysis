'''
Author: lidong
Date: 2021-07-02 14:32:11
LastEditors: lidong
LastEditTime: 2021-07-06 15:34:30
Description: file content
'''

import torch
from core.models.head.BaseHead import BaseHead
from torch import nn

class ConvHead(BaseHead):
    def __init__(self, in_channels, out_channels, use_act=True, **kwargs):
        super().__init__(**kwargs)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

        self.use_act = use_act
        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.use_act:
            self.act = nn.Sigmoid()

    def get_output_shape(self):
        assert self.in_channels != 0
        # tmp_input = {'images': torch.rand(torch.Size(self.input_shape)).unsqueeze(0)}
        feature = torch.rand((10, self.in_channels, 5, 5))
        target = torch.rand((10, self.out_channels, 5, 5))
        tmp_input = {
            'feature': feature,
            'target': 
            {
                self.kwargs['name']: target
            },
            'head':{},
            'is_train': False}
        # pdb.set_trace()
        tmp_output = self.forward(tmp_input)
        return tmp_output


    def forward(self, input):

        loss_func = self.kwargs['criterion']
        feature = input['feature']
        out = self.conv(feature)
        if self.use_act:
            out = self.act(out)
        input['head'][self.kwargs['name']] = {}
        input['head'][self.kwargs['name']]['output'] = out
        if input.get('is_train', True):
            loss = loss_func(out, input['target'][self.kwargs['name']])
            input['head'][self.kwargs['name']]['loss'] = loss
        return input