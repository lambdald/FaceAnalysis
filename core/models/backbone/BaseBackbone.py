'''
Author: lidong
Date: 2021-04-30 14:25:35
LastEditors: lidong
LastEditTime: 2021-06-17 12:50:28
Description: file content
'''
import torch.nn as nn
import torch
import math
from abc import abstractmethod


class BaseBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = []
        self.param_setted = False
        self.input_shape = None
        self.init_method = 'kaiming'    # kaiming or xavier

    def set_input_shape(self, input_shape):
        self.input_shape = list(input_shape)

    @abstractmethod
    def forward(self, input):
        pass

    def get_output_shape(self):
        """ return [c, h, w] """
        assert self.input_shape != None
        # tmp_input = {'images': torch.rand(torch.Size(self.input_shape)).unsqueeze(0)}
        image = torch.cat((torch.rand(torch.Size(self.input_shape)).unsqueeze(0), torch.rand(torch.Size(self.input_shape)).unsqueeze(0)), dim=0)
        tmp_input = {'image': image}
        # pdb.set_trace()
        tmp_output = self.forward(tmp_input)
        return tmp_output['feature'].shape[1:]

    def init_params(self):
        if self.init_method == 'kaiming':
            init_method = torch.nn.init.kaiming_normal_
        elif self.init_method == 'xavier':
            init_method = torch.nn.init.xavier_normal_
        else:
            print('unknown init method', self.init_method)
            init_method = torch.nn.init.kaiming_normal_


        for m in self.modules():
            if hasattr(m, 'requires_grad') and not m.requires_grad:
                continue
            if isinstance(m, nn.Conv2d):
                init_method(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv3d):
                init_method(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init_method(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.PReLU):
                init_method(m.weight)

