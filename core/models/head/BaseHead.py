'''
Author: lidong
Date: 2021-04-30 14:52:40
LastEditors: lidong
LastEditTime: 2021-07-05 13:48:26
Description: file content
'''
import torch
from torch import nn

class BaseHead(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.in_channels = 0
        self.out_dim = 0
        self.kwargs = kwargs

    def get_output_shape(self):
        assert self.in_channels != 0
        # tmp_input = {'images': torch.rand(torch.Size(self.input_shape)).unsqueeze(0)}
        feature = torch.rand((10, self.in_channels, 5, 5))
        target = torch.randint(0, 9, (10,))
        tmp_input = {
            'feature': feature,
            'target': target,
            'head':{},
            'is_train': False}
        # pdb.set_trace()
        tmp_output = self.forward(tmp_input)
        return tmp_output


    def init_params(self):
        for m in self.modules():
            if hasattr(m, 'requires_grad') and not m.requires_grad:
                continue
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.PReLU):
                torch.nn.init.xavier_normal_(m.weight)
