import torch
from core.models.head.BaseHead import BaseHead
from torch import nn

class ClassifyHeader(BaseHead):
    def __init__(self, **kwargs):
        super().__init__()

        self.in_channels = kwargs['in_channels']
        self.out_dim = kwargs['num_classes']

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channels, self.out_dim)
        self.args = kwargs
    def forward(self, input):

        b = input['feature'].shape[0]
        x = self.avgpool(input['feature']).view(b, -1)
        out = self.fc(x)

        input[self.args['name']] = {
            'output': out,
        }

        if self.args['isTrain']:
            loss = self.args['criterion'](out, input['target'])
            input[self.args['name']].update({
                    'loss': loss
                })
        return input
