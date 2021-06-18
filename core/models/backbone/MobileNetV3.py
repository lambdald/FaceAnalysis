from __future__ import division
import torch
import torch.nn as nn
import sys

sys.path.append('/home/lidong/code/DistributedFurnace')

from core.models.backbone.BaseBackbone import BaseBackbone
bn_momentum = 0.01

class hswish(nn.Module):

    def __init__(self, inplace=False):
        super(hswish, self).__init__()
        self.act = nn.ReLU6(inplace)

    def forward(self, x):
        return x * self.act(x + 3) / 6


class hsigmoid(nn.Module):
    def __init__(self, ):
        super(hsigmoid, self).__init__()

    @staticmethod
    def forward(x):
        x = (x + 1) / 2
        return x.clamp_(0, 1)


class SE2d(nn.Module):
    def __init__(self, planes, reduction=4):
        super(SE2d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(planes, planes // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(planes // reduction, planes),
            hsigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()

    def forward(self, x):
        n, c, _, _ = x.size()
        y = self.avg_pool(x).view(n, c)
        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        hswish(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        hswish(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, c_in, c_out, ks, exp_size, se, act, s):
        super(InvertedResidual, self).__init__()
        self.use_res_connect = (s == 1 and c_in == c_out)
        if se:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(c_in, exp_size, 1, 1, 0, bias=False),
                nn.BatchNorm2d(exp_size),
                act(inplace=True),
                # dw
                nn.Conv2d(exp_size, exp_size, ks, s, padding=ks // 2, groups=exp_size, bias=False),
                nn.BatchNorm2d(exp_size),
                act(inplace=True),
                SE2d(exp_size),
                # pw-linear
                nn.Conv2d(exp_size, c_out, 1, 1, 0, bias=False),
                nn.BatchNorm2d(c_out),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(c_in, exp_size, 1, 1, 0, bias=False),
                nn.BatchNorm2d(exp_size),
                act(inplace=True),
                # dw
                nn.Conv2d(exp_size, exp_size, ks, s, padding=ks // 2, groups=exp_size, bias=False),
                nn.BatchNorm2d(exp_size),
                act(inplace=True),
                # pw-linear
                nn.Conv2d(exp_size, c_out, 1, 1, 0, bias=False),
                nn.BatchNorm2d(c_out),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(BaseBackbone):
    def __init__(self, interverted_residual_setting, last_channel, final_bn,  **kwargs):
        super().__init__()
        self.input_shape = kwargs['input_shape']
        self.get_shape = False
        self.param_setted = False
        block = InvertedResidual
        input_channel = 16
        self.last_channel = last_channel
        self.init_channel = self.input_shape[0]

        self.out_channels = kwargs['out_channels']
        self.features = [conv_bn(self.init_channel, input_channel, 2)]

        # self.features = [conv_bn(1, input_channel, 2)]
        # building inverted residual blocks
        for ks, exp_size, c_out, se, act, s in interverted_residual_setting:
            self.features.append(block(input_channel, c_out, ks, exp_size, se, act, s))
            input_channel = c_out
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_out = nn.Sequential(
            nn.Conv2d(self.last_channel, self.out_channels, 1, 1, 0, bias=True),
            hswish(inplace=True)
        )

        self.init_params()

    def forward(self, input):
        x = input['image']
        x = self.features(x)
        x = self.avg_pool(x)
        feature = self.feature_out(x)
        input.update({'feature': feature})
        return input


def mobilenetv3_small(pretrained=False, **kwargs):
    interverted_residual_setting = [
        # ks, exp size, c_out, se,  act, s
        [3, 16, 16, True, nn.ReLU, 2],
        [3, 72, 24, False, nn.ReLU, 2],
        [3, 88, 24, False, nn.ReLU, 1],
        [5, 96, 40, True, hswish, 2],
        [5, 240, 40, True, hswish, 1],
        [5, 240, 40, True, hswish, 1],
        [5, 120, 48, True, hswish, 1],
        [5, 144, 48, True, hswish, 1],
        [5, 288, 96, True, hswish, 2],
        [5, 576, 96, True, hswish, 1],
        [5, 576, 96, True, hswish, 1]
    ]
    model = MobileNetV3(interverted_residual_setting=interverted_residual_setting, last_channel=576, final_bn=True,
                        **kwargs)
    if pretrained:
        raise NotImplementedError('pretrained weights unavailable')
    return model


def mobilenetv3_big(pretrained=False, **kwargs):
    interverted_residual_setting = [
        # ks, exp size, c_out, se,  act, s
        [3, 16, 16, False, nn.ReLU, 1],
        [3, 64, 24, False, nn.ReLU, 2],
        [3, 72, 24, False, nn.ReLU, 1],
        [5, 72, 40, True, nn.ReLU, 2],
        [5, 120, 40, True, nn.ReLU, 1],
        [5, 120, 40, True, nn.ReLU, 1],
        [3, 240, 80, False, hswish, 2],
        [3, 200, 80, False, hswish, 1],
        [3, 184, 80, False, hswish, 1],
        [3, 184, 80, False, hswish, 1],
        [3, 480, 112, True, hswish, 1],
        [3, 672, 112, True, hswish, 1],
        [5, 672, 160, True, hswish, 1],
        [5, 672, 160, True, hswish, 2],
        [5, 960, 160, True, hswish, 1],

    ]
    model = MobileNetV3(interverted_residual_setting=interverted_residual_setting, last_channel=960, final_bn=False,
                        **kwargs)
    if pretrained:
        raise NotImplementedError('pretrained weights unavailable')
    return model


if __name__ == '__main__':

    print('test')
    t_d = torch.zeros([4, 3, 224, 224])

    model = mobilenetv3_big(input_shape=[3, 224, 224])
    print(model)
    model({'images': t_d})

    model1 = mobilenetv3_small(input_shape=[3, 224, 224])
    print(model1)
    model1({'images': t_d})
