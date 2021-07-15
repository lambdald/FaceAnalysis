'''
Author: lidong
Date: 2021-04-30 14:57:28
LastEditors: lidong
LastEditTime: 2021-07-06 16:11:43
Description: file content
'''
import torch
from torch import nn
import importlib
import abc
from core.utils.build_from_arch import build_from_arch

class ForwardNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def build_models(cfg):
        net = ForwardNetwork()
        net.cfg = cfg
        cfg_net = cfg['net']
        intput_shape = cfg_net['input_shape']
        cfg_backbone = cfg_net['backbone']
        cfg_backbone['kwargs']['net_cfg']= cfg_net['net_cfg']
        cfg_backbone['kwargs']['input_shape'] = intput_shape
        net.backbone = build_from_arch(cfg_backbone['arch'], cfg_backbone['kwargs'])
        net.backbone.init_params()
        out_feature_shape = net.backbone.get_output_shape()
        print('backbone output:', out_feature_shape)

        
        out_feature_channels = out_feature_shape[0]

        cfg_head = cfg_net['head']
        net.heads = []
        for name, head in cfg_head.items():
            
            head['loss']['kwargs']['net_cfg'] = cfg_net['net_cfg']

            # 每一个head都有一个loss
            criterion = build_from_arch(head['loss']['arch'], head['loss']['kwargs'])

            head['kwargs'].update({
                'in_channels': out_feature_channels,
                'criterion': criterion,
                'name': name,
                'net_cfg': cfg_net['net_cfg']
            })
            head['kwargs']['isTrain'] = cfg['isTrain']
            net.heads.append(build_from_arch(head['arch'], head['kwargs']))
            net.heads[-1].init_params()
            setattr(net, name, net.heads[-1])

            out = net.heads[-1].get_output_shape()['head'][name]['output']
            if hasattr(out, 'shape'):
                print(f'head {name} output:', out.shape[1:])
            else:
                print(f'head {name} output type: ', type(out))
        return net


    def forward(self, input: dict):
        input = self.backbone(input)

        input['head'] = {}
        for hid, head in enumerate(self.heads):
            input = head(input)
        
        if not input.get('is_train', True):
            return input

        loss = []
        loss_with_name = {}
        for name, head in self.cfg['net']['head'].items():
            loss.append(input['head'][name]['loss'] * head['alpha'])
            loss_with_name[head['loss']['name']] = input['head'][name]['loss'].item()


        #! record all loss to input dict.
        input['loss'] = {}
        input['loss']['named_loss'] = loss_with_name
        input['loss']['sum'] = sum(loss)
        return input

    def init_params(self):
        self.backbone.init_params()
        for head in self.heads:
            head.init_params()
            