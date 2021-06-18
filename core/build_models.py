'''
Author: lidong
Date: 2021-04-29 16:39:28
LastEditors: lidong
LastEditTime: 2021-06-09 14:10:14
Description: file content
'''
import torch

from core.models.ForwardNetwork import ForwardNetwork

def build_models(cfg):

    model = ForwardNetwork.build_models(cfg)

    model.init_params()

    return model
