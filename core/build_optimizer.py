'''
Author: lidong
Date: 2021-05-06 09:32:37
LastEditors: lidong
LastEditTime: 2021-06-17 09:55:48
Description: file content
'''

import torch
from torch.optim import lr_scheduler, optimizer
from core.optimizers.gradual_warmup_lr_scheduler import GradualWarmupScheduler

def impl_build_optimizer(cfg, net: torch.nn.Module):
    optim_type = cfg['type'].lower()
    if optim_type == 'sgd':
        optim = torch.optim.SGD(net.parameters(), lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    elif optim_type == 'adam':
        optim = torch.optim.Adam(net.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    else:
        raise NotImplementedError('unknown optimizer type: ', cfg['type'])
    return optim


def impl_build_scheduler(cfg, optimizer):

    sche_type = cfg['type'].lower()
    if sche_type == 'step':
        sche = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['step_size'], gamma=cfg['gamma'])
    elif sche_type == 'cos':
        sche = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['iter'], eta_min =1e-5)
    elif sche_type == 'multistep':
        sche = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['milestones'], gamma=cfg['gamma'])
    else:
        raise NotImplementedError('unknown scheduler type: ', cfg['type'])

    warmup_epoch = cfg['warmup']
    if warmup_epoch != 0:
        warmup_sche = GradualWarmupScheduler(optimizer, 1, warmup_epoch, sche)
        sche = warmup_sche
    return sche

def build_optimizer(cfg: dict, net):
    cfg_stg = cfg['strategy']
    cfg_optim = cfg_stg['optimizer']
    cfg_sche = cfg_stg['lr_scheduler']
    optimizer = impl_build_optimizer(cfg_optim, net)
    scheduler = impl_build_scheduler(cfg_sche, optimizer)
    return optimizer, scheduler


    