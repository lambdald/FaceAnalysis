'''
Author: lidong
Date: 2021-04-30 13:30:10
LastEditors: lidong
LastEditTime: 2021-07-05 13:55:26
Description: file content
'''
import torch
from torch.utils import data

from core.datasets.minst import get_minst_dataset
from torchvision import transforms
from core.datasets.base_dataset import FullDatasets
from core.datasets import widerface

from core.utils.build_from_arch import build_from_arch, find_object_by_arch
def build_datasets(cfg):
    if cfg['data_fmt'] == 'widerface':
        dataset = widerface.WiderFaceDataset(cfg['data_path'], cfg['data_annot'] , **cfg['kwargs'])
    else:
        raise NotImplementedError('unknown data_fmt:', cfg['data_fmt'])
    return dataset


def build_processed_dataset(cfg):
    dataset_cls = find_object_by_arch(cfg['dataset_arch'])
    processor_cls = find_object_by_arch(cfg['processor_arch'])

    class ProcessedDataset(dataset_cls, processor_cls):
        def __init__(self, data_path, data_annot, kwargs) -> None:
            super(ProcessedDataset, self).__init__(data_path, data_annot, kwargs)
    return ProcessedDataset(cfg['data_path'], cfg['data_annot'] , cfg['kwargs'])

def impl_build_dataloader(cfg_train, is_train=False):

    datasets = []
    transforms = build_from_arch(cfg_train['transforms']['pipeline'], cfg_train['transforms']['kwargs'] )
    
    for data_name, data_cfg in cfg_train['datasets'].items():
        data_cfg['kwargs']['transforms'] = transforms
        data = build_processed_dataset(data_cfg)
        # data = build_datasets(data_cfg)
        datasets.append(data)

    full_datasets = FullDatasets(datasets)

    if is_train:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            full_datasets,
            shuffle=True,
            drop_last=False
        )
        train_loader = torch.utils.data.DataLoader(
            full_datasets,
            batch_size=cfg_train['batchsize'],
            num_workers=cfg_train['workers'],
            pin_memory=True,
            sampler=train_sampler,
            collate_fn=build_from_arch(cfg_train['collater']['arch'], cfg_train['collater']['kwargs'])
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            full_datasets,
            batch_size=cfg_train['batchsize'],
            num_workers=cfg_train['workers'],
            pin_memory=True,
            shuffle=True,
            collate_fn=build_from_arch(cfg_train['collater']['arch'], cfg_train['collater']['kwargs'])
        )
    return train_loader

    
def build_dataloader(cfg, is_train=False):
    
    train_loader = impl_build_dataloader(cfg['train'], is_train=True)
    test_loader = impl_build_dataloader(cfg['test'], is_train=False)
    return train_loader, test_loader