'''
Author: lidong
Date: 2021-06-23 14:39:36
LastEditors: lidong
LastEditTime: 2021-06-23 15:05:35
Description: file content
'''

from algorithms.base_trainer import BaseTrainer

class DetectionTrainer(BaseTrainer):

    def __init__(self, net, optimizer, lr_scheduler, train_loader, test_loader, cfg, recorder) -> None:
        super().__init__(net, optimizer, lr_scheduler, train_loader, test_loader, cfg, recorder=recorder)

    