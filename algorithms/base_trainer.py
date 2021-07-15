'''
Author: lidong
Date: 2021-05-06 09:28:37
LastEditors: lidong
LastEditTime: 2021-06-23 14:57:29
Description: file content
'''
import torch
from torch.cuda.amp import GradScaler, autocast

class BaseTrainer:
    
    def __init__(self, net, optimizer, lr_scheduler, train_loader, test_loader, cfg, recorder=None) -> None:
        
        self.net = net
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.recorder = recorder
        self.cfg = cfg
        self.rank = self.cfg['rank']

    def fetch_wrap_batch(self, images, targets, is_train=True):
        inputs = {
            'image': images.cuda(),
            'target': [t.cuda() for t in targets],
            'is_train': is_train,
        }
        return inputs


    def train(self):
        scaler = GradScaler()
        epoch = self.cfg['strategy']['epoch']
        if self.rank == 0:
            self.recorder.start_train()
        for ep in range(epoch):
        # set sampler
            self.train_loader.sampler.set_epoch(ep)
            if self.rank == 0:
                self.recorder.start_batch()

            #! train
            self.net.train()
            for idx, (images, targets) in enumerate(self.train_loader):

                if len(targets)==0:
                    continue

                inputs = self.fetch_wrap_batch(images, targets)
                
                if self.cfg['strategy'].get('use_amp', False):
                    outputs = self.net(inputs)
                else:
                    with autocast():
                        outputs = self.net(inputs)
    
                loss = outputs['loss']['sum']
                loss_with_name = outputs['loss']['named_loss']
                self.optimizer.zero_grad()
                if self.cfg['strategy'].get('use_amp', False):
                    loss.backward()
                    self.optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                # if rank==0:
                #     print(f'{idx}/{loss.item():.3f}', end='\t')
                if self.rank == 0:
                    self.recorder.record_batch(loss.item(), loss_with_name)
                        
            self.lr_scheduler.step()
            if self.rank == 0:
                self.eval()
            if self.rank == 0:
                self.recorder.record_epoch(self.net)

    def eval(self):
        #! eval
        if self.rank == 0:
            self.recorder.start_eval_batch()
            for idx, (images, targets) in enumerate(self.test_loader):
                if len(targets)==0:
                    continue
                inputs = {
                    'image': images.cuda(),
                    'target': [t.cuda() for t in targets],
                    'is_train': False,
                }
                outputs = self.net(inputs)
                loss = outputs['loss']['sum']
                loss_with_name = outputs['loss']['named_loss']
                self.recorder.record_eval_batch(loss.item(), loss_with_name)
            self.recorder.record_eval_epoch()