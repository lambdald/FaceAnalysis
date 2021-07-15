'''
Author: lidong
Date: 2021-04-30 15:21:20
LastEditors: lidong
LastEditTime: 2021-07-05 13:52:58
Description: file content
'''

import os
from tensorboardX import SummaryWriter
import logging
import os.path as osp
from pathlib import Path

import torch
from core.utils.meter import AverageMeter, Timer, get_time_stamp
import sys

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    # formatter = logging.Formatter(
    #     "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    # )
    formatter = logging.Formatter(
        "[%(asctime)s]%(message)s", datefmt='%Y/%m/%d-%H:%M:%S'
    )

    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    # 禁止log向上传播
    # https://docs.python.org/3/library/logging.html#logging.Logger.propagate
    logger.propagate = False    

    if (logger.hasHandlers()):
        logger.handlers.clear()
    # 写入文件
    fh = logging.FileHandler(filename, "a", encoding='utf-8', delay=True)
    fh.setFormatter(formatter)
    fh.setLevel(level_dict[verbosity])
    logger.addHandler(fh)
    # 终端显示
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    sh.setLevel(level_dict[verbosity])
    logger.addHandler(sh)

    return logger


def make_sure_path_exist(dir):
    if not osp.exists(dir):
        p = Path(dir)
        p.mkdir(parents=True)



class UnWrappedModel(torch.nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, inputs):
        x = {'image': inputs, 'calc_loss': False, 'is_train':False}
        all_in_one = self.net(x)
        out = list(all_in_one['head'][x]['output'] for x in all_in_one['head'])

        return all_in_one['feature']
        if len(out) == 1:
            out = out[0]
        return out


class Recorder:
    def __init__(self, cfg):
        cfg_recorder = cfg['recorder']
        self.num_epoch  = cfg['strategy']['epoch']
        work_dir = cfg_recorder['work_dir']
        self.task_name = cfg['task_name']
        make_sure_path_exist(work_dir)

        self.logger_path = osp.join(work_dir, cfg_recorder['logger_path'])
        self.summary_writer_path = osp.join(
            work_dir, cfg_recorder['summary_writer_dir'])
        make_sure_path_exist(self.summary_writer_path)
        self.logger = get_logger(self.logger_path, name='log')
        self.summary_writer = SummaryWriter(log_dir=osp.join(self.summary_writer_path, get_time_stamp()), comment=self.task_name+'_'+get_time_stamp())
        # self.summary_writer = SummaryWriter(self.summary_writer_path)

        self.logger_epoch_fmt = '({}/{})'
        self.logger_batch_fmt = '[batch {}]'
        self.logger_iter_fmt = '[iter {}]'


        self.last_loss = {}

        self.log_frequency = cfg_recorder['batch_log_frequency']
        self.chpt_frequency = cfg_recorder['checkpoint_frequency']
        self.chpt_dir = osp.join(work_dir, cfg_recorder['checkpoint_dir'])
        make_sure_path_exist(self.chpt_dir)
        self.chpt_save_fmt = osp.join(self.chpt_dir, cfg['task_name']+'_{:03}.pth')
        
    def start_train(self):
        """准备训练，记录训练和测试的结果，以及耗时（评估训练时间）
        """
        self.train_loss_meter_epoch = AverageMeter(name=f'TrainLoss')
        self.eval_loss_meter_epoch = AverageMeter(name=f'EvalLoss')
        self.timer_epoch = Timer('epoch')
        self.timer_epoch.start()
        self.epoch_id = 0
        self.iter = 0
    
    # epoch
    def record_epoch(self, net):
        self.timer_epoch.record()
        self.train_loss_meter_epoch.update(self.train_loss_meter_batch.avg)
        self.summary_writer.add_scalar(
            f'train_loss', self.train_loss_meter_batch.avg, self.epoch_id)
        self.epoch_id += 1

        self.log_epoch_info()
        self.save_checkpoint(net)
        
    def log_epoch_info(self):
        
        eta = self.timer_epoch.ETA(self.num_epoch)

        s = (self.logger_iter_fmt.format(self.iter)
        + self.logger_epoch_fmt.format(self.epoch_id, self.num_epoch)
        + '[' + str(self.train_loss_meter_epoch) +']'
        + f'[{self.timer_epoch.avg_meter} s|ETA:{eta/60/60:.3f} h]')
        self.logger.info(s)

    # 记录train batch信息
    def start_batch(self):
        self.batch_id = 0
        self.train_loss_meter_batch = AverageMeter(name=f'BatchLoss')
        self.timer_batch = Timer('batch')
        self.timer_batch.start()

    def record_batch(self, loss_all: float, loss: dict):
        self.train_loss_meter_batch.update(loss_all)
        self.timer_batch.record()
        for name in loss:
            self.summary_writer.add_scalar(f'train_loss_{name}', loss[name], self.iter)
        self.summary_writer.add_scalar(
            f'train_loss_all', loss_all, self.iter)
        self.last_loss = loss
        self.batch_id += 1
        self.iter += 1
        self.log_batch_info()

    def log_batch_info(self):
        if self.batch_id % self.log_frequency != 0:
            return   
        s = (self.logger_iter_fmt.format(self.iter)
        + self.logger_epoch_fmt.format(self.epoch_id, self.num_epoch)
        + self.logger_batch_fmt.format(self.batch_id)
        + '[' + ('|'.join([f'{k}-{v:.3f}' for k,v in self.last_loss.items()])) + ']'
        + '[' + str(self.train_loss_meter_batch) +']'
        + f'[{self.timer_batch.avg_meter} s]')

        self.logger.info(s)


    # eval
    def start_eval_batch(self):
        self.eval_batch_id = 0
        self.eval_loss_meter_batch = AverageMeter(name='BatchLoss')
        self.timer_batch_eval = Timer('batch')
        self.timer_batch_eval.start()

    def record_eval_batch(self, loss_all: float, loss: dict):
        self.eval_loss_meter_batch.update(loss_all)
        self.timer_batch_eval.record()
        # for name in loss:
        #     self.summary_writer.add_scalar(f'eval_loss_{name}', loss[name], self.iter)
        # self.summary_writer.add_scalar(
        #     f'eval_loss_all', loss_all, self.iter)
        self.last_loss = loss
        self.eval_batch_id += 1

        self.log_eval_batch_info()

    def log_eval_batch_info(self):
        if self.eval_batch_id % self.log_frequency != 0:
            return   
        s = ("Eval="+self.logger_epoch_fmt.format(self.epoch_id, self.num_epoch)
        + self.logger_batch_fmt.format(self.batch_id)
        + '[' + ('|'.join([f'{k}-{v:.3f}' for k,v in self.last_loss.items()])) + ']'
        + '[' + str(self.eval_loss_meter_batch) +']'
        + f'[{self.timer_batch_eval.avg_meter} s]')

        self.logger.info(s)
    # epoch
    def record_eval_epoch(self):
        self.eval_loss_meter_epoch.update(self.eval_loss_meter_batch.avg)
        self.summary_writer.add_scalar(
            f'eval_loss', self.eval_loss_meter_batch.avg, self.epoch_id)
        self.log_eval_epoch_info()
        
    def log_eval_epoch_info(self):
        s = ("Eval=" + self.logger_epoch_fmt.format(self.epoch_id, self.num_epoch)
        + '[' + str(self.eval_loss_meter_epoch) +']'
        + f'[{self.timer_batch_eval.avg_meter.sum} s]')
        self.logger.info(s)


    def save_checkpoint(self, net: torch.nn.Module):
        
        if self.epoch_id % self.chpt_frequency != 0:
            return
        save_path = self.chpt_save_fmt.format(self.epoch_id-1)
        torch.save(net.state_dict(), save_path)
        self.logger.info('save checkpoint to '+ save_path)

    def load_checkpoint_by_epoch(self, epoch_id, net: torch.nn.Module):
        assert epoch_id < self.num_epoch
        chpt_path = self.chpt_save_fmt.format(epoch_id)
        self.load_checkpoint(chpt_path, net)

    def load_checkpoint(self, chpt_path, net: torch.nn.Module):
        assert os.path.exists(chpt_path)
        net.load_state_dict(torch.load(chpt_path))
        

    def log_final_loss(self):
        # TODO
        pass

    def log(self, info):
        self.logger.info(info)

    def viz_network(self, net, input_shape):

        dummy = torch.autograd.Variable(torch.randn(input_shape).unsqueeze(0).cuda())
        real_net = UnWrappedModel(net)
        self.summary_writer.add_graph(real_net, dummy)