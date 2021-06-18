'''
Author: lidong
Date: 2021-04-30 16:23:34
LastEditors: lidong
LastEditTime: 2021-06-17 16:27:15
Description: file content
'''
import numpy as np
from functools import reduce
import time
import datetime

def get_time_stamp() -> str:
    dt=datetime.datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
    return dt

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, length=0, fmt=':.3f'):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.maxlen = length

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.history = []

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1

        self.history.append(val)
        if self.maxlen > 0 and len(self.history) > self.maxlen:
            self.sum -= self.history[0]
            del self.history[0]
            self.count -= 1
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class Timer:
    def __init__(self, name):
        self.avg_meter = AverageMeter(name=f'{name}_timer')
        self.time_points = []
    def start(self):
        self.time_points = []
        self.record()
    def record(self):
        self.time_points.append(time.time())
        if len(self.time_points) > 1:
            self.avg_meter.update(self.time_points[-1] - self.time_points[-2])

    def ETA(self, num_iter):
        '''ETA time for train'''
        return (num_iter - len(self.time_points) + 1) * self.avg_meter.avg