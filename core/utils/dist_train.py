'''
Author: lidong
Date: 2021-04-30 13:08:16
LastEditors: lidong
LastEditTime: 2021-06-17 10:25:39
Description: file content
'''
import os
 
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def init_distributed_env():
    # 初始化分布式训练环境。
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device_count = torch.cuda.device_count()

    # 设置默认的设备编号
    torch.cuda.set_device(local_rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl")
    device_ids = list(range(device_count))

    return [world_size, rank, local_rank] 