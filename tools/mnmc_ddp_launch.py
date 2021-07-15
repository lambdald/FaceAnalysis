'''
Author: lidong
Date: 2021-04-30 15:19:42
LastEditors: lidong
LastEditTime: 2021-07-07 16:12:24
Description: 
Multiple Nodes Multi-GPU Cards Training with DistributedDataParallel and torch.distributed.launch
'''
import os
import sys
import torch
import argparse
import time
import json

from torch import tensor
from torch.nn.functional import feature_alpha_dropout
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print('distributed train framework:', sys.path[-1])

from algorithms.ssd_facedet import DetectionTrainer
from core.utils.recorder import Recorder
from core.utils.random import set_random_seed
from core.build_optimizer import build_optimizer
from core.build_dataloader import build_dataloader
from core.build_models import build_models
from core.utils.config import parse_cfg_file
from core.utils.dist_train import init_distributed_env




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='default.yaml',
                        type=str, help='yaml train config file.')
    parser.add_argument('--local_rank', type=int, help='local rank id.')

    return parser.parse_args()


def fetch_input_data(images, targets, is_train, is_inference):

    if isinstance(targets, list):
        labels = [t.cuda() for t in targets]
    elif isinstance(targets, dict):
        lables = {k:v.cuda() for k, v in targets.items()}
    elif isinstance(targets, torch.tensor):
        lables = targets.cuda()
    else:
        raise NotImplementedError('unknown targets type:', type(targets))
        
    inputs = {
        'image': images.cuda(),
        'target': lables,
        'is_train': is_train,
        'is_inference': is_inference
    }
    return inputs

def main():
    set_random_seed(1024)
    #! 1. parse args.
    args = parse_args()
    cfg_filepath = args.config
    if not os.path.exists(cfg_filepath):
        print(f'config file <{cfg_filepath}> does not exist.')
        exit(0)
    cfg = parse_cfg_file(cfg_filepath)

    #! 2. initialize distributed environment.
    world_size, rank, local_rank = init_distributed_env()
    if local_rank == 0:
        print(json.dumps(cfg, indent=2, ensure_ascii=False))

    print(f"[init] == local rank: {local_rank}, global rank: {rank} ==")
    # add rank info to cfg.
    cfg.update({
        'rank': rank,
        'local_rank': local_rank,
        'world_size': world_size
    })

    #! 3. build network.
    cfg['isTrain'] = True
    models = build_models(cfg).cuda()
    net = DDP(models, device_ids=[local_rank], output_device=local_rank)

    #! 4. build data_loader.
    train_loader, test_loader = build_dataloader(cfg, is_train=True)

    #! 5. build optimizer and lr_shcheduler.

    optimizer, lr_scheduler = build_optimizer(cfg, net)

    #! 6. build recorder.
    if rank == 0:
        recorder = Recorder(cfg)
        recorder.viz_network(net, cfg['net']['input_shape'])

    #! 7. start to train.
    if rank == 0:
        print("  =======  Training  =======   \n")

    scaler = GradScaler()
    epoch = cfg['strategy']['epoch']
    if rank == 0:
        recorder.start_train()
    for ep in range(epoch):
        # set sampler
        train_loader.sampler.set_epoch(ep)
        if rank == 0:
            recorder.start_batch()

        #! train
        net.train()
        for idx, (images, targets) in enumerate(train_loader):

            if len(targets) == 0:
                continue

            inputs = fetch_input_data(images, targets, True, False)


            if cfg['strategy'].get('use_amp', False):
                outputs = net(inputs)
            else:
                with autocast():
                    outputs = net(inputs)

            loss = outputs['loss']['sum']
            if torch.isnan(loss):
                recorder.log(f'batch {idx} loss is nan.')
                continue

            loss_with_name = outputs['loss']['named_loss']
            optimizer.zero_grad()
            if cfg['strategy'].get('use_amp', False):
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            # if rank==0:
            #     print(f'{idx}/{loss.item():.3f}', end='\t')
            if rank == 0:
                recorder.record_batch(loss.item(), loss_with_name)

        lr_scheduler.step()

        #! eval
        if rank == 0 and cfg['strategy']['eval']:
            with torch.no_grad():
                net.eval()
                recorder.start_eval_batch()
                for idx, (images, targets) in enumerate(test_loader):
                    if len(targets) == 0:
                        continue
                    inputs = fetch_input_data(images, targets, True, True)
                    outputs = net(inputs)
                    loss = outputs['loss']['sum']
                    loss_with_name = outputs['loss']['named_loss']
                    recorder.record_eval_batch(loss.item(), loss_with_name)
                recorder.record_eval_epoch()

        if rank == 0:
            recorder.record_epoch(net)


if __name__ == "__main__":
    # cv2.setNumThreads(0)
    # cv2.ocl.setUseOpenCL(False)
    # torch.multiprocessing.set_sharing_strategy('file_system') # 可能会导致最后一个batch阻塞，有守护进程没有退出
    # torch.multiprocessing.set_start_method('spawn') # 子进程启动的方式

    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    main()
    print('Done.')
