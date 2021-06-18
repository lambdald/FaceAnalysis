'''
Author: lidong
Date: 2021-04-29 16:09:00
LastEditors: lidong
LastEditTime: 2021-06-04 16:26:54
Description: file content
'''
import yaml

def parse_cfg_file(filename: str)->dict:
    with open(filename, 'rt', encoding='utf8') as infile:
        cfg = yaml.load(infile, Loader=yaml.FullLoader)
    return cfg


def save_cfg_file(cfg, filename):
    with open(filename, 'wt', encoding='utf8') as outfile:
        yaml.dump(cfg, outfile)


if __name__ == '__main__':
    cfg = {
        'net': 'resnet',
        'strategy':
        {
            'lr':0.001,
            'batchsize':256
        }
    }
    filename = 'default.yaml'
    save_cfg_file(cfg,filename)
    cfg_loaded = parse_cfg_file(filename)
    print(cfg_loaded)
    print(cfg == cfg_loaded)



    filename = '/home/lidong/code/DistributedFurnace/configs/default.yaml'
    cfg_loaded = parse_cfg_file(filename)
    print(cfg_loaded)