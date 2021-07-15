'''
Author: lidong
Date: 2021-06-09 13:11:05
LastEditors: lidong
LastEditTime: 2021-07-05 13:37:36
Description: file content
'''

import importlib

def find_object_by_arch(arch_path: str):
    module_name, cls_name = arch_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name)
    return cls

def build_from_arch(arch_path: str, kwargs: dict):
    """根据字符串名字调用方法或构建对象

    Args:
        arch_path (str): [description]
        kwargs (dict): [description]

    Returns:
        [type]: [description]
    """
    cls = find_object_by_arch(arch_path)
    print(arch_path, cls)

    return cls(**kwargs)