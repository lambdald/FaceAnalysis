'''
Author: lidong
Date: 2021-06-08 17:20:25
LastEditors: lidong
LastEditTime: 2021-06-15 16:26:13
Description: file content
'''

import albumentations as A
from albumentations.augmentations.functional import scale
from albumentations.augmentations.transforms import RandomResizedCrop


def build_transform(width, height, **kwargs):

    if kwargs.get('is_train', False):
        transform = A.Compose([
            # A.ShiftScaleRotate(rotate_limit=30, p=0.5),
            A.RandomResizedCrop(width=width, height=height, scale=(0.8, 1.0), ratio=(0.8, 1.2), p=0.5),
            A.Resize(width=width, height=height, always_apply=True),
            A.MotionBlur(blur_limit=3),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.1),
            A.Normalize(always_apply=True)
        ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.7))
    else:
        transform = A.Compose([
            A.Normalize(always_apply=True)
        ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.7))
    return transform