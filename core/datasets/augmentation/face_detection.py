'''
Author: lidong
Date: 2021-06-08 17:20:25
LastEditors: lidong
LastEditTime: 2021-07-02 11:05:49
Description: file content
'''

import albumentations as A
from albumentations.augmentations.functional import scale
from albumentations.augmentations.transforms import RandomResizedCrop


def build_transform(width, height, **kwargs):
    # pascal_voc, coco
    if kwargs.get('is_train', False):
        transform = A.Compose([
            # A.ShiftScaleRotate(rotate_limit=30, p=0.2),
            A.RandomResizedCrop(width=width, height=height, scale=(0.8, 1.0), ratio=(0.8, 1.2), p=0.5),
            A.Resize(width=width, height=height, always_apply=True),
            A.MotionBlur(blur_limit=3),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.1),
            A.Normalize(always_apply=True)
        ], bbox_params=A.BboxParams(format='coco', min_visibility=0.8))
    else:
        transform = A.Compose([
            A.Resize(width=width, height=height, always_apply=True),
            A.Normalize(always_apply=True)
        ], bbox_params=A.BboxParams(format='coco', min_visibility=0.8))
    return transform