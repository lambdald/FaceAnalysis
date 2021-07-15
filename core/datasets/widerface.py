'''
Author: lidong
Date: 2021-06-04 14:33:22
LastEditors: lidong
LastEditTime: 2021-07-02 09:45:56
Description: file content
'''

import torch
from torchvision import transforms
from core.datasets.base_dataset import BaseDataset
import os
import cv2

def parse_widerface_annot_file(data_path, dets_file_name) -> dict:
    '''
      Parse the WiderFace annotation file:
        The format of txt ground truth.
        File name
        Number of bounding box
        x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose
    '''
    fid = open(dets_file_name, 'r')

    face_det_dict = {}
    n_bbox_to_read = 0
    img_name = ''
    for line in fid:
        line = line.strip()
        # print(img_name, n_bbox_to_read, line)
        if n_bbox_to_read == 0:
            img_name = line.strip()
            face_det_dict[img_name] = []
            n_bbox_to_read = -1
        elif n_bbox_to_read == -1:
            n_bbox_to_read = int(line)
            if n_bbox_to_read==0:
                n_bbox_to_read = 1
        else:
            bbox_info = [float(x) for x in line.split()]  # split on whitespace

            face_det_dict[img_name].append(bbox_info[:4] + [0])
            n_bbox_to_read -= 1

    face_det_dict = clean_dataset(data_path, face_det_dict)
    return face_det_dict


def clean_dataset(data_path, face_det_dict):
    """删除空数据

    Args:
        face_det_dict ([type]): [description]

    Returns:
        [type]: [description]
    """
    new_dict ={}
    for img_name, bboxs in face_det_dict.items():

        img_path = os.path.join(data_path, img_name)
        if not os.path.exists(img_path):
            continue
        new_bbox = []
        for box in bboxs:

            if sum(box[:4]) != 0 and box[2] != 0 and box[3] != 0:
                box[2] += box[0]
                box[3] += box[1]
                new_bbox.append(box)

        if len(new_bbox):
            new_dict[img_name] = new_bbox
    return new_dict

class WiderFaceDataset(BaseDataset):
    
    def __init__(self, data_path, data_annot, transforms=None, **kwargs) -> None:
        
        super().__init__()

        self.data_path = data_path
        self.data_annot = data_annot
        self.transforms = transforms

        self.data = parse_widerface_annot_file(data_path, data_annot)
        self.keys = list(self.data.keys())
        self.kwargs = kwargs
        
    def __getitem__(self, index) -> tuple:
        img, bboxes = self.pull_item(index)
        return (img, bboxes)

    def pull_item(self, index):
        img_name = self.keys[index]
        img_path = os.path.join(self.data_path, img_name)

        img = self.read_image(img_path)
        bboxes = self.data[img_name]

        if transforms:
            transformed = self.transforms(image=img, bboxes=bboxes)
            img = transformed['image']
            bboxes = transformed['bboxes']
        h, w = img.shape[:2]
        # bboxes = [(b[0]/w, b[1]/h, b[2]/w, b[3]/h, b[4]) for b in bboxes]
        return self.to_chw(img), bboxes
        
    def __len__(self) -> int:
        return len(self.keys)
        
class WiderFaceHeatmapDataset(WiderFaceDataset):

    def __init__(self, data_path, data_annot, transforms, **kwargs) -> None:
        super().__init__(data_path, data_annot, transforms=transforms, **kwargs)

    def __getitem__(self, index) -> tuple:
        img, bboxes = self.pull_item(index)
        return (img, bboxes)

    def draw_heatmap(self):
        pass