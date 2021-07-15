'''
Author: lidong
Date: 2021-06-25 16:39:57
LastEditors: lidong
LastEditTime: 2021-07-02 13:15:20
Description: file content
'''

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
import cv2

from torch.utils import data
from core.datasets.base_dataset import BaseDataset

class FaceDet(BaseDataset):
    '''
    Face Dataset using COCO format.
    '''

    def __init__(self, data_path, annot_path, kwargs:dict) -> None:
        super().__init__()

        self.data_path = data_path
        self.annot_path = annot_path

        self.transforms = kwargs.get('transforms', None)
        self.kwargs = kwargs


        self.coco = coco.COCO(self.annot_path)
        self.coco.createIndex()
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)

        print('Loaded {} {} samples'.format('=', self.num_samples))

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        json.dump(self.convert_eval_format(results), 
              open('{}/results.json'.format(save_dir), 'w'))

    def pull_item(self, index):

        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.data_path, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        img = self.read_image(img_path)
        bboxes = [
            [ann['bbox'][0],
             ann['bbox'][1],
             ann['bbox'][2],
             ann['bbox'][3],
             ann['category_id']] for ann in anns]   # to pascal_voc format

        if self.transforms:
            transformed = self.transforms(image=img, bboxes=bboxes)
            img = transformed['image']
            bboxes = transformed['bboxes']
        # bboxes = [(b[0]/w, b[1]/h, b[2]/w, b[3]/h, b[4]) for b in bboxes]
        bboxes = [(b[0], b[1], b[2] + b[0], b[3] + b[1], b[4]) for b in bboxes]

        return img, bboxes


    def convert_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = 1
                for dets in all_bboxes[image_id][cls_ind]:
                    bbox = dets[:4]
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = dets[4]
                    bbox_out  = list(map(self._to_float, bbox))
                    keypoints = np.concatenate([
                        np.array(dets[5:39], dtype=np.float32).reshape(-1, 2), 
                        np.ones((17, 1), dtype=np.float32)], axis=1).reshape(51).tolist()
                    keypoints  = list(map(self._to_float, keypoints))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(score)),
                        "keypoints": keypoints
                    }
                    detections.append(detection)
        return detections


    def run_eval(self, results, save_dir):
    # result_json = os.path.join(opt.save_dir, "results.json")
    # detections  = convert_eval_format(all_boxes)
    # json.dump(detections, open(result_json, "w"))
        self.save_results(results, save_dir)
        coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "keypoints")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()


if __name__ == '__main__':
    data_path = '/home/lidong/data/WiderFace/WIDER_val/images'
    annot_path = '/home/lidong/data/WiderFace/wider_face_val_annot_coco_style.json'
    facedata = FaceDet(data_path, annot_path)
    print('data count:', len(facedata))
    for i in range(len(facedata)):
        print(i)
        img, bboxes = facedata[i]
        print(img.shape)

        img = np.ascontiguousarray(img)
        for bbox in bboxes:
            bbox = [int(b) for b in bbox]
            cv2.rectangle(img, tuple(bbox[:2]), tuple(bbox[2:4]), (255,0,0))
        
        cv2.imwrite(f'/home/lidong/tmp/test_{i:02}.png', img)

        if i > 10:
            break