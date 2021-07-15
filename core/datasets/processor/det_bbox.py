'''
Author: lidong
Date: 2021-06-25 10:18:06
LastEditors: lidong
LastEditTime: 2021-07-02 13:27:28
Description: file content
'''

from torch.utils.data import Dataset
from core.datasets.base_dataset import BaseProcessor

class DetBbox(BaseProcessor):
    def __getitem__(self, index):
        return self.process(index)
    
    def process(self, index):
        image, bboxes = self.pull_item(index)
        h, w = image.shape[:2]
        bboxes = [(b[0]/w, b[1]/h, b[2]/w, b[3]/h, b[4]) for b in bboxes]
        return self.to_chw(image), bboxes

    def post_process(self, pred):
        return pred