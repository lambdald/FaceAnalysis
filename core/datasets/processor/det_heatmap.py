'''
Author: lidong
Date: 2021-06-24 17:52:35
LastEditors: lidong
LastEditTime: 2021-07-07 09:35:33
Description: file content
'''
import numpy as np
from torch.utils.data import Dataset
from core.datasets.utils import draw_gaussian_by_bbox

class DetHeatmap(Dataset):

    def __getitem__(self, index):
        return self.process(index)
    
    def process(self, index):
        image, bboxes = self.pull_item(index)[:2]
        num_classes = self.kwargs['num_classes']
        num_downsample = self.kwargs['num_downsample']
        
        
        h, w = image.shape[:2]
        h_down, w_down = h//num_downsample, w//num_downsample
        #  heatmap, offsetmap, shape(w,h)map
        heatmap = np.zeros((num_classes,h_down, w_down), np.float32)
        offsetmap = np.zeros((2, h_down, w_down), np.float32)  # (dx, dy)
        shapemap = np.zeros((2, h_down, w_down), np.float32)    # (w, h)
        for bbox in bboxes:
            n_cls = bbox[4]
            c_x, c_y = int(round(bbox[2] - bbox[0])), int(round(bbox[3] - bbox[1]))
            # heatmap
            draw_gaussian_by_bbox(heatmap[n_cls],[x/num_downsample for x in bbox[:4]], 0.2)

            # offset
            offset = (c_x%num_downsample / num_downsample, c_y%num_downsample/num_downsample)
            mask = np.zeros((h_down, w_down), np.float32)
            draw_gaussian_by_bbox(mask,[x/num_downsample for x in bbox[:4]], 0.2)
            
            offsetmap[:, mask > 0] = np.array(offset).reshape(-1, 1)
            # shape
            w_norm = c_x/w
            h_norm = c_x/h
            mask = np.zeros((h_down, w_down), np.float32)
            draw_gaussian_by_bbox(mask,[x/num_downsample for x in bbox[:4]], 0.2)
            shapemap[:, mask>0] = np.array((w_norm, h_norm)).reshape(-1, 1)
        # targets = np.concatenate([heatmap, offsetmap, shapemap], axis=1)
        return (self.to_chw(image), {'hm': heatmap, 'offset':offsetmap, 'wh': shapemap})