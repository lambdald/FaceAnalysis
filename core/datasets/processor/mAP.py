'''
Author: lidong
Date: 2021-06-23 17:24:38
LastEditors: lidong
LastEditTime: 2021-07-02 11:02:27
Description: file content
'''
import torch
import numpy as np
def box_iou(b1,b2):
    '''b1,b2均为[x1,y1,x2,y2]'''
    x1_1,y1_1,x2_1,y2_1=b1
    x1_2,y1_2,x2_2,y2_2=b2
    
    x1=max(x1_1,x1_2)
    y1=max(y1_1,y1_2)
    x2=min(x2_1,x2_2)
    y2=min(y2_1,y2_2)
    
    if x2-x1+1<=0 or y2-y1+1<=0:
        return 0
    else:
        inter=(x2-x1+1)*( y2-y1+1)
        union=(x2_1-x1_1+1)*(y2_1-y1_1+1)+(x2_2-x1_2+1)*(y2_2-y1_2+1)-inter
        iou=inter/union
        return iou

def ap(predict, target, num_classes):
    """calculate average precision

    Args:
        predict ([type]): [batch, num_class, topk, bbox+conf]
        target ([type]): [batch, bbox+class]
    """

    assert len(predict) == len(target)

    num_image = len(predict)

    for i in range(num_image):
        for i in range(num_classes):
            

