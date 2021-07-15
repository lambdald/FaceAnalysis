'''
Author: lidong
Date: 2021-06-23 15:44:55
LastEditors: lidong
LastEditTime: 2021-07-07 09:34:01
Description: file content
'''

from typing import Iterable
import numpy as np
import cv2


def gaussian2D(shape, sigma):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]

    if not isinstance(sigma, Iterable):
        sigma = (sigma, sigma)
    s_y, s_x = sigma
    #h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h = np.exp(-(x * x) / (2 * s_x * s_x)-(y * y) / (2 * s_y * s_y))
    if h.size == 0:
        return h
    thresh = np.finfo(h.dtype).eps * h.max()
    h[h < thresh] = 0
    return h


def draw_gaussian_by_bbox(heatmap, bbox, ratio=0.5):

    x0, y0, x1, y1 = bbox
    height, width = y1 - y0, x1-x0

    # 计算一个合理的范围，使得只要峰值落在高斯核3sigma区域内，目标检测的iou最大（考虑大部分预测w、h的情况
    # 高斯区域应该从大到小衰减:高斯区域大，有助于训练heatmap（梯度好）；高斯区域小，有助于提高定位精度。
    gaussian_h = min(max(int(round(height*ratio)), 12),int(round( height)))
    gaussian_w = min(max(int(round(width*ratio)), 12),int(round( width)))

    # 3sigma->[u-3sigma, u+3sigma]
    sigma_y = gaussian_h / 6
    sigma_x = gaussian_w / 6

    gaussian = gaussian2D((gaussian_h, gaussian_w), (sigma_y, sigma_x))
    g_y = int(round(y0+(height-gaussian_h)/2))
    g_x = int(round(x0+(width-gaussian_w)/2))
    heatmap[g_y:g_y+gaussian_h, g_x:g_x+gaussian_w] = np.maximum(gaussian, heatmap[g_y:g_y+gaussian_h, g_x:g_x+gaussian_w])


if __name__ == '__main__':

    heatmap = np.ndarray((300, 300), float)
    bbox = (50, 50, 150, 200)
    for i in range(1, 11):
        draw_gaussian_by_bbox(heatmap, bbox, i/10)
        heatmap *= 255
        h_map = heatmap.astype(np.uint8)
        cv2.rectangle(h_map, bbox[:2], bbox[2:], 255)
        cv2.imwrite(f'heatmap_{i}.png', h_map)
