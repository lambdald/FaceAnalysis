<!--
 * @Author: lidong
 * @Date: 2021-06-25 15:34:29
 * @LastEditors: lidong
 * @LastEditTime: 2021-07-02 09:53:29
 * @Description: file content
-->

# 本文件夹实现了数据集的加载

为了减少工作量，相同的任务类型，原始数据需要转换成统一的格式。
* 目标检测：COCO
* 关键点：COCO

## WiderFace

使用脚本core/datasets/dataset/convert_widerface_coco.py将WiderFace转换成coco格式数据