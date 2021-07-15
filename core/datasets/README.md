<!--
 * @Author: lidong
 * @Date: 2021-06-25 09:50:37
 * @LastEditors: lidong
 * @LastEditTime: 2021-06-25 09:59:51
 * @Description: file content
-->

# datasets文件夹说明

通过双重继承，完成功能的组合。

BaseDataset实现加载数据、获取原始数据。

BaseProcessor实现数据预处理，预处理的结果就是直接输入网络的数据。

## dataset文件夹说明

对于不同的数据集，需要实现不同的数据读取类，类继承自BaseDataset(torch.utils.data.Dataset)。
这里不实现__getitem__,只实现pull_item方法用于获取原始数据。

## processor文件夹

对于不同的训练策略，可能需要对原始数据做一些处理，然后再输入到网络中。
例如目标检测的输入可能是image+bbox，也可能是image+heatmap。
processor会调用pull_item。