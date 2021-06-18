<!--
 * @Author: lidong
 * @Date: 2021-06-07 20:10:14
 * @LastEditors: lidong
 * @LastEditTime: 2021-06-07 22:56:21
 * @Description: file content
-->


# 网络结构

* backbone
* head
    * head1
    * head2
    * ...

将数据输入后， 输入dict(images=x), 将结果保存到dict['feature']。

每个head都会使用feature得到一个预测结果，并且计算相应的loss。

所有的loss会加权求和，得到最终的loss。

最好能计算归一化的loss