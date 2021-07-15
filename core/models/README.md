<!--
 * @Author: lidong
 * @Date: 2021-06-07 20:10:14
 * @LastEditors: lidong
 * @LastEditTime: 2021-07-02 15:47:33
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

最好能计算归一化的loss。

输入数据
```python
{
    'image': tensor
    'target':
    {
        'head1': label1,
        'head2': label2
        ...
    }
}

经过backbone后
```python
{
    'image': tensor
    'target':
    {
        'head1': label1,
        'head2': label2
        ...
    }
    'feature': tensor
}

经过多个head之后
```python
{
    'image': tensor
    'target':
    {
        'head1': label1,
        'head2': label2
        ...
    }
    'feature': tensor

    'head':
    {
        'head1':
        {
            'output': predict
            'loss': loss1 # target-head1
        }
        'head2':
        {
            'output': predict
            'loss': loss2 # target-head2
        }
        ...
    }
    'loss':
    {
        'sum': sum_loss,    # 总loss，用于梯度下降
        'named_loss':
        {
            'head1': loss1,
            'head2': loss2,
            ...
        }
    }
}