<!--
 * @Author: lidong
 * @Date: 2021-06-23 14:00:24
 * @LastEditors: lidong
 * @LastEditTime: 2021-06-23 14:06:19
 * @Description: file content
-->

# Face Detect

## Model Input/Output:
```python
{
    'image': InputBatchTensor
    'is_train': bool
    'feature': backbone output tensor
    'head':
    {
        'head1':
        {
            'output': OutputTensor
            'loss': Loss1
        }
        'head2':
        {
            'output': OutputTensor
            'loss': Loss2
        }
        ...
        ...
    }
    'loss':
    {
        'named_loss':
        {
            'head1': loss1
            'head2': loss2
        }
        'sum': loss
    }
    'eval': bbox and class
}
```