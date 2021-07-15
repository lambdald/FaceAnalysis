###
 # @Author: lidong
 # @Date: 2021-06-25 15:51:39
 # @LastEditors: lidong
 # @LastEditTime: 2021-06-25 16:03:28
 # @Description: file content
### 

project_path=$(dirname $(cd `dirname $0`; pwd))

# 第一个参数$1 存储了WiderFace数据路径
python \
${project_path}/../core/datasets/dataset/convert_widerface_coco.py \
--datadir $1 \
--subset all --outdir $1
