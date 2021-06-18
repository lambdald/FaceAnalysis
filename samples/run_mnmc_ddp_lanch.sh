###
 # @Author: lidong
 # @Date: 2021-04-30 16:58:39
 # @LastEditors: lidong
 # @LastEditTime: 2021-06-16 14:04:24
 # @Description: file content
### 
project_path=$(dirname $(cd `dirname $0`; pwd))

ulimit -n 65535

CUDA_VISIBLE_DEVICES=1,2,3,5 OMP_NUM_THREADS=1 \
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=22223 \
    ${project_path}/tools/mnmc_ddp_launch.py --config $1