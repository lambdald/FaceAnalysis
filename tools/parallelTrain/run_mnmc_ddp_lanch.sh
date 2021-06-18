CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=1 \
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=22222 \
    mnmc_ddp_launch.py