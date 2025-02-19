export TORCH_DISTRIBUTED_DEBUG=INFO
export PYTHONPATH=$PYTHONPATH:/workspace/source/E2E-AD/ours_v1
export CUDA_VISIBLE_DEVICES=0
python TCP/train.py --gpus 1 \
    --batch_size 32 \
    --logdir "/workspace/log/TCP/chkpt/ours_v1" \
    --id "debug"