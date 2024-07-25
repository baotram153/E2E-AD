export TORCH_DISTRIBUTED_DEBUG=INFO
export PYTHONPATH=$PYTHONPATH:/workspace/source/ours_v2
export CUDA_VISIBLE_DEVICES=0
python TCP/train.py --gpus 1 \
    --batch_size 2 \
    --logdir "/workspace/log/TCP/chkpt/ours_v2" \
    --id "debug"