export TORCH_DISTRIBUTED_DEBUG=INFO
export PYTHONPATH=$PYTHONPATH:/workspace/source/E2E-AD/ours_v6
export PYTHONPATH=$PYTHONPATH:/workspace/source/E2E-AD/ours_v6/TCP
export PYTHONPATH=$PYTHONPATH:/workspace/source/E2E-AD/ours_v6/TCP/RAFT
export PYTHONPATH=$PYTHONPATH:/workspace/source/E2E-AD/ours_v6/TCP/RAFT/core
export CUDA_VISIBLE_DEVICES=0
python TCP/train.py --gpus 1 \
    --batch_size 16 \
    --logdir "/workspace/log/TCP/chkpt/ours_v6" \
    --id "debug"

# remember to regenerate data before training
# ckpt from epoch 4