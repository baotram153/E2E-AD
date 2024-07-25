export TORCH_DISTRIBUTED_DEBUG=INFO
export PYTHONPATH=$PYTHONPATH:/workspace/source/E2E-AD/ours_v1.1
export CUDA_VISIBLE_DEVICES=1
python TCP/train.py --gpus 1 \
    --batch_size 8 \
    --logdir "/workspace/log/TCP/chkpt/ours_v1/ours_v1.1" \
    --id "debug"