export TORCH_DISTRIBUTED_DEBUG=INFO
export PYTHONPATH=$PYTHONPATH:/workspace/source/E2E-AD/ours_v3
export PYTHONPATH=$PYTHONPATH:/workspace/source/E2E-AD/ours_v3/TCP/Deformable_DETR
export CUDA_VISIBLE_DEVICES=0
python TCP/train.py --gpus 1 \
    --batch_size 1 \
    --epochs 60 \
    --logdir "/workspace/log/TCP/chkpt/ours_v3" \
    --id "Deform-Transformer_Baseline"