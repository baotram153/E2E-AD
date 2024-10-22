python main.py train --dataroot="/workspace/datasets/CARLA-bev-seg-data/data_collect_town01" --logdir="./runs" --gpuid=0
tensorboard --logdir="./runs" --bind_all