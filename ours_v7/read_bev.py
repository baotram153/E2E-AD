import numpy as np
import os
import cv2 as cv

np.set_printoptions(threshold=np.inf)   # print the entire matrix

# file_path = '/workspace/datasets/CARLA-bev-seg-data/data_collect_town01/routes_town01_10_18_02_47_24/bev_masks'
# file_name = '0036.npy'

file_path = '/media/ticklab/2TB-HDD/dataset/CARLA-bev-seg-data/data_collect_town01/routes_town01_10_18_04_51_19/bev_masks'
file_name = '0039.npy'
save_path = '/media/ticklab/2TB-HDD/dataset/CARLA-bev-seg-data/data_collect_town01/routes_town01_10_18_04_51_19/visualize_bev'

if not os.path.exists(save_path):
    os.mkdir(save_path)

np_arr = np.load(os.path.join(file_path, file_name))

img = (np_arr*255).astype(np.uint8)

cv.imwrite(os.path.join(save_path, "0.png"), img)

# print(f"bev mask: {np_arr}")
# print(f"no. of 1s: {np.sum(np_arr)}")
# print(f"bev shape: {np_arr.shape}")