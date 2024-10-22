import os
import numpy as np
import torch
from plugins.lift_splat_shoot.src.tools import img_transform

class GlobalConfig:
	""" base architecture configurations """
	# evaluate options
	draw_wps = False

	# Data
	seq_len = 1 # input timesteps
	pred_len = 4 # future waypoints predicted

	# data root
	root_dir_all = "/workspace/datasets/CARLA-data/"
	data_dir_pattern = "data_collect_{}"

	train_towns = ['town01', 'town03', 'town04',  'town06', ]
	val_towns = ['town02', 'town05', 'town07', 'town10']

	'''debug'''
	# train_towns = ['town01']		# comment out town + '_addition'
	# train_towns = ['towndebug']
	# val_towns = ['town02', 'town05', 'town07', 'town10']

	train_data, val_data = [], []
	for town in train_towns:		
		train_data.append(os.path.join(root_dir_all, data_dir_pattern.format(town)))
		train_data.append(os.path.join(root_dir_all, data_dir_pattern.format(town+'_addition')))
	for town in val_towns:
		val_data.append(os.path.join(root_dir_all, data_dir_pattern.format(town+'_val')))

	ignore_sides = True 	# don't consider side cameras
	ignore_rear = True 		# don't consider rear cameras

	input_resolution = 256

	scale = 1 	# image pre-processing
	crop = 256 	# image pre-processing

	lr = 1e-4 # learning rate

	# Controller
	turn_KP = 0.75
	turn_KI = 0.75
	turn_KD = 0.3
	turn_n = 40 	# buffer size

	speed_KP = 5.0
	speed_KI = 0.5
	speed_KD = 1.0
	speed_n = 40 	# buffer size

	# LSS module
	H=900
	W=1600
	resize_lim=(0.193, 0.225)
	final_dim=(128, 352)
	bot_pct_lim=(0.0, 0.22)
	rot_lim=(-5.4, 5.4)
	rand_flip=True
	ncams=5
	max_grad_norm=5.0
	pos_weight=2.13

	lss_ckpt_path='plugins/lift_splat_shoot/runs/model90000.pt'

	xbound=[-50.0, 50.0, 0.5]
	ybound=[-50.0, 50.0, 0.5]
	zbound=[-10.0, 10.0, 20.0]
	dbound=[4.0, 45.0, 1.0]

	grid_conf = grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }

	data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'bot_pct_lim': bot_pct_lim,
                    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                    'Ncams': ncams,
	}
	outC = 1

	# rots, trans, intrins, post_rots, post_trans
	fH, fW = final_dim
	resize = max(fH/H, fW/W)
	resize_dims = (int(W*resize), int(H*resize))
	newW, newH = resize_dims
	crop_h = int((1 - np.mean(bot_pct_lim))*newH) - fH    # what if this is negative? -> padding, chấp nhận cắt phần dưới :v
	crop_w = int(max(0, newW - fW) / 2)
	crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
	flip = False
	rotate = 0
        
	# return resize, resize_dims, crop, flip, rotate

    # build intrinsic matrix
	'''
	K = [[Fx, 0,  w/2, 0],
			[0,  Fy, h/2, 0],
			[0,  0,  1,   0]]
	'''
	fov = 100
	focal = W / (2.0 * np.tan(fov * np.pi / 360.0))
	intrinsic = torch.eye(3)
	intrinsic[0, 0] = focal
	intrinsic[1, 1] = focal
	intrinsic[0, 2] = W / 2.0
	intrinsic[1, 2] = H / 2.0

    # build rotation matrix
	rotation = torch.Tensor([ [0.0, 1.0, 0.0] , [0.0, 0.0, -1.0] , [1.0, 0.0, 0.0] ])   # DriveAdapter

	# build translation matrix
	translation = torch.Tensor([0, 2, 1.5])    # DriveAdapter
    
	# rots = []
	# trans = []
	# intrins = []
	# post_rots = []
	# post_trans = []
        
	# img.save("imgs/augmented_img.png")

	max_throttle = 0.75 # upper limit on throttle signal value in dataset
	brake_speed = 0.4 	# desired speed below which brake is triggered -> lower the speed under this threshold to brake 
	brake_ratio = 1.1 	# ratio of speed to desired speed at which brake is triggered -> increase smooth when brake
	clip_delta = 0.25 	# maximum change in speed input to logitudinal controller


	aim_dist = 4.0 		# distance to search around for aim point
	angle_thresh = 0.3 	# outlier control detection angle
	dist_thresh = 10 	# target point y-distance for outlier filtering


	speed_weight = 0.05
	value_weight = 0.001
	features_weight = 0.01
	
	rl_ckpt = "roach/log/ckpt_11833344.pth"

	img_aug = True


	def __init__(self, **kwargs):
		for k,v in kwargs.items():
			setattr(self, k, v)
