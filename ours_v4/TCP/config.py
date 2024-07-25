import os

class GlobalConfig:
	""" base architecture configurations """
	# data
	seq_len = 1 # input timesteps
	pred_len = 4 # future waypoints predicted

	resnet = 'resnet34'
	if (resnet == 'resnet34' or resnet == 'resnet18'): block_expansion = 1
	else: block_expansion = 4
	front_cam_size = (256, 900)		# h, w
	layer1_size = (64, 225)
	layer2_size = (32, 113)
	layer3_size = (16, 57)
	layer4_size = (8, 29)

	# data root
	root_dir_all = "/workspace/datasets/CARLA-data"
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
	init_wp_weight = 0.1
	final_wp_weight = 0.5
	
	rl_ckpt = "roach/log/ckpt_11833344.pth"

	img_aug = True

	# positional encoding
	hidden_dim = 256
	position_embedding = 'sine'

	# backbone
	lr_backbone = 2e-5
	backbone = 'resnet50'
	masks = False
	num_feature_levels = 4
	dilation = False


	def __init__(self, **kwargs):
		for k,v in kwargs.items():
			setattr(self, k, v)
