from collections import deque
import numpy as np
import torch 
from torch import nn
from TCP.resnet import *
from TCP.transformer import *

from demo import viz
import raft
from utils.utils import InputPadder


class PIDController(object):
	def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
		self._K_P = K_P
		self._K_I = K_I
		self._K_D = K_D

		self._window = deque([0 for _ in range(n)], maxlen=n)
		self._max = 0.0
		self._min = 0.0

	def step(self, error):
		self._window.append(error)
		self._max = max(self._max, abs(error))
		self._min = -abs(self._max)

		if len(self._window) >= 2:
			integral = np.mean(self._window)
			derivative = (self._window[-1] - self._window[-2])
		else:
			integral = 0.0
			derivative = 0.0

		return self._K_P * error + self._K_I * integral + self._K_D * derivative

class TCP(nn.Module):

	def __init__(self, config):
		super().__init__()
		self.config = config

		self.turn_controller = PIDController(K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n)
		self.speed_controller = PIDController(K_P=config.speed_KP, K_I=config.speed_KI, K_D=config.speed_KD, n=config.speed_n)

		self.perception = resnet34(pretrained=True)

		self.flow_backbone = torch.nn.DataParallel(raft.RAFT(self.config))
		self.flow_backbone.load_state_dict(torch.load(self.config.model))

		self.flow_backbone = self.flow_backbone.module
		self.flow_backbone.cuda()
		self.flow_backbone.eval()

		self.flow_cnn_decoder = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, stride=2, padding=1),
                      nn.MaxPool2d(kernel_size=3,  stride=2, padding=1),
                      nn.ReLU(inplace=True),
                      nn.BatchNorm2d(num_features=16),
                      nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=2, padding=1),
                      nn.BatchNorm2d(num_features=64),
                      nn.Conv2d(in_channels=64, out_channels=512, kernel_size=3, stride=2, padding=1),
                      nn.BatchNorm2d(num_features=512),
					  nn.AdaptiveAvgPool2d((1, 1)))
		
		self.flow_mlp_decoder = nn.Sequential(nn.Linear(512, 256),
							nn.ReLU(inplace=True),
							nn.Linear(256, 256),
							nn.ReLU(inplace=True),
						)

		for param in self.flow_backbone.parameters():
			param.requires_grad = False

		self.measurements = nn.Sequential(	
							nn.Linear(1+2+6, 256),
							nn.ReLU(inplace=True),
							nn.Linear(256, 256),
							nn.ReLU(inplace=True),
						)

		self.join_traj = nn.Sequential(
							nn.Linear(256+256+1000, 512),
							nn.ReLU(inplace=True),
							nn.Linear(512, 512),
							nn.ReLU(inplace=True),
							nn.Linear(512, 256),
							nn.ReLU(inplace=True),
						)

		# self.join_ctrl = nn.Sequential(
		# 					nn.Linear(128+512, 512),
		# 					nn.ReLU(inplace=True),
		# 					nn.Linear(512, 512),
		# 					nn.ReLU(inplace=True),
		# 					nn.Linear(512, 256),
		# 					nn.ReLU(inplace=True),
		# 				)

		self.speed_branch = nn.Sequential(
							nn.Linear(1000+256, 256),
							nn.ReLU(inplace=True),
							nn.Linear(256, 256),
							nn.Dropout2d(p=0.5),
							nn.ReLU(inplace=True),
							nn.Linear(256, 1),
						)

		self.value_branch_traj = nn.Sequential(
					nn.Linear(256, 256),
					nn.ReLU(inplace=True),
					nn.Linear(256, 256),
					nn.Dropout2d(p=0.5),
					nn.ReLU(inplace=True),
					nn.Linear(256, 1),
				) 
		
		self.value_branch_ctrl = nn.Sequential(
					nn.Linear(256, 256),
					nn.ReLU(inplace=True),
					nn.Linear(256, 256),
					nn.Dropout2d(p=0.5),
					nn.ReLU(inplace=True),
					nn.Linear(256, 1),
				)
		# shared branches_neurons
		dim_out = 2

		self.policy_head = nn.Sequential(
				nn.Linear(256, 256),
				nn.ReLU(inplace=True),
				nn.Linear(256, 256),
				nn.Dropout2d(p=0.5),
				nn.ReLU(inplace=True),
			)
		
		# self.decoder_ctrl = nn.GRUCell(input_size=256+4, hidden_size=256)
		# self.output_ctrl = nn.Sequential(
		# 		nn.Linear(256, 256),
		# 		nn.ReLU(inplace=True),
		# 		nn.Linear(256, 256),
		# 		nn.ReLU(inplace=True),
		# 	)
		self.dist_mu = nn.Sequential(nn.Linear(256, dim_out), nn.Softplus())
		self.dist_sigma = nn.Sequential(nn.Linear(256, dim_out), nn.Softplus())


		self.decoder_traj = nn.GRUCell(input_size=4, hidden_size=256)
		self.output_traj = nn.Linear(256, 2)

		# self.init_att = nn.Sequential(
		# 		nn.Linear(128, 256),
		# 		nn.ReLU(inplace=True),
		# 		nn.Linear(256, 29*8),
		# 		nn.Softmax(1)
		# 	)

		# self.wp_att = nn.Sequential(
		# 		nn.Linear(256+256, 256),
			# 	nn.ReLU(inplace=True),
			# 	nn.Linear(256, 29*8),
			# 	nn.Softmax(1)
			# )

		# self.merge = nn.Sequential(
		# 		nn.Linear(512+256, 512),
		# 		nn.ReLU(inplace=True),
		# 		nn.Linear(512, 256),
		# 	)
		
		self.pos_embedding = PositionalEncoding(d_model=256)

		decoder_layer = nn.TransformerDecoderLayer(d_model=256, nhead=8, batch_first=True)
		self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

		self.increase_dim = nn.Linear(8*29, 256)

		self.decrease_dim = nn.Sequential(	
			nn.Linear(256*5, 512),
			nn.ReLU(inplace=True),
			nn.Linear(512, 256),
			nn.ReLU(inplace=True),
		)



	def forward(self, img, img_seq, state, target_point):
		# print(img_seq[0])
		img1 = img_seq[:, 0, :, :].permute(0, 3, 1, 2).float()
		img2 = img_seq[:, 1, :, :].permute(0, 3, 1, 2).float()

		padder = InputPadder(img1.shape)
		img1, img2 = padder.pad(img1, img2)

		flow_low, flow_up = self.flow_backbone(img1, img2, iters=20, test_mode=True)

		# for i in range(img.shape[0]):
		# 	print(flow_up.shape, img1.shape)
		# 	viz(img1, flow_up, i)
		# 	exit()

		flow_cnn_feature = self.flow_cnn_decoder(flow_up)
		flow_feature_flatten = torch.flatten(flow_cnn_feature, start_dim=1)
		flow_feature_emb = self.flow_mlp_decoder(flow_feature_flatten)

		feature_emb, cnn_feature = self.perception(img)

		perception_feature = torch.cat([feature_emb, flow_feature_emb], 1)
		outputs = {}
		outputs['pred_speed'] = self.speed_branch(perception_feature)
		measurement_feature = self.measurements(state)
		
		j_traj = self.join_traj(torch.cat([perception_feature, measurement_feature], 1))
		outputs['pred_value_traj'] = self.value_branch_traj(j_traj)
		outputs['pred_features_traj'] = j_traj
		z = j_traj
		output_wp = list()
		traj_hidden_state = list()

		# initial input variable to GRU
		x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).type_as(z)

		# autoregressive generation of output waypoints
		for _ in range(self.config.pred_len):
			x_in = torch.cat([x, target_point], dim=1)	# waypoint + target point -> dim = 4
			z = self.decoder_traj(x_in, z)
			traj_hidden_state.append(z)
			dx = self.output_traj(z)
			x = dx + x
			output_wp.append(x)

		pred_wp = torch.stack(output_wp, dim=1)
		outputs['pred_wp'] = pred_wp

		traj_hidden_state = torch.stack(traj_hidden_state, dim=1)

		# init_att = self.init_att(measurement_feature).view(-1, 1, 8, 29)	# original 
		# feature_emb = torch.sum(cnn_feature*init_att, dim=(2, 3))
		# j_ctrl = self.join_ctrl(torch.cat([feature_emb, measurement_feature], 1))
		# outputs['pred_value_ctrl'] = self.value_branch_ctrl(j_ctrl)
		# outputs['pred_features_ctrl'] = j_ctrl
		# policy = self.policy_head(j_ctrl)
		# outputs['mu_branches'] = self.dist_mu(policy)
		# outputs['sigma_branches'] = self.dist_sigma(policy)

		# x = j_ctrl

		traj_hidden_state_embedded = self.pos_embedding(traj_hidden_state)
		tgt = torch.cat([traj_hidden_state_embedded, measurement_feature.unsqueeze(dim=1)], dim=1)	# (batch_size, seq, emb)
		
		_ , n_channels, height, width = cnn_feature.shape
		memory =  cnn_feature.view(-1, n_channels, height*width)	# key, value
		memory_256 = self.increase_dim(memory)

		out = self.transformer_decoder(tgt, memory_256)		# output dim: tgt

		out = torch.flatten(out, 1)	# (256*5, 1)
		j_ctrl_final = self.decrease_dim(out)

		outputs['pred_value_ctrl'] = self.value_branch_ctrl(j_ctrl_final)

		outputs['pred_features_ctrl'] = j_ctrl_final
		policy = self.policy_head(j_ctrl_final)
		outputs['mu_branches'] = self.dist_mu(policy)
		outputs['sigma_branches'] = self.dist_sigma(policy)

		# x = j_ctrl
		# mu = outputs['mu_branches']
		# sigma = outputs['sigma_branches']
		# future_feature, future_mu, future_sigma = [], [], []

		# # initial hidden variable to GRU
		# h = torch.zeros(size=(x.shape[0], 256), dtype=x.dtype).type_as(x)

		# for _ in range(self.config.pred_len):
		# 	x_in = torch.cat([x, mu, sigma], dim=1)
		# 	h = self.decoder_ctrl(x_in, h)
		# 	wp_att = self.wp_att(torch.cat([h, traj_hidden_state[:, _]], 1)).view(-1, 1, 8, 29)
		# 	new_feature_emb = torch.sum(cnn_feature*wp_att, dim=(2, 3))
		# 	merged_feature = self.merge(torch.cat([h, new_feature_emb], 1))
		# 	dx = self.output_ctrl(merged_feature)
		# 	x = dx + x

		# 	policy = self.policy_head(x)
		# 	mu = self.dist_mu(policy)
		# 	sigma = self.dist_sigma(policy)
		# 	future_feature.append(x)
		# 	future_mu.append(mu)
		# 	future_sigma.append(sigma)

		# outputs['future_feature'] = future_feature
		# outputs['future_mu'] = future_mu
		# outputs['future_sigma'] = future_sigma
		return outputs

	def process_action(self, pred, command, speed, target_point):
		action = self._get_action_beta(pred['mu_branches'].view(1,2), pred['sigma_branches'].view(1,2))
		acc, steer = action.cpu().numpy()[0].astype(np.float64)
		if acc >= 0.0:
			throttle = acc
			brake = 0.0
		else:
			throttle = 0.0
			brake = np.abs(acc)

		throttle = np.clip(throttle, 0, 1)
		steer = np.clip(steer, -1, 1)
		brake = np.clip(brake, 0, 1)

		metadata = {
			'speed': float(speed.cpu().numpy().astype(np.float64)),
			'steer': float(steer),
			'throttle': float(throttle),
			'brake': float(brake),
			'command': command,
			'target_point': tuple(target_point[0].data.cpu().numpy().astype(np.float64)),
		}
		return throttle, brake, steer, metadata

	def _get_action_beta(self, alpha, beta):
		x = torch.zeros_like(alpha)
		x[:, 1] += 0.5
		mask1 = (alpha > 1) & (beta > 1)
		x[mask1] = (alpha[mask1]-1)/(alpha[mask1]+beta[mask1]-2)

		mask2 = (alpha <= 1) & (beta > 1)
		x[mask2] = 0.0

		mask3 = (alpha > 1) & (beta <= 1)
		x[mask3] = 1.0

		# mean
		mask4 = (alpha <= 1) & (beta <= 1)
		x[mask4] = alpha[mask4]/torch.clamp((alpha[mask4]+beta[mask4]), min=1e-5)

		x = x * 2 - 1

		return x

	def control_pid(self, waypoints, velocity, target):
		''' Predicts vehicle control with a PID controller.
		Args:
			waypoints (tensor): output of self.plan()
			velocity (tensor): speedometer input
		'''
		assert(waypoints.size(0)==1)
		waypoints = waypoints[0].data.cpu().numpy()
		target = target.squeeze().data.cpu().numpy()

		# flip y (forward is negative in our waypoints)
		waypoints[:,1] *= -1
		target[1] *= -1

		# iterate over vectors between predicted waypoints
		num_pairs = len(waypoints) - 1
		best_norm = 1e5
		desired_speed = 0
		aim = waypoints[0]
		for i in range(num_pairs):
			# magnitude of vectors, used for speed
			desired_speed += np.linalg.norm(
					waypoints[i+1] - waypoints[i]) * 2.0 / num_pairs

			# norm of vector midpoints, used for steering
			norm = np.linalg.norm((waypoints[i+1] + waypoints[i]) / 2.0)
			if abs(self.config.aim_dist-best_norm) > abs(self.config.aim_dist-norm):
				aim = waypoints[i]
				best_norm = norm

		aim_last = waypoints[-1] - waypoints[-2]

		angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
		angle_last = np.degrees(np.pi / 2 - np.arctan2(aim_last[1], aim_last[0])) / 90
		angle_target = np.degrees(np.pi / 2 - np.arctan2(target[1], target[0])) / 90

		# choice of point to aim for steering, removing outlier predictions
		# use target point if it has a smaller angle or if error is large
		# predicted point otherwise
		# (reduces noise in eg. straight roads, helps with sudden turn commands)
		use_target_to_aim = np.abs(angle_target) < np.abs(angle)
		use_target_to_aim = use_target_to_aim or (np.abs(angle_target-angle_last) > self.config.angle_thresh and target[1] < self.config.dist_thresh)
		if use_target_to_aim:
			angle_final = angle_target
		else:
			angle_final = angle

		steer = self.turn_controller.step(angle_final)
		steer = np.clip(steer, -1.0, 1.0)

		speed = velocity[0].data.cpu().numpy()
		brake = desired_speed < self.config.brake_speed or (speed / desired_speed) > self.config.brake_ratio

		delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
		throttle = self.speed_controller.step(delta)
		throttle = np.clip(throttle, 0.0, self.config.max_throttle)
		throttle = throttle if not brake else 0.0

		metadata = {
			'speed': float(speed.astype(np.float64)),
			'steer': float(steer),
			'throttle': float(throttle),
			'brake': float(brake),
			'wp_4': tuple(waypoints[3].astype(np.float64)),
			'wp_3': tuple(waypoints[2].astype(np.float64)),
			'wp_2': tuple(waypoints[1].astype(np.float64)),
			'wp_1': tuple(waypoints[0].astype(np.float64)),
			'aim': tuple(aim.astype(np.float64)),
			'target': tuple(target.astype(np.float64)),
			'desired_speed': float(desired_speed.astype(np.float64)),
			'angle': float(angle.astype(np.float64)),
			'angle_last': float(angle_last.astype(np.float64)),
			'angle_target': float(angle_target.astype(np.float64)),
			'angle_final': float(angle_final.astype(np.float64)),
			'delta': float(delta.astype(np.float64)),
		}

		return throttle, brake, steer, metadata


	def get_action(self, mu, sigma):
		action = self._get_action_beta(mu.view(1,2), sigma.view(1,2))
		acc, steer = action[:, 0], action[:, 1]
		if acc >= 0.0:
			throttle = acc
			brake = torch.zeros_like(acc)
		else:
			throttle = torch.zeros_like(acc)
			brake = torch.abs(acc)

		throttle = torch.clamp(throttle, 0, 1)
		steer = torch.clamp(steer, -1, 1)
		brake = torch.clamp(brake, 0, 1)

		return throttle, brake, steer