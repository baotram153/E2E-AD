from collections import deque
import numpy as np
import torch 
from torch import nn
from TCP.resnet import *
from TCP.transformer import *


class PIDController(object):
	def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
		self._K_P = K_P
		self._K_I = K_I
		self._K_D = K_D

		self._window = deque([0 for _ in range(n)], maxlen=n)
		self._max = 0.0
		self._min = 0.0
	
	def set_params (self, K_P, K_I, K_D):
		self._K_P = K_P
		self._K_I = K_I
		self._K_D = K_D

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

class LearnedPIDController(object):
	def __init__(self, K_P: torch.Tensor, K_I: torch.Tensor, K_D: torch.Tensor, n=40, batch_size=1):
		self._K_P = K_P
		self._K_I = K_I
		self._K_D = K_D
		# self.error_queue = deque([0 for _ in range(n)], maxlen=n)
		# self._window = np.array((error_queue))
		# self._window = torch.

	def set_params (self, K_P, K_I, K_D, error_window):
		self._K_P = K_P
		self._K_I = K_I
		self._K_D = K_D
		self._window = error_window

	def step(self):
		# print(self._window.shape)
		# print(f"K_P: {self._K_P}")
		# exit()
		error = self._window[:, -1]
		integral = torch.mean(self._window, axis=1)
		derivative = (self._window[:, -1] - self._window[:, -2])
		# if (window[-2] != 0) else:
		# 	integral = 0.0
		# 	derivative = 0.0
		return (self._K_P * error + self._K_I * integral + self._K_D * derivative).to(dtype=torch.float32)

class TCP(nn.Module):

	def __init__(self, config):
		super().__init__()
		self.config = config
		self.angle_err_window = torch.zeros(config.turn_n)
		self.speed_err_window = torch.zeros(config.speed_n)
		self.turn_n = config.turn_n
		self.speed_n = config.speed_n

		# self.turn_controller = LearnedPIDController(K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD)
		# self.speed_controller = LearnedPIDController(K_P=config.speed_KP, K_I=config.speed_KI, K_D=config.speed_KD)

		self.turn_controller = PIDController(K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD)
		self.speed_controller = PIDController(K_P=config.speed_KP, K_I=config.speed_KI, K_D=config.speed_KD)

		self.perception = resnet34(pretrained=True)		# out: feature embedding (1000), cnn feature (512)

		self.measurements = nn.Sequential(	
							nn.Linear(1+2+6, 256),
							nn.ReLU(inplace=True),
							nn.Linear(256, 256),
							nn.ReLU(inplace=True),
						)

		# self.join_traj = nn.Sequential(		# input: encoded state (128) + feature embedding (1000)
		# 					nn.Linear(128+1000, 512),
		# 					nn.ReLU(inplace=True),
		# 					nn.Linear(512, 512),
		# 					nn.ReLU(inplace=True),
		# 					nn.Linear(512, 256),
		# 					nn.ReLU(inplace=True),
		# 				)

		self.join_traj = nn.Sequential(		# input: encoded state (128) + feature embedding (1000)
							nn.Linear(256+1000, 512),
							nn.ReLU(inplace=True),
							nn.Linear(512, 256),
							nn.ReLU(inplace=True),
							nn.Linear(256, 256),
							nn.ReLU(inplace=True),
						)

		# self.join_ctrl = nn.Sequential(		# input: encoded state (128) + cnn feature (512)
		# 					nn.Linear(256+512, 512),
		# 					nn.ReLU(inplace=True),
		# 					nn.Linear(512, 512),
		# 					nn.ReLU(inplace=True),
		# 					nn.Linear(512, 256),
		# 					nn.ReLU(inplace=True),
		# 				)

		self.speed_branch = nn.Sequential(	# input: feature embedding
							nn.Linear(1000, 256),
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
				nn.Dropout2d(p=0.5),	# why has dropout?
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
		# 		nn.ReLU(inplace=True),
		# 		nn.Linear(256, 29*8),
		# 		nn.Softmax(1)
		# 	)

		# self.merge = nn.Sequential(
		# 		nn.Linear(512+256, 512),
		# 		nn.ReLU(inplace=True),
		# 		nn.Linear(512, 256),
		# 	)
		
		self.pos_embedding = PositionalEncoding(d_model=256)

		decoder_layer = nn.TransformerDecoderLayer(d_model=256, nhead=8, batch_first=True)
		self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

		self.increase_dim = nn.Linear(8*29, 256)

		self.feed_forward = nn.Sequential(	
			nn.Linear(256*5, 512),
			nn.ReLU(inplace=True),
			nn.Linear(512, 256),
			nn.ReLU(inplace=True),
		)

		self.PID_head = nn.Sequential (		
			nn.Linear(2*4+256+256, 256),		# n recent turn error + policy
			nn.ReLU(inplace=True),
			nn.Linear(256, 128),
			nn.ReLU(inplace=True),
			nn.Linear(128, 6)			# steering / speed K_P, K_I, K_D
		)

		# self.speed_PID_head = nn.Sequential (		
		# 	nn.Linear(256+256, 256),		# n recent steering / speed error + policy
		# 	nn.ReLU(inplace=True),
		# 	nn.Linear(256, 128),
		# 	nn.ReLU(inplace=True),
		# 	nn.Linear(128, 3)			# steering / speed K_P, K_I, K_D
		# )



	def forward(self, img, state, target_point, speed_error, angle_error):
		speed = state[0, 0] * 12
		feature_emb, cnn_feature = self.perception(img)
		outputs = {}
		outputs['pred_speed'] = self.speed_branch(feature_emb)
		measurement_feature = self.measurements(state)
		
		j_traj = self.join_traj(torch.cat([feature_emb, measurement_feature], 1))
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

		traj_hidden_state_embedded = self.pos_embedding(traj_hidden_state)
		tgt = torch.cat([traj_hidden_state_embedded, measurement_feature.unsqueeze(dim=1)], dim=1)	# (batch_size, seq, emb)
		
		_ , n_channels, height, width = cnn_feature.shape
		memory =  cnn_feature.view(-1, n_channels, height*width)
		memory_256 = self.increase_dim(memory)

		out = self.transformer_decoder(tgt, memory_256)		# output dim: tgt

		out = torch.flatten(out, 1)	# (256*5, 1)
		j_ctrl_final = self.feed_forward(out)

		outputs['pred_value_ctrl'] = self.value_branch_ctrl(j_ctrl_final)

		outputs['pred_features_ctrl'] = j_ctrl_final
		policy = self.policy_head(j_ctrl_final)
		outputs['mu_branches'] = self.dist_mu(policy)
		outputs['sigma_branches'] = self.dist_sigma(policy)

		pred_flatten = pred_wp.view(-1, 2*pred_wp.shape[1])
		PID_input = torch.cat([pred_flatten, measurement_feature, policy], dim=1)
		PID_params = self.PID_head(PID_input)
		
		# acce, steer = self.control_pid_module(speed_error, angle_error, PID_params)
		# print(PID_params, PID_params.shape)
		# exit()
		acce, steer = self.control_pid(pred_wp, speed, PID_params, evaluate=False)

		action = torch.stack([acce, steer], dim=1)
		outputs['pred_action'] = action

		return outputs

	def process_action(self, pred_action, command=None, speed=None, target_point=None):
		# action = self._get_action_beta(pred['mu_branches'].view(1,2), pred['sigma_branches'].view(1,2))
		acc, steer = pred_action[:, 0], pred_action[:, 1]
		acc, steer = acc.cpu().numpy().astype(np.float64), steer.cpu().numpy().astype(np.float64)
		throttle, brake = np.zeros_like(acc), np.zeros_like(acc)

		mask = acc >= 0.0
		mask_inversed = acc < 0.0

		throttle[mask] = acc[mask]
		brake[mask] = 0.0
		throttle[mask_inversed] = 0.0
		brake[mask_inversed] = np.abs(acc[mask_inversed])

		throttle = torch.from_numpy(np.clip(throttle, 0, 1)).to('cuda')
		steer = torch.from_numpy(np.clip(steer, -1, 1)).to('cuda')
		brake = torch.from_numpy(np.clip(brake, 0, 1)).to('cuda')

		metadata = {
			# 'speed': float(speed.cpu().numpy().astype(np.float64)),
			'steer': tuple(steer),
			'throttle': tuple(throttle),
			'brake': tuple(brake),
			# 'command': command,
			# 'target_point': tuple(target_point[0].data.cpu().numpy().astype(np.float64)),
		}
		return throttle, brake, steer, metadata

	def process_single_action(self, action, command=None, speed=None, target_point=None):
		# pred_action.shape = (1,2)
		acc, steer = action.cpu().numpy()[0].astype(np.float64), action.cpu().numpy()[1].astype(np.float64)
		throttle, brake = np.zeros_like(acc), np.zeros_like(acc)

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
			'steer': float(steer),
			'throttle': float(throttle),
			'brake': float(brake),
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
	
	# def calc_angle_speed_error(self, waypoints, cur_speed):
	# 	assert(waypoints.size(0)==1)
	# 	waypoints = waypoints[0].data.cpu().numpy()
	# 	waypoints[:,1] *= -1
	# 	num_pairs = len(waypoints) - 1
	# 	best_norm = 1e5
	# 	next_speed = 0
	# 	aim = waypoints[0]
	# 	for i in range(num_pairs):
	# 		# magnitude of vectors, used for speed
	# 		next_speed += np.linalg.norm(
	# 				waypoints[i+1] - waypoints[i]) * 2.0 / num_pairs
	# 		# norm of vector midpoints, used for steering
	# 		norm = np.linalg.norm((waypoints[i+1] + waypoints[i]) / 2.0)
	# 		if abs(self.config.aim_dist-best_norm) > abs(self.config.aim_dist-norm):
	# 			aim = waypoints[i]
	# 			best_norm = norm	
	# 	angle_err = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90		# must be in range [-1, 1]
	# 	speed_err = next_speed - cur_speed
	# 	return angle_err, speed_err

	def control_pid_module(self, speed_error, angle_error, params):
		''' Predicts vehicle control with a PID controller.
		Args:
			waypoints (tensor): output of self.plan()
			velocity (tensor): speedometer input
		'''
		if (params != None):
			turn_K_I, turn_K_P, turn_K_D = params[:, 0], params[:, 1], params[:, 2]		# (batch_size, 3)
			speed_K_I, speed_K_P, speed_K_D = params[:, 3], params[:, 4], params[:, 5]		# (batch_size, 3)
			self.turn_controller.set_params(turn_K_I, turn_K_P, turn_K_D, angle_error)
			self.speed_controller.set_params(speed_K_I, speed_K_P, speed_K_D, speed_error)
			# self.turn_controller.set_params(turn_K_I, turn_K_P, turn_K_D)
			# self.speed_controller.set_params(speed_K_I, speed_K_P, speed_K_D)

		steer = self.turn_controller.step()
		acce = self.speed_controller.step()

		return acce, steer

	def control_pid(self, waypoints, velocity, params, evaluate=True, target=None):
		''' Predicts vehicle control with a PID controller.
		Args:
			waypoints (tensor): output of self.plan()
			velocity (tensor): speedometer input
		'''
		turn_K_I, turn_K_P, turn_K_D, speed_K_I, speed_K_P, speed_K_D = params[0, :]

		self.turn_controller.set_params(turn_K_I, turn_K_P, turn_K_D)
		self.speed_controller.set_params(speed_K_I, speed_K_P, speed_K_D)

		assert(waypoints.size(0)==1)
		waypoints = waypoints[0].data.cpu().numpy()
		# target = target.squeeze().data.cpu().numpy()

		# flip y (forward is negative in our waypoints)
		waypoints[:,1] *= -1
		# target[1] *= -1

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

		# aim_last = waypoints[-1] - waypoints[-2]

		angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90		# must be in range [-1, 1]
		# angle_last = np.degrees(np.pi / 2 - np.arctan2(aim_last[1], aim_last[0])) / 90
		# angle_target = np.degrees(np.pi / 2 - np.arctan2(target[1], target[0])) / 90

		# # choice of point to aim for steering, removing outlier predictions
		# # use target point if it has a smaller angle or if error is large
		# # predicted point otherwise
		# # (reduces noise in eg. straight roads, helps with sudden turn commands)

		# use_target_to_aim = np.abs(angle_target) < np.abs(angle)
		# use_target_to_aim = use_target_to_aim or (np.abs(angle_target-angle_last) > self.config.angle_thresh and target[1] < self.config.dist_thresh)
		# if use_target_to_aim:
		# 	angle_final = angle_target
		# else:
		# 	angle_final = angle

		angle_final = angle

		steer = self.turn_controller.step(angle_final)

		speed = velocity.cpu().numpy()
		# print(speed.shape)
		# print(desired_speed.shape)
		delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
		acce = self.speed_controller.step(delta)

		if not evaluate:
			batch_size = 1
			acce = acce[None].expand(batch_size)
			steer = steer[None].expand(batch_size)
			return acce, steer

		steer = steer.cpu()
		acce = acce.cpu()
		steer = np.clip(steer, -1.0, 1.0)

		
		brake = desired_speed < self.config.brake_speed or (speed / desired_speed) > self.config.brake_ratio

		throttle = np.clip(acce, 0.0, self.config.max_throttle)
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
			# 'target': tuple(target.astype(np.float64)),
			'desired_speed': float(desired_speed.astype(np.float64)),
			'angle': float(angle.astype(np.float64)),
			# 'angle_last': float(angle_last.astype(np.float64)),
			# 'angle_target': float(angle_target.astype(np.float64)),
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