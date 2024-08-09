from collections import deque
import numpy as np
import torch 
from torch import nn
from TCP.resnet import *
from TCP.transformer import *

from TCP.transformer import PositionalEncoding

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
	# def block_forward

	
		

class TCP(nn.Module):

	def __init__(self, config):
		super().__init__()
		self.config = config

		self.turn_controller = PIDController(K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n)
		self.speed_controller = PIDController(K_P=config.speed_KP, K_I=config.speed_KI, K_D=config.speed_KD, n=config.speed_n)

		self.perception = resnet34(pretrained=True)

		self.position_embedding = PositionalEncoding(d_model=256)

		self.measurements = nn.Sequential(	
							nn.Linear(1+2+6, 256),
							nn.ReLU(inplace=True),
							nn.Linear(256, 256),
							nn.ReLU(inplace=True),
						)

		self.join_traj = nn.Sequential(
							nn.Linear(256+1000, 512),
							nn.ReLU(inplace=True),
							nn.Linear(512, 512),
							nn.ReLU(inplace=True),
							nn.Linear(512, 256),
							nn.ReLU(inplace=True),
						)

		self.speed_branch = nn.Sequential(
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
				nn.Dropout2d(p=0.5),
				nn.ReLU(inplace=True),
			)

		self.dist_mu = nn.Sequential(nn.Linear(256, dim_out), nn.Softplus())
		self.dist_sigma = nn.Sequential(nn.Linear(256, dim_out), nn.Softplus())

		self.decoder_traj = nn.GRUCell(input_size=4, hidden_size=256)
		self.output_traj = nn.Linear(256, 2)

		# decoder_layer_traj = nn.TransformerDecoderLayer(d_model=256, nhead=8, batch_first=True)		# ego query as input
		# self.decoder_traj = nn.TransformerDecoder(decoder_layer_traj, num_layers=8)

		self.ctrl_head = nn.Sequential(	
			nn.Linear(256*5, 512),
			nn.ReLU(inplace=True),
			nn.Linear(512, 256),
			nn.ReLU(inplace=True),
		)
		
		self.expansion = config.block_expansion
		num_classes = 1000
		self.fc_layer1 = nn.Linear(64*self.expansion, num_classes)
		self.fc_layer2 = nn.Linear(128*self.expansion, num_classes)
		self.fc_layer3 = nn.Linear(256*self.expansion, num_classes)
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

		'''LOOK CLOSER'''
		self.config = config

		self.front_cam_size = config.front_cam_size		# (h, w)
		self.layer1_size = config.layer1_size
		self.layer2_size = config.layer2_size
		self.layer3_size = config.layer3_size
		self.layer4_size = config.layer4_size

		self.pos_embedding = PositionalEncoding(d_model=256)
		
		decoder_layer_block1 = nn.TransformerDecoderLayer(d_model=256, nhead=8, batch_first=True)
		self.transformer_decoder_block1 = nn.TransformerDecoder(decoder_layer_block1, num_layers=4)

		decoder_layer_block2 = nn.TransformerDecoderLayer(d_model=256, nhead=8, batch_first=True)
		self.transformer_decoder_block2 = nn.TransformerDecoder(decoder_layer_block2, num_layers=4)

		decoder_layer_block3 = nn.TransformerDecoderLayer(d_model=256, nhead=8, batch_first=True)
		self.transformer_decoder_block3 = nn.TransformerDecoder(decoder_layer_block3, num_layers=4)

		decoder_layer_block4 = nn.TransformerDecoderLayer(d_model=256, nhead=8, batch_first=True)
		self.transformer_decoder_block4 = nn.TransformerDecoder(decoder_layer_block4, num_layers=4)

		# self.init_traj_query = nn.Sequential(
		# 	nn.Linear(256 + 1000, 512),
		# 	nn.ReLU(inplace=True),
		# 	nn.Linear(512, 256),
		# 	nn.ReLU(inplace=True)
		# )

		self.refine_traj_query = nn.Sequential(
			nn.Linear(256 + 256 + 1000, 512),
			nn.ReLU(inplace=True),
			nn.Linear(512, 256),
			nn.ReLU(inplace=True)
		)

		self.tokenize_feature_layer1 = nn.Sequential(
			nn.Linear(self.layer1_size[0]*self.layer1_size[1], 512),
			nn.ReLU(inplace=True),
			nn.Linear(512, 256),
			nn.ReLU(inplace=True)
		)

		self.tokenize_feature_layer2 = nn.Sequential(
			nn.Linear(self.layer2_size[0]*self.layer2_size[1], 512),
			nn.ReLU(inplace=True),
			nn.Linear(512, 256),
			nn.ReLU(inplace=True)
		)

		self.tokenize_feature_layer3 = nn.Sequential(
			nn.Linear(self.layer3_size[0]*self.layer3_size[1], 512),
			nn.ReLU(inplace=True),
			nn.Linear(512, 256),
			nn.ReLU(inplace=True)
		)

		self.tokenize_feature_layer4 = nn.Sequential(
			nn.Linear(self.layer4_size[0]*self.layer4_size[1], 512),
			nn.ReLU(inplace=True),
			nn.Linear(512, 256),
			nn.ReLU(inplace=True)
		)

		self.dropout = nn.Dropout(p=0.1)	# dropout after positional embedding

	def refine_query (self, query_res, cnn_emb):
		traj_query_res = query_res[:, 1:, :]		# (bs, 4, 256)
		ego_query_res = query_res[:, 0, :].unsqueeze(dim=1)		# (bs, 1, 256)
		# print(ego_query_res.shape)
		# print(traj_query_res.shape)
		# print(cnn_emb.shape)
		# exit()

		combined_query_res = torch.cat([traj_query_res, ego_query_res.expand(-1, traj_query_res.shape[1], -1), cnn_emb.expand(-1, traj_query_res.shape[1], -1)], dim=2)		# (bs, 5, 256+256+1000)
		
		traj_query = self.refine_traj_query(combined_query_res)							# (bs, 4, 256)
		query = torch.cat([ego_query_res, traj_query], dim=1)
		return query
	
	def extract_feature_from_resnet (self, layer4_embed, cnn_feature_layer1, cnn_feature_layer2, cnn_feature_layer3):
		layer1_embed = self.avgpool(cnn_feature_layer1)
		# print(layer1_embed.shape)
		# exit()
		layer1_embed = self.fc_layer1(torch.flatten(layer1_embed, 1))

		layer2_embed = self.avgpool(cnn_feature_layer2)
		layer2_embed = self.fc_layer2(torch.flatten(layer2_embed, 1))

		layer3_embed = self.avgpool(cnn_feature_layer3)
		layer3_embed = self.fc_layer3(torch.flatten(layer3_embed, 1))
	
		feature_embeds = torch.stack([layer1_embed, layer2_embed, layer3_embed, layer4_embed], dim=1)
		return feature_embeds

	def forward(self, img, state, target_point):
		feature_embed, cnn_feature_layer1, cnn_feature_layer2, cnn_feature_layer3, cnn_feature_layer4 = self.perception(img)	# feature_embed: (1000x1), cnn_features: (4x?)
		feature_embeds = self.extract_feature_from_resnet(feature_embed, cnn_feature_layer1, cnn_feature_layer2, cnn_feature_layer3)

		outputs = {}
		outputs['pred_speed'] = self.speed_branch(feature_embed)
		measurement_feature = self.measurements(state)
		
		j_traj = self.join_traj(torch.cat([feature_embed, measurement_feature], 1))
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

		pred_init_wp = torch.stack(output_wp, dim=1)
		outputs['pred_init_wp'] = pred_init_wp

		traj_hidden_states = torch.stack(traj_hidden_state, dim=1)

		'''
		- cnn_features_block... orderedd by LookCloser module, not Resnet module -> pasted from resnet in an inversed order
		'''
		cnn_emb_block1 = feature_embeds[:, 0, :].unsqueeze(dim=1)  
		cnn_emb_block2 = feature_embeds[:, 1, :].unsqueeze(dim=1)
		cnn_emb_block3 = feature_embeds[:, 2, :].unsqueeze(dim=1)
		cnn_emb_block4 = feature_embeds[:, 3, :].unsqueeze(dim=1)				# (bs, 1, 1000)

		cnn_features_block1 = cnn_feature_layer4
		cnn_features_block2 = cnn_feature_layer3
		cnn_features_block3 = cnn_feature_layer2
		cnn_features_block4 = cnn_feature_layer1

		ego_query = measurement_feature.unsqueeze(dim=1)

		query_block1_init = torch.cat([ego_query, traj_hidden_states], dim=1)								# bs, n_tokens, embed_dim
		query_block1 = self.refine_query(query_block1_init, cnn_emb_block1)
		query_embed_block1 = query_block1 + self.pos_embedding(query_block1)
		query_embed_block1 = self.dropout(query_embed_block1)
		bs, n_channels, height, width = cnn_features_block1.shape
		cnn_features_block1 = self.tokenize_feature_layer4(cnn_features_block1.view(bs, -1, height*width))
		# print(query_embed_block1.shape)
		# print(cnn_features_block1.shape)
		# exit()
		query_res_block1 = self.transformer_decoder_block1(query_embed_block1, cnn_features_block1)	# (bs, 5, 256)
		
		query_block2 = self.refine_query(query_res_block1, cnn_emb_block2)
		query_embed_block2 = query_block2 + self.pos_embedding(query_block2)
		query_embed_block2 = self.dropout(query_embed_block2)
		bs, n_channels, height, width = cnn_features_block2.shape
		cnn_features_block2 = self.tokenize_feature_layer3(cnn_features_block2.view(bs, -1, height*width))
		query_res_block2 = self.transformer_decoder_block2(query_embed_block2, cnn_features_block2)

		query_block3 = self.refine_query(query_res_block2, cnn_emb_block3)
		query_embed_block3 = query_block3 + self.pos_embedding(query_block3)
		query_embed_block3 = self.dropout(query_embed_block3)
		bs, n_channels, height, width = cnn_features_block3.shape
		cnn_features_block3 = self.tokenize_feature_layer2(cnn_features_block3.view(bs, -1, height*width))
		query_res_block3 = self.transformer_decoder_block3(query_embed_block3, cnn_features_block3)

		query_block4 = self.refine_query(query_res_block3, cnn_emb_block4)
		query_embed_block4 = query_block4 + self.pos_embedding(query_block4)
		query_embed_block4 = self.dropout(query_embed_block4)
		bs, n_channels, height, width = cnn_features_block4.shape
		cnn_features_block4 = self.tokenize_feature_layer1(cnn_features_block4.view(bs, -1, height*width))
		query_res_block4 = self.transformer_decoder_block4(query_embed_block4, cnn_features_block4)

		query_block4_refined = self.refine_query(query_res_block4, cnn_emb_block4)

		traj_queries = query_block4_refined[:, 1:, :]
		output_wp = []
		for wp in range(self.config.pred_len):
			dx = self.output_traj(traj_queries[:, wp, :].squeeze(dim=1))
			x = dx + x
			output_wp.append(x)

		pred_final_wp = torch.stack(output_wp, dim=1)
		outputs['pred_final_wp'] = pred_final_wp

		query_flatten = torch.flatten(query_block4_refined, 1)			# (256*5, 1)
		j_ctrl_final = self.ctrl_head(query_flatten)

		outputs['pred_value_ctrl'] = self.value_branch_ctrl(j_ctrl_final)

		outputs['pred_features_ctrl'] = j_ctrl_final

		policy = self.policy_head(j_ctrl_final)
		outputs['mu_branches'] = self.dist_mu(policy)
		outputs['sigma_branches'] = self.dist_sigma(policy)

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