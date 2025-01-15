import argparse
import os
from collections import OrderedDict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.distributions import Beta


import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

from TCP.model import TCP
from TCP.data import CARLA_Data
from TCP.config import GlobalConfig


class TCP_planner(pl.LightningModule):
	def __init__(self, config, lr):
		super().__init__()
		self.lr = lr
		self.config = config
		self.model = TCP(config)
		self._load_weight()

	def _load_weight(self):
		rl_state_dict = torch.load(self.config.rl_ckpt, map_location='cpu')['policy_state_dict']
		self._load_state_dict(self.model.value_branch_traj, rl_state_dict, 'value_head')
		self._load_state_dict(self.model.value_branch_ctrl, rl_state_dict, 'value_head')
		self._load_state_dict(self.model.dist_mu, rl_state_dict, 'dist_mu')
		self._load_state_dict(self.model.dist_sigma, rl_state_dict, 'dist_sigma')

	def _load_state_dict(self, il_net, rl_state_dict, key_word):
		rl_keys = [k for k in rl_state_dict.keys() if key_word in k]
		il_keys = il_net.state_dict().keys()
		assert len(rl_keys) == len(il_net.state_dict().keys()), f'mismatch number of layers loading {key_word}'
		new_state_dict = OrderedDict()
		for k_il, k_rl in zip(il_keys, rl_keys):
			new_state_dict[k_il] = rl_state_dict[k_rl]
		il_net.load_state_dict(new_state_dict)
	
	def forward(self, batch):
		pass

	def training_step(self, batch, batch_idx):
		front_img = batch['front_img']
		speed = batch['speed'].to(dtype=torch.float32).view(-1,1) / 12.
		target_point = batch['target_point'].to(dtype=torch.float32)
		command = batch['target_command']

		speed_error = batch['speed_error']
		angle_error = batch['angle_error']

		state = torch.cat([speed, target_point, command], 1)
		value = batch['value'].view(-1,1)
		feature = batch['feature']

		gt_waypoints = batch['waypoints']

		# speed_error = torch.stack(speed_error, dim=1)
		# angle_error = torch.stack(angle_error, dim=1)

		pred = self.model(front_img, state, target_point, speed_error, angle_error)

		dist_sup = Beta(batch['action_mu'], batch['action_sigma'])
		dist_pred = Beta(pred['mu_branches'], pred['sigma_branches'])
		kl_div = torch.distributions.kl_divergence(dist_sup, dist_pred)

		action_loss_ctrl = torch.mean(kl_div[:, 0]) *0.5 + torch.mean(kl_div[:, 1]) *0.5
		speed_loss = F.l1_loss(pred['pred_speed'], speed) * self.config.speed_weight
		value_loss = (F.mse_loss(pred['pred_value_traj'], value) + F.mse_loss(pred['pred_value_ctrl'], value)) * self.config.value_weight
		feature_loss = (F.mse_loss(pred['pred_features_traj'], feature) + F.mse_loss(pred['pred_features_ctrl'], feature))* self.config.features_weight
		action = []

		for i in range(batch['action_mu'].shape[0]):	# batch)
			action.append(self.model._get_action_beta(batch['action_mu'][i].view(1,2), batch['action_sigma'][i].view(1,2)))
		action = torch.stack(action, dim=0).squeeze()
		# print(action.shape)
		# print(pred['pred_action'].shape)
		# exit()

		action_loss_wp = (F.mse_loss(pred['pred_action'], action)) * self.config.final_ctrl_weight

		# future_feature_loss = 0
		# future_action_loss = 0
		# for i in range(self.config.pred_len):
		# 	dist_sup = Beta(batch['future_action_mu'][i], batch['future_action_sigma'][i])
		# 	dist_pred = Beta(pred['future_mu'][i], pred['future_sigma'][i])
		# 	kl_div = torch.distributions.kl_divergence(dist_sup, dist_pred)
		# 	future_action_loss += torch.mean(kl_div[:, 0]) *0.5 + torch.mean(kl_div[:, 1]) *0.5
		# 	future_feature_loss += F.mse_loss(pred['future_feature'][i], batch['future_feature'][i]) * self.config.features_weight
		# future_feature_loss /= self.config.pred_len
		# future_action_loss /= self.config.pred_len

		wp_loss = F.l1_loss(pred['pred_wp'], gt_waypoints, reduction='none').mean()
		# loss = action_loss + speed_loss + value_loss + feature_loss + wp_loss+ future_feature_loss + future_action_loss
		loss = action_loss_ctrl + speed_loss + value_loss + feature_loss + wp_loss + action_loss_wp

		self.log('train_action_loss_control', action_loss_ctrl.item())
		self.log('train_wp_loss_loss', wp_loss.item())
		self.log('train_speed_loss', speed_loss.item())
		self.log('train_value_loss', value_loss.item())
		self.log('train_feature_loss', feature_loss.item())
		# self.log('train_future_feature_loss', future_feature_loss.item())
		# self.log('train_future_action_loss', future_action_loss.item())
		self.log('train_action_loss_waypoint', action_loss_wp.item())
		return loss

	def configure_optimizers(self):
		optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-7)
		lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 30, 0.5)
		return [optimizer], [lr_scheduler]

	def validation_step(self, batch, batch_idx):
		front_img = batch['front_img']
		speed = batch['speed'].to(dtype=torch.float32).view(-1,1) / 12.
		target_point = batch['target_point'].to(dtype=torch.float32)
		command = batch['target_command']
		state = torch.cat([speed, target_point, command], 1)
		value = batch['value'].view(-1,1)
		feature = batch['feature']
		gt_waypoints = batch['waypoints']

		speed_error = batch['speed_error']
		angle_error = batch['angle_error']

		# print(len(feature), len(feature[0]))
		# print(len(speed_error), len(speed_error[0]))
		# exit()
		
		# speed_error = torch.stack(speed_error, dim=1)
		# angle_error = torch.stack(angle_error, dim=1)
		# print(angle_error)
		# print(f"speed_error.shape: {len(angle_error), len(angle_error[0])}")
		# exit()

		pred = self.model(front_img, state, target_point, speed_error, angle_error)
		dist_sup = Beta(batch['action_mu'], batch['action_sigma'])
		dist_pred = Beta(pred['mu_branches'], pred['sigma_branches'])

		kl_div = torch.distributions.kl_divergence(dist_sup, dist_pred)
		action_loss = torch.mean(kl_div[:, 0]) *0.5 + torch.mean(kl_div[:, 1]) *0.5
		speed_loss = F.l1_loss(pred['pred_speed'], speed) * self.config.speed_weight
		value_loss = (F.mse_loss(pred['pred_value_traj'], value) + F.mse_loss(pred['pred_value_ctrl'], value)) * self.config.value_weight
		feature_loss = (F.mse_loss(pred['pred_features_traj'], feature) +F.mse_loss(pred['pred_features_ctrl'], feature))* self.config.features_weight
		wp_loss = F.l1_loss(pred['pred_wp'], gt_waypoints, reduction='none').mean()

		B = batch['action_mu'].shape[0]
		batch_steer_l1 = 0
		batch_brake_l1 = 0
		batch_throttle_l1 = 0
		# for i in range(B):
		# 	# throttle, steer, brake = self.model.get_action(pred['mu_branches'][i], pred['sigma_branches'][i])
		# 	throttle, brake, steer, metadata = self.model.process_action(pred['pred_action'])
		# 	batch_throttle_l1 += torch.abs(throttle-batch['action'][i][0])
		# 	batch_steer_l1 += torch.abs(steer-batch['action'][i][1])
		# 	batch_brake_l1 += torch.abs(brake-batch['action'][i][2])
		
		throttle, brake, steer, metadata = self.model.process_action(pred['pred_action'])
		batch_throttle_l1 += torch.abs(throttle-batch['action'][:, 0])
		batch_steer_l1 += torch.abs(steer-batch['action'][:, 1])
		batch_brake_l1 += torch.abs(brake-batch['action'][:, 2])

		batch_throttle_l1 = torch.sum(batch_throttle_l1)/B
		batch_steer_l1 = torch.sum(batch_steer_l1)/B
		batch_brake_l1 = torch.sum(batch_brake_l1)/B

		# future_feature_loss = 0
		# future_action_loss = 0
		# for i in range(self.config.pred_len-1):
		# 	dist_sup = Beta(batch['future_action_mu'][i], batch['future_action_sigma'][i])
		# 	dist_pred = Beta(pred['future_mu'][i], pred['future_sigma'][i])
		# 	kl_div = torch.distributions.kl_divergence(dist_sup, dist_pred)
		# 	future_action_loss += torch.mean(kl_div[:, 0]) *0.5 + torch.mean(kl_div[:, 1]) *0.5
		# 	future_feature_loss += F.mse_loss(pred['future_feature'][i], batch['future_feature'][i]) * self.config.features_weight
		# future_feature_loss /= self.config.pred_len
		# future_action_loss /= self.config.pred_len

		val_loss = wp_loss + batch_throttle_l1+5*batch_steer_l1+batch_brake_l1

		self.log("val_action_loss", action_loss.item(), sync_dist=True)
		self.log('val_speed_loss', speed_loss.item(), sync_dist=True)
		self.log('val_value_loss', value_loss.item(), sync_dist=True)
		self.log('val_feature_loss', feature_loss.item(), sync_dist=True)
		self.log('val_wp_loss_loss', wp_loss.item(), sync_dist=True)
		# self.log('val_future_feature_loss', future_feature_loss.item(), sync_dist=True)
		# self.log('val_future_action_loss', future_action_loss.item(), sync_dist=True)
		self.log('val_throttle_loss', batch_throttle_l1.item(), sync_dist=True)
		self.log('val_brake_loss', batch_brake_l1.item(), sync_dist=True)
		self.log('val_steer_loss', batch_steer_l1.item(), sync_dist=True)
		self.log('val_loss', val_loss.item(), sync_dist=True)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('--id', type=str, default='TCP', help='Unique experiment identifier.')
	parser.add_argument('--epochs', type=int, default=60, help='Number of train epochs.')
	parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
	parser.add_argument('--val_every', type=int, default=3, help='Validation frequency (epochs).')
	parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
	parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')
	parser.add_argument('--gpus', type=int, default=1, help='number of gpus')

	args = parser.parse_args()
	args.logdir = os.path.join(args.logdir, args.id)

	# Config
	config = GlobalConfig()

	# Data
	train_set = CARLA_Data(root=config.root_dir_all, data_folders=config.train_data, img_aug = config.img_aug)
	print(len(train_set))
	val_set = CARLA_Data(root=config.root_dir_all, data_folders=config.val_data,)
	print(len(val_set))

	dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8)
	dataloader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8)

	TCP_model = TCP_planner(config, args.lr)

	checkpoint_callback = ModelCheckpoint(save_weights_only=False, mode="min", monitor="val_loss", save_top_k=2, save_last=True,
											dirpath=args.logdir, filename="best_{epoch:02d}-{val_loss:.3f}")
	checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"
	trainer = pl.Trainer.from_argparse_args(args,
											default_root_dir=args.logdir,
											gpus = args.gpus,
											accelerator='ddp',
											sync_batchnorm=True,
											plugins=DDPPlugin(find_unused_parameters=False),
											profiler='simple',
											benchmark=True,
											log_every_n_steps=1,
											flush_logs_every_n_steps=5,
											callbacks=[checkpoint_callback,
														],
											check_val_every_n_epoch = args.val_every,
											max_epochs = args.epochs
											)

	trainer.fit(TCP_model, dataloader_train, dataloader_val)




		




