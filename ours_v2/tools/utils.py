import numpy as np
from TCP.config import GlobalConfig

def angle_speed_from_wps (waypoints):
	config = GlobalConfig()
	# assert (waypoints.size(0) == 1)
	# waypoints = waypoints[0].data.cpu().numpy()
	waypoints[:,1] *= -1
	num_pairs = len(waypoints) - 1
	best_norm = 1e5
	next_speed = 0
	aim = waypoints[0]
	for i in range(num_pairs):
		# magnitude of vectors, used for speed
		next_speed += np.linalg.norm(
				waypoints[i+1] - waypoints[i]) * 2.0 / num_pairs
		# norm of vector midpoints, used for steering
		norm = np.linalg.norm((waypoints[i+1] + waypoints[i]) / 2.0)
		if abs(config.aim_dist-best_norm) > abs(config.aim_dist-norm):
			aim = waypoints[i]
			best_norm = norm	
	angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90		# must be in range [-1, 1]

	return angle, next_speed