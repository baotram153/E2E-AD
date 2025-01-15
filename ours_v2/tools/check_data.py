import os
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('-t', "--town", default="town01")
args = parser.parse_args()

town_path_pattern = "/workspace/datasets/CARLA-data/data_collect_{}"   # add "addition" when needed
log_path_pattern = "/workspace/source/E2E-AD/TCP/2080/data_collect_{}_results.json"   # change server when needed

town_dir = town_path_pattern.format(args.town)
log_file = log_path_pattern.format(args.town)

routes_cur = sorted(os.listdir(town_dir))
n_routes_cur = len(routes_cur)
with open(log_file) as log:
    result_dict = json.load(log)
n_routes = result_dict['_checkpoint']['progress'][1]
route_record = result_dict['_checkpoint']['records']

for idx, route in enumerate(routes_cur):
    measurement_dir = os.path.join(town_dir, route, 'measurements')
    route_len_cur = len(os.listdir(measurement_dir))
    route_len = int(route_record[idx]['meta']['duration_game']*2)
    if (route_len_cur < route_len - 2 or route_len_cur > route_len + 2 ):
        print(f"{route}'s length doesn't match record! Index in dir: {idx}")
        print(f"route's length in record: {route_len}, current route length: {route_len_cur}")
        while True:
            idx_tmp = idx + 1
            next_route = routes_cur[idx_tmp]
            measurement_dir = os.path.join(town_dir, route, 'measurements')
            route_len_cur_tmp = len(measurement_dir)
            if (route_len_cur >= route_len-2 or route_len_cur <= route_len+2):
                print(f"{next_route}'s length match record! Index in dir: {idx_tmp}")
                exit()
    