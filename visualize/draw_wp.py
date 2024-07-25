from PIL import Image, ImageDraw
import numpy as np
import json
import argparse
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help="Name of the model")
parser.add_argument('--route', type=str, help="Route's id")
parser.add_argument('--frame', type=str, help="Frame number")
parser.add_argument('--save', type=bool, default=False, help="Save the image to the waypoint folder")
args = parser.parse_args()

json_pattern = "/workspace/log/TCP/qualitative_res/result_{model}/visualization/{route_name}/meta/{frame}.json"
bev_pattern = "/workspace/log/TCP/qualitative_res/result_{model}/visualization/{route_name}/bev/{frame}.png"
save_dir_pattern = "/workspace/log/TCP/qualitative_res/result_{model}/visualization/{route_name}/waypoints"

# model = "ours_v1.1"
# route = "routes_town05_long_07_18_14_05_45"
# frame = "0001"

n_wps = 4
model = args.model
route = args.route
frame = args.frame

save_path = pathlib.Path(save_dir_pattern.format(model=model, route_name=route))
save_path.mkdir(exist_ok=True)
json_path = json_pattern.format(model=model, route_name=route, frame=frame)
bev_path = bev_pattern.format(model=model, route_name=route, frame=frame)

with open(json_path) as json_file:
    dict = json.load(json_file)

with Image.open(bev_path) as img:
    height, width = img.size
    pos = np.array([height / 2., width / 2.])
    draw = ImageDraw.Draw(img)

    # draw wps
    for i in range (n_wps):
        wp_local = np.array(dict['wp_{}'.format(i+1)])
        wp_local = np.array([wp_local[0], -wp_local[1]])
        wp_global = pos + wp_local*10
        x, y = wp_global
        r = 2
        topleft = (x-r, y-r)
        bottomright = (x+r, y+r)
        draw.ellipse([topleft, bottomright], fill=(0,255,0,255))

    # draw target
    target_local = np.array(dict['target'])
    target_local = np.array([target_local[0], -target_local[1]])
    target_global = pos + target_local*10
    x, y = target_global
    r = 3
    topleft = (x-r, y-r)
    bottomright = (x+r, y+r)
    draw.ellipse([topleft, bottomright], fill=(255, 0, 0, 255))
    
    if (args.save):
        img.save(save_path / '{}.png'.format(frame))
    else:
        img.show()

# with Image.open(img_path) as img:
#     img.show()