from PIL import Image, ImageDraw
import numpy as np
import json
import argparse
import pathlib
import os
import cv2 as cv

parser = argparse.ArgumentParser()
parser.add_argument('root_dir', type=str, default="/workspace/log/TCP/qualitative_res")
parser.add_argument('--model', type=str, help="Name of the model")
parser.add_argument('--route', type=str, help="Route's id")
parser.add_argument('--frame', type=str, help="Frame number")
parser.add_argument('--save', type=bool, default=False, help="Save the image to the waypoint folder")
args = parser.parse_args()

# organized
# json_pattern = "result_{model}/visualization/{route_name}/meta/{frame}.json"
# bev_pattern = "result_{model}/visualization/{route_name}/bev/{frame}.png"
# save_dir_pattern = "result_{model}/visualization/{route_name}/waypoints"

# haven't organized
json_pattern = "result_{model}/{route_name}/meta"
bev_pattern = "result_{model}/{route_name}/bev"
save_dir_pattern = "result_{model}/{route_name}/video"

n_wps = 4
root_dir = args.root_dir
model = args.model
route = args.route
frame = args.frame

save_dir = pathlib.Path(os.path.join(root_dir, save_dir_pattern.format(model=model, route_name=route)))
save_dir.mkdir(exist_ok=True)
json_dir = os.path.join(root_dir, json_pattern.format(model=model, route_name=route))
bev_dir = os.path.join(root_dir, bev_pattern.format(model=model, route_name=route))

with Image.open(os.path.join(bev_dir, "{frame}.png".format(frame="0000"))) as img:
    width, height = img.size
fourcc = cv.VideoWriter_fourcc(*'mp4v') 
video = cv.VideoWriter('video.avi', fourcc, 1, (width, height))

for i in range(len(os.listdir(bev_dir))):
    bev_path = os.path.join(bev_dir, "{frame}.png".format(frame=str(i).zfill(4)))
    json_path = os.path.join(json_dir, "{frame}.json".format(frame=str(i).zfill(4)))

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
            wp_global = pos + wp_local*10   # check later if 10 is correct
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

        video.write(img)

        cv.destroyAllWindows()
        video.release()