from PIL import Image, ImageDraw
import numpy as np
import json
import argparse
import pathlib
import os
import cv2 as cv

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, default="/workspace/log/TCP/qualitative_res")
parser.add_argument('--model', type=str, help="Name of the model")
parser.add_argument('--route', type=str, help="Route's id")
parser.add_argument('--start_frame', type=int, help="Frame number")
parser.add_argument('--end_frame', type=int, help="Frame number")   
parser.add_argument('--rgb', type=bool, default=False, help="Make a video for the front cam")
parser.add_argument('--failure', type=str, help="Type of failure made by ego vehicle")
parser.add_argument('--visualize_entire_route', type=bool, default=False)
parser.add_argument('--organized', type=bool, default=False, help="True if the image folder is put into visualization folder")

args = parser.parse_args()

# organized
if args.organized:
    json_pattern = "result_{model}/visualization/{route_name}/meta"
    bev_pattern = "result_{model}/visualization/{route_name}/bev"
    rgb_pattern = "result_{model}/visualization/{route_name}/rgb"
    save_dir_pattern = "result_{model}/video"
else:
    json_pattern = "result_{model}/{route_name}/meta"
    bev_pattern = "result_{model}/{route_name}/bev"
    rgb_pattern = "result_{model}/{route_name}/rgb"
    save_dir_pattern = "result_{model}/video"


n_wps = 4
root_dir = args.root_dir
model = args.model
route = args.route
visualize_entire_route = args.visualize_entire_route
organized = args.organized

save_dir = pathlib.Path(os.path.join(root_dir, save_dir_pattern.format(model=model, route_name=route)))
save_dir.mkdir(exist_ok=True)
json_dir = os.path.join(root_dir, json_pattern.format(model=model, route_name=route))
bev_dir = os.path.join(root_dir, bev_pattern.format(model=model, route_name=route))
total_frame = len(os.listdir(bev_dir))

if visualize_entire_route:
    end_frame = total_frame
    start_frame = 1
else:
    end_frame = args.end_frame if (args.end_frame < total_frame) else total_frame
    start_frame = args.start_frame if (args.start_frame > 1) else 1

def make_vid_frontcam():
    rgb_dir = os.path.join(root_dir, rgb_pattern.format(model=model, route_name=route))

    with Image.open(os.path.join(rgb_dir, "{frame}.png".format(frame="0002"))) as img:
        width, height = img.size

    fourcc = cv.VideoWriter_fourcc(*'mp4v') 
    video_path = os.path.join(save_dir, "{failure}_{route}_{model}_rgb.mp4".format(route=route, model=model, failure=args.failure))
    video_rgb = cv.VideoWriter(video_path, fourcc, fps=5, frameSize=(width, height))

    for frame in range(start_frame, end_frame, 1):
        rgb_path = os.path.join(rgb_dir, "{frame}.png".format(frame=str(frame).zfill(4)))

        with Image.open(rgb_path) as img:
            cv_img = np.array(img.convert('RGB'))[:, :, [2,1,0]].copy()
            video_rgb.write(cv_img)

    cv.destroyAllWindows()
    video_rgb.release()
    print(f"RGB video done! Check the video in {video_path}")

with Image.open(os.path.join(bev_dir, "{frame}.png".format(frame="0002"))) as img:
    width, height = img.size

fourcc = cv.VideoWriter_fourcc(*'mp4v') 
video_path = os.path.join(save_dir, "{failure}_{route}_{model}_bev.mp4".format(route=route, model=model, failure=args.failure))
video_bev = cv.VideoWriter(video_path, fourcc, fps=5, frameSize=(width, height))

for frame in range(start_frame, end_frame, 1):
    bev_path = os.path.join(bev_dir, "{frame}.png".format(frame=str(frame).zfill(4)))
    json_path = os.path.join(json_dir, "{frame}.json".format(frame=str(frame).zfill(4)))

    with open(json_path) as json_file:
        dict = json.load(json_file)

    with Image.open(bev_path) as img:
        # print(type(img))
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
        
        cv_img = np.array(img.convert('RGB'))[:, :, [2,1,0]].copy()
        video_bev.write(cv_img)

cv.destroyAllWindows()
video_bev.release()
print(f"BEV video done! Check the video in {video_path}")

if (args.rgb):
    make_vid_frontcam()