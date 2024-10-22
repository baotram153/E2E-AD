from PIL import Image, ImageDraw
import numpy as np
import json
import argparse
import pathlib
import os
import cv2 as cv

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, default="imgs")
parser.add_argument('--save_dir', type=str, default="vids")   

args = parser.parse_args()

root_dir = args.root_dir
save_dir = args.save_dir
os.makedirs(save_dir, exist_ok=True)
frames = sorted(os.listdir(root_dir))
total_frame = len(frames)

with Image.open(os.path.join(root_dir, frames[0])) as img:
    width, height = img.size

fourcc = cv.VideoWriter_fourcc(*'mp4v') 
video_path = os.path.join(save_dir, "visualize_bev.mp4")
video_bev = cv.VideoWriter(video_path, fourcc, fps=5, frameSize=(width, height))

for frame in frames:
    bev_path = os.path.join(root_dir, frame)

    with Image.open(bev_path) as img:
        cv_img = np.array(img.convert('RGB'))[:, :, [2,1,0]].copy()
        video_bev.write(cv_img)

cv.destroyAllWindows()
video_bev.release()
print(f"BEV video done! Check the video in {video_path}")