import imageio
import ffmpeg
import os
import shutil
from pathlib import Path
import numpy as np
import mediapy
import cv2

import os
import numpy as np
from pathlib import Path
import mediapy


def save_video(renders, video_path, fps=30):
    video_path = str(video_path)  
    mediapy.write_video(video_path, renders, fps=fps, qp=18)


def save_render_video(renders_dict, vid_root, fps=30):
    os.makedirs(vid_root, exist_ok=True)

    for view, frame_dict in renders_dict.items():
        sorted_frames = sorted(frame_dict.keys())

        renders = [
            (frame_dict[frame].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            for frame in sorted_frames
        ]

        video_path = os.path.join(vid_root, f'{view}.mp4')

        save_video(renders, video_path, fps=fps)

def save_render_gt_video(renders_dict, gt_dict, vid_root, fps=30):
    os.makedirs(vid_root, exist_ok=True)

    for view, frame_dict in renders_dict.items():
        sorted_frames = sorted(frame_dict.keys())

        renders = [
            (frame_dict[frame].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            for frame in sorted_frames
        ]

        video_path = os.path.join(vid_root, f'{view}_render.mp4')

        save_video(renders, video_path, fps=fps)


    for view, frame_dict in gt_dict.items():
        sorted_frames = sorted(frame_dict.keys())

        gt = [
            (frame_dict[frame].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            for frame in sorted_frames
        ]

        video_path = os.path.join(vid_root, f'{view}_gt.mp4')

        save_video(gt, video_path, fps=fps)