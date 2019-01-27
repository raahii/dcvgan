import os
import re
import shutil
from pathlib import Path

import numpy as np
import skvideo.io
from joblib import Parallel, delayed
from scipy.misc import imresize

import utils
import dataio

def preprocess_isogd_dataset(dataset_path, save_path, mode, length, img_size, n_jobs=-1):
    '''
    Preprocessing function for Chalearn LAP IsoGD Database
    http://www.cbsr.ia.ac.cn/users/jwan/database/isogd.html
    '''
    # read samples in 'train'
    with open(dataset_path/f"{mode}_list.txt") as f:
        rows = f.readlines()
    
    # perform preprocess
    color_videos, depth_videos, labels = [], [], []
    for row in rows:
        color, depth, label = row.strip().split(" ")
        color_videos.append(dataset_path/color)
        depth_videos.append(dataset_path/depth)
        labels.append(label)
        
    def _preprocess(color_path, depth_path, label, save_path, length, img_size):
        if not (color_path.exists() and depth_path.exists()):
          print('Sample not found, skipped. {}'.format(color_path.parents[0]))
          return

        # read color, depth frames
        color_video = dataio.read_video(color_path)
        depth_video = dataio.read_video(depth_path)
        T, H, W, C = color_video.shape
        
        if T < length:
            return

        # crop to be a square (H, H) video,
        tr_y, tr_x, bl_y, bl_x  = utils.detect_face(color_video)
        if tr_y == -1:
            return

        center_x = (tr_x - bl_x) // 2 + bl_x
        left_x = max(center_x - (H//2), 0)

        color_video = color_video[:, :, left_x:left_x+H]
        depth_video = depth_video[:, :, left_x:left_x+H]

        # resize
        color_video = [imresize(img, (img_size, img_size)) for img in color_video]
        depth_video = [imresize(img, (img_size, img_size), 'nearest') for img in depth_video]
        color_video, depth_video = np.stack(color_video), np.stack(depth_video)
        depth_video = depth_video[...,0] # save as grayscale image

        # save
        name = "{}_{}_{}".format(color_path.parents[0].name, color_path.name[2:7], label)
        dataio.save_video_as_images(color_video, save_path/name/'color')
        dataio.save_video_as_images(depth_video, save_path/name/'depth')
        (save_path/'color').mkdir(parents=True, exist_ok=True)
        (save_path/'depth').mkdir(parents=True, exist_ok=True)
        dataio.write_video(color_video, save_path/'color'/(name+".mp4"))
        dataio.write_video(depth_video, save_path/'depth'/(name+".mp4"))

        return [name, T]

    # perform preprocess with multi threads
    video_infos = Parallel(n_jobs=n_jobs, verbose=3)\
                    ([delayed(_preprocess)(color_path, depth_path, label, save_path, length, img_size) \
                      for color_path, depth_path, label in zip(color_videos, depth_videos, labels)])

    # list file of train samples
    with open(save_path / 'list.txt', 'w') as f:
        for info in video_infos:
            if info is None:
                continue
            f.write("{} {}\n".format(*info)) # color_video, depth_video, n_frames

