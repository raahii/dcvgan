import os
import shutil
from pathlib import Path

from torch.utils.data import Dataset

import numpy as np
import skvideo.io
from joblib import Parallel, delayed
from scipy.misc import imresize

import utils
import dataio

class VideoDataset(Dataset):
    def __init__(self, dataset_path, preprocess_func, video_length=16, mode="train"):
        root_path = dataset_path / 'preprocessed' / mode
        if not root_path.exists():
            print('>> Preprocessing ... (->{})'.format(root_path))
            root_path.mkdir(parents=True, exist_ok=True)
            try:
                preprocess_func(dataset_path, root_path)
            except Exception as e:
                shutil.rmtree(str(root_path))
                raise e
        
        # collect video folder paths
        video_list = []
        with open(dataset_path/"train_list.txt") as f:
            for line in f.readlines():
                # append [color_path, depth_path, n_frames]
                color_path, depth_path, n_frames = line.strip().split(" ")
                video_list.append([
                    root_path / color_path,
                    root_path / depth_path,
                    n_frames
                ])
        
        self.dataset_path = dataset_path
        self.root_path = root_path
        self.video_list = video_list
        self.video_lengths = video_length
        
    def __len__(self):
        return len(self.video_list)
        
    def __getitem__(self, i):
        color_path, depth_path, n_frames = self.video_list[i]
        
        # if video is longer, choose subsequence
        if n_frames < self.video_length:
            raise Exception("Invalid Video Found! Video length is insufficient!")
        elif n_frames == self.video_length:
            frames_to_read = range(n_frames)
        else:
            t = np.random.randint(n_frames - self.video_length)
            frames_to_read = range(t, t+self.video_length)
            
        # read color video
        placeholder = str(color_path / "{:03d}.jpg")
        color_video = [lycon.load(placeholder.format(i)) for i in frames_to_read] 
        color_video = np.stack(video)
        
        # read depth video
        placeholder = str(depth_path / "{:03d}.jpg")
        depth_video = [lycon.load(placeholder.format(i)) for i in frames_to_read] 
        depth_video = np.stack(video)

        rgbd_video = np.concatenate([color_video, depth_video], axis=3)
        rgbd_video = rgbd_video.transpose(3,0,1,2) # change to channel first
        rgbd_video = rgbd_video / 128.0 - 1.0      # change value range
        
        return rgbd_video


def preprocess_isogd_dataset(dataset_path, save_path, n_jobs=-1):
    # read samples in 'train'
    with open(dataset_path/"train_list.txt") as f:
        rows = f.readlines()
    
    # perform preprocess
    color_videos, depth_videos, labels = [], [], []
    for row in rows:
        color, depth, label = row.strip().split(" ")
        color_videos.append(dataset_path/color)
        depth_videos.append(dataset_path/depth)
        labels.append(label)
        
    def _preprocess(color_path, depth_path, label, save_path, f):
        MINIMUM_FRAMES = 16
          
        if not (color_path.exists() and depth_path.exists()):
          print('Sample Not found, skipped. {}'.format(color_path.parents[0]))
          return

        # read color, depth frames
        color_video = dataio.read_video(color_path)
        depth_video = dataio.read_video(depth_path)
        T, H, W, C = color_video.shape
        
        if T < MINIMUM_FRAMES:
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
        color_video = [imresize(img, (64, 64)) for img in color_video]
        depth_video = [imresize(img, (64, 64)) for img in depth_video]
        color_video, depth_video = np.stack(color_video), np.stack(depth_video)

        # save
        name = "{}_{}_{}".format(color_path.parents[0].name, color_path.name[2:7], label)
        dataio.save_video_as_images(color_video, save_path/name/'color')
        dataio.save_video_as_images(depth_video, save_path/name/'depth')
        f.exec("write", "{}\t{}\t{}\n".format(name+'/color', name+'/depth', T)) 

    # list file of train samples
    f = open(save_path / 'list.txt', 'w')
    f = dataio.SerializableFileObject(f)

    # perform preprocess
    Parallel(n_jobs=n_jobs, verbose=3)\
            ([delayed(_preprocess)(color_path, depth_path, label, save_path, f) \
              for color_path, depth_path, label in zip(color_videos, depth_videos, labels)])
    
    f.exec("close")

if __name__=="__main__":
    dataset = VideoDataset(Path("data/isogd/"), preprocess_isogd_dataset)
