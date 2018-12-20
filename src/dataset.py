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
    def __init__(self, dataset_path, preprocess_func, video_length=16, image_size=64, \
                       mode="train"):
        # TODO: currently, mode only support 'train'

        root_path = dataset_path / 'preprocessed' / mode
        if not root_path.exists():
            print('>> Preprocessing ... (->{})'.format(root_path))
            root_path.mkdir(parents=True, exist_ok=True)
            try:
                preprocess_func(dataset_path, root_path, video_length, image_size)
            except Exception as e:
                shutil.rmtree(str(root_path))
                raise e
        
        # collect video folder paths
        video_list = []
        with open(root_path/"list.txt") as f:
            for line in f.readlines():
                # append [color_path, depth_path, n_frames]
                color_path, depth_path, n_frames = line.strip().split(" ")
                video_list.append([
                    root_path / color_path,
                    root_path / depth_path,
                    int(n_frames)
                ])
        
        self.dataset_path = dataset_path
        self.root_path = root_path
        self.video_list = video_list
        self.video_length = video_length
        
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
            
        # # read color video
        # placeholder = str(color_path / "{:03d}.jpg")
        # color_video = [dataio.read_img(placeholder.format(i)) for i in frames_to_read] 
        # color_video = np.stack(color_video)
        
        # read depth video
        placeholder = str(depth_path / "{:03d}.jpg")
        depth_video = [dataio.read_img(placeholder.format(i)) for i in frames_to_read] 
        depth_video = np.stack(depth_video)
        depth_video = depth_video[...,None]

        # rgbd_video = np.concatenate([color_video, depth_video], axis=3)
        # rgbd_video = rgbd_video.transpose(3,0,1,2) # change to channel first
        # rgbd_video = rgbd_video / 128.0 - 1.0      # change value range
        
        depth_video = depth_video.transpose(3,0,1,2) # change to channel first
        depth_video = depth_video / 128.0 - 1.0      # change value range
        
        # return rgbd_video
        return depth_video

def preprocess_isogd_dataset(dataset_path, save_path, length, img_size, n_jobs=-1):
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
        
    def _preprocess(color_path, depth_path, label, save_path, length):
        if not (color_path.exists() and depth_path.exists()):
          print('Sample Not found, skipped. {}'.format(color_path.parents[0]))
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
        depth_video = [imresize(img, (img_size, img_size)) for img in depth_video]
        color_video, depth_video = np.stack(color_video), np.stack(depth_video)
        depth_video = depth_video[...,0] # save as grayscale image

        # save
        name = "{}_{}_{}".format(color_path.parents[0].name, color_path.name[2:7], label)
        dataio.save_video_as_images(color_video, save_path/name/'color')
        dataio.save_video_as_images(depth_video, save_path/name/'depth')

        return [name+'/color', name+'/depth', T]

    # perform preprocess
    video_infos = Parallel(n_jobs=n_jobs, verbose=3)\
                    ([delayed(_preprocess)(color_path, depth_path, label, save_path, length) \
                      for color_path, depth_path, label in zip(color_videos, depth_videos, labels)])

    # list file of train samples
    with open(save_path / 'list.txt', 'w') as f:
        for info in video_infos:
            if info is None:
                continue
            f.write("{} {} {}\n".format(*info)) # color_video, depth_video, n_frames

if __name__=="__main__":
    dataset = VideoDataset(Path("data/isogd/"), preprocess_isogd_dataset)
    print(dataset[0].shape)
