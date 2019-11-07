import os
import re
import shutil
from pathlib import Path
from typing import List

import numpy as np
import skvideo.io
from joblib import Parallel, delayed

import dataio
import util


def detect_face(video_tensor, num_frames_to_use=6):
    """
    detect human face in a video and return average position.

    Parameters
    ----------
    video_tensor: numpy.array
        video tensor in the shape of (frame, height, width, channel)

    num_frames_to_use : int
        num frames to detect face
    """
    import face_recognition

    frames = np.linspace(
        0, len(video_tensor), num_frames_to_use, endpoint=False
    ).astype(np.int)

    locs = []
    for t in frames:
        locations = face_recognition.face_locations(video_tensor[t])
        if len(locations) != 0:
            locs.append(np.asarray(list(locations[0])))

    locs = np.asarray(locs)
    if len(locs) == 0:
        return [-1, -1, -1, -1]
    else:
        mean = locs.mean(axis=0).astype(np.int)
        return mean


def preprocess_isogd_dataset(
    dataset_path: Path,
    save_path: Path,
    mode: str,
    length: int,
    img_size: int,
    n_jobs: int = -1,
):
    """
    Preprocessing function for Chalearn LAP IsoGD Database
    http://www.cbsr.ia.ac.cn/users/jwan/database/isogd.html
    """
    # read samples in 'train'
    with open(dataset_path / f"{mode}_list.txt") as f:
        rows = f.readlines()

    # perform preprocess
    color_videos, depth_videos, labels = [], [], []
    for row in rows:
        color, depth, label = row.strip().split(" ")
        color_videos.append(dataset_path / color)
        depth_videos.append(dataset_path / depth)
        labels.append(label)

    def _preprocess(color_path, depth_path, label, save_path, length, img_size):
        if not (color_path.exists() and depth_path.exists()):
            print("Sample not found, skipped. {}".format(color_path.parents[0]))
            return

        # read color, depth frames
        color_video = dataio.read_video(color_path)
        depth_video = dataio.read_video(depth_path)
        T, H, W, C = color_video.shape

        if T < length + 1:
            return

        # crop to be a square (H, H) video,
        tr_y, tr_x, bl_y, bl_x = detect_face(color_video)
        if tr_y == -1:
            return

        center_x = (tr_x - bl_x) // 2 + bl_x
        left_x = max(center_x - (H // 2), 0)

        flow_video = util.calc_optical_flow(color_video)

        color_video = color_video[:, :, left_x : left_x + H]
        depth_video = depth_video[:, :, left_x : left_x + H]
        flow_video = flow_video[:, :, left_x : left_x + H]

        # resize
        color_video = [
            dataio.resize_img(img, (img_size, img_size)) for img in color_video
        ]
        depth_video = [
            dataio.resize_img(img, (img_size, img_size), "nearest")
            for img in depth_video
        ]
        flow_video = [
            dataio.resize_img(img, (img_size, img_size), "nearest")
            for img in flow_video
        ]
        color_video = np.stack(color_video)
        depth_video = np.stack(depth_video)
        depth_video = depth_video[..., 0]  # save as grayscale image
        flow_video = np.stack(flow_video)

        # for dataset
        name = "{}_{}_{}".format(
            color_path.parents[0].name, color_path.name[2:7], label
        )
        dataio.save_video_as_images(color_video, save_path / name / "color")
        dataio.save_video_as_images(depth_video, save_path / name / "depth")
        np.save(str(save_path / name / "optical-flow"), flow_video)

        (save_path / "color").mkdir(parents=True, exist_ok=True)
        (save_path / "depth").mkdir(parents=True, exist_ok=True)
        (save_path / "optical-flow").mkdir(parents=True, exist_ok=True)

        # for visualization
        dataio.write_video(color_video, save_path / "color" / (name + ".mp4"))
        dataio.write_video(depth_video, save_path / "depth" / (name + ".mp4"))

        flow_video = util.visualize_optical_flow(flow_video)
        dataio.write_video(flow_video, save_path / "optical-flow" / (name + ".mp4"))

        return [name, T]

    # perform preprocess with multi threads
    video_infos = Parallel(n_jobs=n_jobs, verbose=3)(
        [
            delayed(_preprocess)(
                color_path, depth_path, label, save_path, length, img_size
            )
            for color_path, depth_path, label in zip(color_videos, depth_videos, labels)
        ]
    )

    # list file of train samples
    with open(save_path / "list.txt", "w") as f:
        for info in video_infos:
            if info is None:
                continue
            f.write("{} {}\n".format(*info))  # color_video, depth_video, n_frames
