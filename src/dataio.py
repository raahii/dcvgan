from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import scipy.io
import skvideo.io
from PIL import Image


def read_img(path: Path, grayscale: bool = False):
    """
    Read a image using opencv function

    Parameters
    ----------
    path : pathlib.Path or string
        path to image
    """
    img = cv2.imread(str(path))
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, -1)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def write_img(img: np.ndarray, path: Path):
    """
    Write a image using opencv function

    Parameters
    ----------
    img:
        image tensor in the shape of (height, width, channel)

    path : pathlib.Path or string
        path to image
    """

    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def resize_img(img: np.ndarray, size: Tuple, mode: str = "linear"):
    """
    Resize image using opencv function

    Parameters
    ----------
    img: np.ndarray
        image tensor in the shape of (height, width, channel)

    size : Tuple
        shape (height, width)

    mode: str
        resize algorithm. choices:
        - "nearest"
        - "linear"
        - "area"
        - "cubic"
        - "lanczos4"
    """
    cv_modes = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "area": cv2.INTER_AREA,
        "cubic": cv2.INTER_CUBIC,
        "lanczos4": cv2.INTER_LANCZOS4,
    }
    return cv2.resize(img, size, interpolation=cv_modes[mode])


def save_video_as_images(video_tensor, path):
    """
    Save video frames into the directory

    Parameters
    ----------
    video_tensor: numpy.array
        video tensor in the shape of (frame, height, width, channel)

    path : pathlib.Path
        path to the video
    """
    path.mkdir(parents=True, exist_ok=True)

    placeholder = str(path / "{:03d}.jpg")
    for i, frame in enumerate(video_tensor):
        write_img(frame, placeholder.format(i))


def read_video(path):
    """
    read a video

    Parameters
    ----------
    path : string or pathlib.Path
        path to the video

    Returns
    -------
    video_tensor : numpy.array
        video tensor in the shape of (frame, height, width, channel)
    """
    videogen = skvideo.io.vreader(str(path))
    video_tensor = np.stack([frame for frame in videogen])

    return video_tensor


def write_video(video_tensor, path, m=1):
    """
    save a video

    Parameters
    ----------
    video_tensor: numpy.array
        video tensor in the shape of (frame, height, width, channel)

    path : string or pathlib.Path
        path to the video
    """
    writer = skvideo.io.FFmpegWriter(str(path))
    if m > 0:
        m = int(m)
        video_tensor = np.repeat(video_tensor, m, axis=0)
    else:
        raise ValueError

    for frame in video_tensor:
        writer.writeFrame(frame)
    writer.close()


def read_depth_mat(path):
    data_dict = scipy.io.loadmat(str(path))

    i, depth_data = 1, []
    while True:
        key = "depth_{}".format(i)
        if key not in data_dict:
            break

        depth_data.append(data_dict[key])
        i += 1

    depth_data = np.stack(depth_data)
    return depth_data


def read_segm_mat(path):
    data_dict = scipy.io.loadmat(str(path))

    i, segm_data = 1, []
    while True:
        key = "segm_{}".format(i)
        if key not in data_dict:
            break

        segm_data.append(data_dict[key])
        i += 1

    segm_data = np.stack(segm_data)
    return segm_data


def read_joints_mat(path):
    data_dict = scipy.io.loadmat(str(path))
    joints2d = data_dict["joints2D"]
    joints2d = joints2d.transpose(2, 1, 0)

    return joints2d
