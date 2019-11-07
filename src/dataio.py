from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np
import skvideo.io


def read_img(path: Union[str, Path], grayscale: bool = False) -> np.ndarray:
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


def write_img(img: np.ndarray, path: Union[str, Path]) -> None:
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


def resize_img(img: np.ndarray, size: Tuple, mode: str = "linear") -> np.ndarray:
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


def save_video_as_images(video_tensor: np.ndarray, path: Path) -> None:
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


def read_video(path: Path) -> np.ndarray:
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


def write_video(video_tensor: np.ndarray, path: Path) -> None:
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

    for frame in video_tensor:
        writer.writeFrame(frame)
    writer.close()
