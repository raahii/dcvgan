from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np
import skvideo.io


def read_img(path: Union[str, Path], grayscale: bool = False) -> np.ndarray:
    """
    Read a image using opencv.

    Parameters
    ----------
    path : pathlib.Path or str
        Path object or file name to read image.

    grayscale : bool
        If true, the image is read as grayscale image.

    Returns
    -------
    img : numpy.ndarray
        Read Image (dtype: np.uint8, axis: (H, W, C), order: RGB).
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
    Write a image using opencv.

    Parameters
    ----------
    img : np.ndarray
        Image to be saved (dtype: uint8, axis: (H, W, C), order: RGB).

    path : pathlib.Path or string
        Path object or file name to be saved image.
    """

    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def resize_img(img: np.ndarray, size: Tuple, mode: str = "linear") -> np.ndarray:
    """
    Resize image using opencv.

    Parameters
    ----------
    img : np.ndarray
        Input image (dtype: uint8, axis: (H, W, C), order: RGB).

    size : Tuple
        Shape after resized (axis: (H, W)).

    mode: str
        Resize algorithm. choices:
        - "nearest"
        - "linear"
        - "area"
        - "cubic"
        - "lanczos4"

    Returns
    -------
    img : numpy.ndarray
        Resized Image.
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
    Save video frames into input path with indexed file name.

    Parameters
    ----------
    video_tensor : numpy.array
        Video to be saved (dtype: uint8, axis: (T, H, W, C), RGB order)

    path : pathlib.Path
        Path object to save video.
    """
    path.mkdir(parents=True, exist_ok=True)

    placeholder = str(path / "{:03d}.jpg")
    for i, frame in enumerate(video_tensor):
        write_img(frame, placeholder.format(i))


def read_video(path: Path) -> np.ndarray:
    """
    Read a video using scikit-video(ffmpeg).

    Parameters
    ----------
    path : pathlib.Path
        Path object to read video.

    Returns
    -------
    video : numpy.ndarray
        Read video (dtype: np.uint8, axis: (T, H, W, C), order: RGB).
    """
    videogen = skvideo.io.vreader(str(path))
    video = np.stack([frame for frame in videogen])

    return video


def write_video(video: np.ndarray, path: Path) -> None:
    """
    Save a video using scikit-video(ffmpeg).

    Parameters
    ----------
    video: numpy.ndarray
        Video to save (dtype: uint8, axis: (T, H, W, C), order: RGB).

    path : pathlib.Path
        Path object to save video
    """
    writer = skvideo.io.FFmpegWriter(str(path))

    for frame in video:
        writer.writeFrame(frame)
    writer.close()
