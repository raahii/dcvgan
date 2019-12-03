from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import skvideo.io
from joblib import Parallel, delayed


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


def write_img(img: np.ndarray, path: Union[str, Path], grayscale: bool = False) -> None:
    """
    Write a image using opencv.

    Parameters
    ----------
    img : np.ndarray
        Image to be saved (dtype: uint8, axis: (H, W, C), order: RGB).

    path : pathlib.Path or string
        Path object or file name to be saved image.

    grayscale : bool
        If true, the image is read as grayscale image.
    """

    if grayscale:
        cv2.imwrite(str(path), img)
    else:
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def resize_video(video: np.ndarray, *args: List[Any]):
    """
    Resize video using opencv.

    Parameters
    ----------
    video : np.ndarray
        Input video (dtype: uint8, axis: (T, H, W, C), order: RGB).

    args : List[Any]
        Parameters for resize_img function.

    Returns
    -------
    resized_video : numpy.ndarray
        Resized video.
    """
    return np.stack([resize_img(img, *args) for img in video])


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
    resized_img : numpy.ndarray
        Resized image.
    """
    cv_modes = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "area": cv2.INTER_AREA,
        "cubic": cv2.INTER_CUBIC,
        "lanczos4": cv2.INTER_LANCZOS4,
    }
    return cv2.resize(img, size, interpolation=cv_modes[mode])


def save_video_as_images(
    video_tensor: np.ndarray, path: Path, grayscale: bool = False
) -> None:
    """
    Save video frames into input path with indexed file name.

    Parameters
    ----------
    video_tensor : numpy.array
        Video to be saved (dtype: uint8, axis: (T, H, W, C), RGB order)

    path : pathlib.Path
        Path object to save video.

    grayscale : bool
        Parameter for write_img function.
    """
    path.mkdir(parents=True, exist_ok=True)

    placeholder = str(path / "{:03d}.jpg")
    for i, frame in enumerate(video_tensor):
        write_img(frame, placeholder.format(i), grayscale)


def read_video(path: Path,) -> np.ndarray:
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


def read_videos_pararell(
    paths: List[Path], n_jobs: int = 8, verbose: int = 0
) -> np.ndarray:
    """
    Write video batches concurrently.

    Parameters
    ----------
    path : List[pathlib.Path]
        List of path objects.

    n_jobs : int
        Number of workers (thread/process).

    verbose : int
        Verbose level of joblib.Parallel.

    Returns
    -------
    videos : numpy.ndarray
        Read video (dtype: np.uint8, axis: (N, T, H, W, C), order: RGB).
    """
    videos = Parallel(n_jobs=n_jobs, verbose=verbose)(
        [delayed(read_video)(p) for p in paths]
    )

    return np.stack(videos)


def write_video(video: np.ndarray, path: Path, fps: int = 16) -> None:
    """
    Save a video using scikit-video(ffmpeg).

    Parameters
    ----------
    video: numpy.ndarray
        Video to save (dtype: uint8, axis: (T, H, W, C), order: RGB).

    path : pathlib.Path
        Path object to save video

    fps : int
        Frame rate of the output video
    """
    writer = skvideo.io.FFmpegWriter(str(path), inputdict={"-r": str(fps)})

    for frame in video:
        writer.writeFrame(frame)
    writer.close()


def write_videos_pararell(
    videos: List[np.ndarray],
    paths: List[Path],
    fps: int = 16,
    n_jobs: int = 8,
    verbose: int = 0,
) -> None:
    """
    Write video batches concurrently.

    Parameters
    ----------
    videos : np.ndarray
        Video batch to save (dtype: uint8, axis: (N, T, H, W, C), order: RGB).

    paths : List[pathlib.Path]
        List of path objects to write video.

    fps : int
        Frame rate of the output video

    n_jobs : int
        Number of workers (thread/process).

    verbose : int
        Verbose level of joblib.Parallel.
    """
    Parallel(n_jobs=n_jobs, verbose=verbose)(
        [delayed(write_video)(v, p, fps=fps) for v, p in zip(videos, paths)]
    )

    return np.stack(videos)
