from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from PIL import Image
from tqdm import tqdm

from generator import ColorVideoGenerator, GeometricVideoGenerator


def current_device() -> torch.device:
    """
    return current device (gpu or cpu)
    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


def images_to_numpy(tensor):
    """
    convert pytorch tensor to numpy array

    Parameters
    ----------
    tensor: torch or torch.cuda
        pytorch images tensor

    Returns
    ---------
    imgs: numpy.array
        numpy images array
    """

    imgs = tensor.data.cpu().numpy()
    imgs = imgs.transpose(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
    imgs = np.clip(imgs, -1, 1)
    imgs = (imgs + 1) / 2 * 255
    imgs = imgs.astype("uint8")

    return imgs


def videos_to_numpy(tensor):
    """
    convert pytorch tensor to numpy array

    Parameters
    ----------
    tensor: torch or torch.cuda
        pytorch tensor in the shape of (batchsize, channel, frames, width, height)

    Returns
    ---------
    imgs: numpy.array
        numpy array in the same shape of input tensor
    """
    videos = tensor.data.cpu().numpy()
    videos = np.clip(videos, -1, 1)
    videos = (videos + 1) / 2 * 255
    videos = videos.astype("uint8")

    return videos


def make_video_grid(videos, rows, cols):
    """
    Convert multiple videos to a single rows x cols grid video.
    It must be len(videos) == rows*cols.

    Parameters
    ----------
    videos: numpy.array
        numpy array in the shape of (batchsize, channel, frames, height, width)

    rows: int
        num rows

    cols: int
        num columns

    Returns
    ----------
    grid_video: numpy.array
        numpy array in the shape of (1, channel, frames, height*rows, width*cols)
    """

    N, C, T, H, W = videos.shape
    assert N == rows * cols

    videos = videos.transpose(1, 2, 0, 3, 4)
    videos = videos.reshape(C, T, rows, cols, H, W)
    videos = videos.transpose(0, 1, 2, 4, 3, 5)
    videos = videos.reshape(C, T, rows * H, cols * W)
    if C == 1:
        videos = np.tile(videos, (3, 1, 1, 1))
    videos = videos[None]

    return videos


def calc_optical_flow(video: np.ndarray):
    flows: List[np.ndarray] = []

    for i in range(len(video) - 1):
        f1 = cv2.cvtColor(video[i], cv2.COLOR_BGR2GRAY)
        f2 = cv2.cvtColor(video[i + 1], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(f1, f2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flows.append(flow)

    return np.stack(flows)


def visualize_optical_flow(flow_video: np.ndarray):
    color_video = []
    shape = list(flow_video[0].shape)
    shape[-1] = 3
    for flow in flow_video:
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        hsv = np.zeros(shape, dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        color_video.append(bgr)

    return color_video


def min_max_norm(x):
    return (x - x.min()) / (x.max() - x.min())


class DebugLayer(nn.Module):
    def __init__(self):
        super(DebugLayer, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x


def init_weights(layer):
    if type(layer) in [nn.Conv2d, nn.ConvTranspose2d]:
        init.normal_(layer.weight.data, 0, 0.02)
        # init.orthogonal_(layer.weight.data)
    elif type(layer) in [nn.BatchNorm2d]:
        init.normal_(layer.weight.data, 1.0, 0.02)
        init.constant_(layer.bias.data, 0.0)


def generate_samples(
    ggen: GeometricVideoGenerator,
    cgen: ColorVideoGenerator,
    num: int,
    batchsize: int = 20,
    desc: str = "generating samples",
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:

    xc_batches: List[np.ndarray] = []
    xg_batches: List[np.ndarray] = []
    for s in tqdm(range(0, num, batchsize), desc=desc, disable=not verbose):
        with torch.no_grad():
            xg = ggen.sample_videos(batchsize)
            xc = cgen.forward_videos(xg)
            if ggen.geometric_info == "depth":
                xg = xg.repeat(1, 3, 1, 1, 1)
            else:
                raise NotImplementedError
        xg = videos_to_numpy(xg)
        xc = videos_to_numpy(xc)

        xg_batches.append(xg)
        xc_batches.append(xc)

    xg = np.concatenate(xg_batches)
    xg = xg[:num]
    xg = xg.transpose(0, 2, 3, 4, 1)

    xc = np.concatenate(xc_batches)
    xc = xc[:num]
    xc = xc.transpose(0, 2, 3, 4, 1)

    return xg, xc
