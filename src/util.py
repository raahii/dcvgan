import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from PIL import Image


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


def min_max_norm(x):
    return (x - x.min()) / (x.max() - x.min())


class DebugLayer(nn.Module):
    def __init__(self):
        super(DebugLayer, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x


def init_normal(layer):
    if type(layer) in [nn.Conv2d, nn.ConvTranspose2d]:
        # print(layer)
        init.normal_(layer.weight.data, 0, 0.02)
    elif type(layer) in [nn.BatchNorm2d]:
        init.normal_(layer.weight.data, 1.0, 0.02)
        init.constant_(layer.bias.data, 0.0)
