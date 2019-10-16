import sys
import time
from pathlib import Path

import cv2
import numpy as np
from benchmarker import Benchmarker
from PIL import Image
from skimage import io
from torch.utils.data import DataLoader

from dataset import VideoDataset

sys.path.append("src")


class ModifiedVideoDataset(VideoDataset):
    def __init__(self, read_img_func):
        super(ModifiedVideoDataset, self).__init__(Path("data/isogd"), None)

        self.read_img = read_img_func

    def __getitem__(self, i):
        color_path, depth_path, n_frames = self.video_list[i]

        frames_to_read = range(self.video_length)

        # read depth video
        placeholder = str(depth_path / "{:03d}.jpg")
        depth_video = [self.read_img(placeholder.format(i)) for i in frames_to_read]
        depth_video = np.stack(depth_video)
        depth_video = depth_video.transpose(3, 0, 1, 2)  # change to channel first
        depth_video = depth_video / 128.0 - 1.0  # change value range

        return depth_video


def pillow_func(path):
    return np.asarray(Image.open(str(path)).convert("RGB"))


def opencv_func(path):
    return cv2.imread(str(path))[:, :, ::-1]


def skimage_func(path):
    return io.imread(str(path))[..., None]


def init_dataloader(read_img_func):
    batchsize = 10
    n_workers = 1

    return DataLoader(
        ModifiedVideoDataset(read_img_func=read_img_func),
        batch_size=batchsize,
        num_workers=n_workers,
    )


def main():
    loop = 10
    with Benchmarker(loop, width=4) as bench:

        @bench("pillow")
        def pillow_backend(bm):
            dataloader = init_dataloader(pillow_func)
            for i in bm:
                for _ in iter(dataloader):
                    pass

        @bench("opencv")
        def opencv_backend(bm):
            dataloader = init_dataloader(opencv_func)
            for i in bm:
                for _ in iter(dataloader):
                    pass

        @bench("scikit-image")
        def skimage_backend(bm):
            dataloader = init_dataloader(skimage_func)
            for i in bm:
                for _ in iter(dataloader):
                    pass


if __name__ == "__main__":
    main()
