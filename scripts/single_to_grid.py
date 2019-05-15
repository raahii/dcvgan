import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from dataio import read_video, save_video_as_images, write_video
from utils import make_video_grid

sys.path.append("src")

parser = argparse.ArgumentParser()
parser.add_argument("inpath", type=Path)
parser.add_argument("outpath", type=Path)
args = parser.parse_args()


def main():
    rows, cols = 10, 10

    video_files = list(args.inpath.glob("*.mp4"))
    n_videos = len(video_files)
    print(f"{n_videos} videos found")

    args.outpath.mkdir(parents=True, exist_ok=True)
    for k in tqdm(range(n_videos // (rows * cols))):
        videos = []
        for i in range(rows * cols):
            f = video_files[k * rows * cols + i]
            videos.append(read_video(f))
        videos = np.stack(videos)
        videos = videos.transpose(0, 4, 1, 2, 3)
        grid_videos = make_video_grid(videos, rows, cols)
        grid_video = grid_videos[0].transpose(1, 2, 3, 0)
        save_video_as_images(grid_video, args.outpath / "{:04d}".format(k))
        write_video(grid_video, args.outpath / "{:04d}.mp4".format(k))


if __name__ == "__main__":
    main()
