import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

import dataio

sys.path.append("src")


def process(video_path, root_dir):
    frame_nums = [0, 3, 6, 9, 12, 15]

    video = dataio.read_video(video_path)
    video_image = np.zeros((64, len(frame_nums) * 64, 3), dtype=np.uint8)
    for i, f in enumerate(frame_nums):
        video_image[:, i * 64 : i * 64 + 64, :] = video[f]

    outpath = root_dir / video_path.name.replace(".mp4", ".jpg")
    dataio.write_img(video_image, outpath)


def main():
    root_dir = Path(sys.argv[1])

    all_videos = root_dir.glob("*.mp4")
    Parallel(n_jobs=-1, verbose=3)(
        [delayed(process)(video_path, root_dir) for video_path in all_videos]
    )


if __name__ == "__main__":
    main()
