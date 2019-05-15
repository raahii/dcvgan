import sys
from pathlib import Path

from tqdm import tqdm

from dataio import read_video, write_video

sys.path.append("src")


def main():
    path = Path(
        "/Users/naka/study/depth_mocogan/evaluation/result/mocogan_isogd/grid_videos"
    )
    outpath = Path("/Users/naka/study/dcvgan/result/prod_isogd")
    count = 0
    for vpath in tqdm(path.glob("*.mp4")):
        video = read_video(vpath)
        for i in range(5):
            for j in range(6):
                v = video[:, 64 * i : 64 * i + 64, 64 * j : 64 * j + 64]
                write_video(v, outpath / "{:05d}.mp4".format(count))
                count += 1


if __name__ == "__main__":
    main()
