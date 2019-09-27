import re
import shutil
import subprocess as sp
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

import dataio

sys.path.append("src")


frame_name_regex = re.compile(r"([0-9]+).jpg")


def frame_number(name):
    """
    Regex to sort files
    """
    match = re.search(frame_name_regex, str(name))
    video_number = match.group(1)

    return video_number


in_path = Path("data/raw/mug/")  # categorical_rgbd_flatten
out_path = Path("data/processed/mug/train")

videos = []
cvideo_path = Path("data/processed/mug/train/color")
cvideo_path.mkdir(parents=True, exist_ok=True)
dvideo_path = Path("data/processed/mug/train/depth")
dvideo_path.mkdir(parents=True, exist_ok=True)
mocogan_path = Path("data/processed/mug/for_mocogan/0")
mocogan_path.mkdir(parents=True, exist_ok=True)


def process(folder):

    id_string = folder.name
    path = out_path / id_string
    path.mkdir(parents=True, exist_ok=True)

    # process color images
    from_files = (folder / "rgb").glob("*.jpg")
    from_files = sorted(from_files, key=frame_number)
    to_path = path / "color"
    to_path.mkdir(parents=True, exist_ok=True)

    video = []
    for i, f in enumerate(from_files):
        video.append(dataio.read_img(f))
        shutil.copy(str(f), to_path / "{:03d}.jpg".format(i))
    video = np.stack(video)
    dataio.write_video(video, (cvideo_path / (id_string + ".mp4")))
    cmd = ["convert", "+append"]
    cmd.extend(from_files)
    cmd.append(mocogan_path / (id_string + ".jpg"))
    sp.call(cmd)

    n_cimgs = len(from_files)

    # process depth images
    from_files = (folder / "depth").glob("*.jpg")
    from_files = sorted(from_files, key=frame_number)
    to_path = path / "depth"
    to_path.mkdir(parents=True, exist_ok=True)

    video = []
    for i, f in enumerate(from_files):
        video.append(dataio.read_img(f))
        shutil.copy(str(f), to_path / "{:03d}.jpg".format(i))
    video = np.stack(video)
    dataio.write_video(video, (dvideo_path / (id_string + ".mp4")))
    n_dimgs = len(from_files)

    if n_cimgs != n_dimgs:
        return

    return [id_string, n_cimgs]


video_folders = []
for k in range(6):
    p = in_path / str(k)
    folders = [x for x in p.iterdir() if x.is_dir()]
    video_folders.extend(folders)

videos = Parallel(n_jobs=8, verbose=3)(
    [delayed(process)(folder) for folder in video_folders]
)

# list file of train samples
with open(out_path / "list.txt", "w") as f:
    for info in videos:
        if info is None:
            continue
        f.write("{} {}\n".format(*info))  # color_video, depth_video, n_frames
