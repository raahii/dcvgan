import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import skvideo.io
from joblib import Parallel, delayed

import context
import dataio
import util

HEIGHT = 240
WIDTH = 360


def preprocess_surreal_dataset(
    dataset_path: Path,
    save_path: Path,
    mode: str,
    length: int,
    img_size: int,
    n_jobs: int = -1,
) -> None:
    """
    Preprocessing function for SURREAL dataset.
    link: https://www.di.ens.fr/willow/research/surreal/data/
    """
    # <root>/data/cmu/
    # -------------- train/
    # -------------- val/
    # -------------- test/
    # ------------------ run0/
    # ------------------ run1/
    # ------------------ run2/
    # ---------------------- <sequenceName>/ #e.g. 01_01
    # --------------------------- <sequenceName>_c%04d.mp4
    # --------------------------- <sequenceName>_c%04d_depth.mat

    # collect all video files
    videos: Dict[str, Dict[str, Path]] = {}

    video_sets = list(sorted((dataset_path / mode).glob("run*")))
    for _set in video_sets:
        for seq_path in _set.iterdir():
            if not seq_path.is_dir() or "ung_" in seq_path.name:
                continue

            for color_video in sorted(seq_path.glob("*.mp4")):
                seq_id = color_video.stem
                _id = f"{_set.name}-{seq_id}"
                video = {
                    "color": color_video,
                    "depth": seq_path / f"{seq_id}_depth.mat",
                    "segm": seq_path / f"{seq_id}_segm.mat",
                    "info": seq_path / f"{seq_id}_info.mat",
                }

                # check all files exist
                for k, v in video.items():
                    if not v.exists():
                        print(
                            f"skipped {_id}: '{k}' is not found ({v}).", file=sys.stderr
                        )
                        continue

                videos[_id] = video
    print(f"collected {len(videos)} videos.")

    # prepare processing output directories
    save_path.mkdir(exist_ok=True)
    (save_path / "color").mkdir(exist_ok=True)
    (save_path / "depth").mkdir(exist_ok=True)
    (save_path / "segm").mkdir(exist_ok=True)

    # perform preprocess with multi threads
    video_infos = Parallel(n_jobs=n_jobs, verbose=3)(
        [
            delayed(_preprocess)(name, video, save_path, length, img_size)
            for name, video in videos.items()
        ]
    )

    # list file of train samples
    with open(save_path / "list.txt", "w") as f:
        for info in video_infos:
            if info is None:
                continue
            f.write(
                "{} {}\n".format(*info)
            )  # color_video, depth_video, segm_video, n_frames


def _preprocess(
    name: str, video: np.ndarray, save_path: Path, length: int, img_size: int
) -> Optional[List[str]]:

    # read all videos
    color_video = dataio.read_video(video["color"])  # (T, H, W, C)
    depth_video = _read_depth_mat(video["depth"])  # (T, H, W)
    segm_video = _read_segm_mat(video["segm"])  # (T, H, W)
    joints = _read_joints2d(video["info"])  # (T, N, 2)

    if len(color_video) < 16:
        print(f"length of color, depth, segm, joints are not matched. {name} skipped.")
        return None

    if (
        len(color_video) != len(depth_video)
        or len(color_video) != len(segm_video)
        or len(color_video) != len(joints)
    ):
        print(f"length of color, depth, segm, joints are not matched. {name} skipped.")
        return None

    # if output path of processed video exists, skip it.
    out_path = save_path / name
    if out_path.exists():
        return None

    # determine crop region from human bone points
    xmin, xmax = joints[0].min(), joints[0].max()
    xmid = int(xmax + xmin / 2)
    xs = xmid - HEIGHT // 2
    xs = min(max(0, xs), WIDTH - HEIGHT)
    xe = xs + HEIGHT

    # crop video and resize
    resize_to = (img_size, img_size)
    color_video = color_video[:, :, xs:xe]
    color_video = dataio.resize_video(color_video, resize_to, "cubic")
    depth_video = depth_video[:, :, xs:xe]
    depth_video = dataio.resize_video(depth_video, resize_to, "nearest")
    segm_video = segm_video[:, :, xs:xe]
    segm_video = dataio.resize_video(segm_video, resize_to, "nearest")

    # save under a temporary directory once.
    temp_path = Path(tempfile.mkdtemp())
    T, H, W = depth_video.shape

    # save color video
    dataio.save_video_as_images(color_video, temp_path / "color")

    # save depth video
    # dataio.save_video_as_images(depth_video, temp_path / "depth", grayscale=True)
    np.save(str(temp_path / "depth"), depth_video)

    # save segmentation video
    n_segm_parts = 25
    segm_video_one_hot = np.eye(n_segm_parts)[segm_video]
    np.save(str(temp_path / "segm"), segm_video_one_hot)

    shutil.move(str(temp_path), str(out_path))

    # save to look
    p = (save_path / "color" / name).with_suffix(".mp4")
    dataio.write_video(color_video, p, fps=20)

    p = (save_path / "depth" / name).with_suffix(".mp4")
    depth_video = np.stack([_convert_depth_to_image(d) for d in depth_video])
    dataio.write_video(depth_video, p, fps=20)

    p = (save_path / "segm" / name).with_suffix(".mp4")
    _segm_video = np.zeros((T, H, W, 3), dtype=np.uint8)
    for i in range(n_segm_parts):
        _segm_video[segm_video == i] = (util.segm_color(i) * 255).astype(np.uint8)
    dataio.write_video(_segm_video, p, fps=20)

    return [name, T]


def _read_depth_mat(path: Path) -> np.ndarray:
    """
    Read depth video in the SURREAL dataset.

    Parameters
    ----------
    path : pathlib.Path
        Path object of the depth mat file

    Returns
    -------
    depth_video : numpy.ndarray
        Read depth video (dtype: float32, shape: (240, 320), axis: (H, W), min: 0, max: 1e10).
    """
    data_dict = scipy.io.loadmat(str(path))
    # data_dict.keys(): dict_keys(['depth_1', 'depth_2', ...])

    depth_imgs: List[np.ndarray] = []
    i = 1
    while True:
        key = f"depth_{i}"
        if key not in data_dict:
            break

        depth_imgs.append(data_dict[key])
        i += 1

    return np.stack(depth_imgs)


def _read_segm_mat(path: Path) -> np.ndarray:
    """
    Read segmentation video in the SURREAL dataset.

    Parameters
    ----------
    path : pathlib.Path
        Path object of the segmentation mat file

    Returns
    -------
    segm_video : numpy.ndarray
        Read segmentation video (dtype: uint8, shape: (240, 320), axis: (H, W), min: 0, max: 24).
    """
    data_dict = scipy.io.loadmat(str(path))
    # data_dict.keys(): dict_keys(['segm_1', 'segm_2', ...])

    segm_imgs: List[np.ndarray] = []
    i = 1
    while True:
        key = f"segm_{i}"
        if key not in data_dict:
            break

        segm_imgs.append(data_dict[key])
        i += 1

    return np.stack(segm_imgs)


def _read_joints2d(path: Path) -> np.ndarray:
    """
    Read joints2d from info.mat in the SURREAL dataset.

    Parameters
    ----------
    path : pathlib.Path
        Path object of the video information mat file

    Returns
    -------
    joints2d : numpy.ndarray
        Read human bone points for each time frame
        (dtype: uint8, shape: (240, 320), axis: (H, W), min: 0, max: 24).
    """
    data_dict = scipy.io.loadmat(str(path))
    joints2d = data_dict["joints2D"]
    joints2d = joints2d.transpose(2, 1, 0)

    return joints2d


def depth_color(v):
    name = "hot"
    cmap = plt.get_cmap(name)
    return list(map(lambda x: int(x * 255), cmap(v)[:3]))


def _convert_depth_to_image(depth: np.ndarray):
    BACKGROUND, BACKGROUND_COLOR = 1e10, 130
    human_mask = np.where(depth < BACKGROUND)
    human_depth = depth[human_mask]

    H, W = depth.shape
    depth_img = np.ones((H, W, 3), dtype=np.uint8) * BACKGROUND_COLOR

    if len(human_depth) == 0:
        return depth_img

    ma, mi = human_depth.max(), human_depth.min()
    human_depth = (human_depth - mi) / (ma - mi)
    human_depth = (human_depth * 255).astype(np.uint8)
    human_depth = np.array([depth_color(v) for v in human_depth], dtype=np.uint8)

    depth_img[human_mask] = human_depth

    return depth_img


if __name__ == "__main__":
    preprocess_surreal_dataset(
        Path("data/raw/surreal"), Path("data/processed/surreal"), "train", -1, 64, -1
    )
