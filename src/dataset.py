import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from torch.utils.data import Dataset

import dataio
import util
from preprocess.isogd import preprocess_isogd_dataset
from preprocess.mug import preprocess_mug_dataset
from preprocess.surreal import preprocess_surreal_dataset

PROCESSED_PATH = Path("data/processed")


class VideoDataset(Dataset):
    def __init__(
        self,
        name: str,
        dataset_path: Path,
        preprocess_func: Callable[[Path, Path, str, int, int, int], None],
        video_length: int = 16,
        image_size: int = 64,
        number_limit: int = -1,
        geometric_info: str = "depth",
        mode: str = "train",
        extension: str = "jpg",
    ):
        # TODO: Now only supporting mode 'train'.
        root_path = PROCESSED_PATH / name / mode
        if not root_path.exists():
            print(">> Preprocessing ... (->{})".format(root_path))
            root_path.mkdir(parents=True, exist_ok=True)
            try:
                preprocess_func(
                    dataset_path, root_path, mode, video_length, image_size, -1
                )
            except Exception as e:
                shutil.rmtree(str(root_path))
                raise e

        # collect video folder paths
        with open(root_path / "list.txt") as f:
            lines = f.readlines()

        if number_limit != -1:
            lines = lines[:number_limit]

        video_list: List[Tuple[Path, int]] = []
        for line in lines:
            # append [color_path, n_frames]
            video_path, n_frames = line.strip().split(" ")
            video_list.append((root_path / video_path, int(n_frames)))

        self.dataset_path = dataset_path
        self.root_path = root_path
        self.video_list = video_list
        self.video_length = video_length
        self.image_size = image_size
        self.geometric_info = geometric_info
        self.ext = extension
        self.name = name

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, i: int):
        path: Path
        n_frames: int
        path, n_frames = self.video_list[i]

        # if video length is longer than pre-defined max length, select subsequence randomly.
        if n_frames < self.video_length + 1:
            raise Exception(f"video length is insufficient: n:{n_frames}, path:{path}")
        elif n_frames == self.video_length:
            frames_to_read = range(n_frames)
        else:
            t = np.random.randint(n_frames - self.video_length)
            frames_to_read = range(t, t + self.video_length)

        # read color video
        placeholder = str(path / "color" / ("{:03d}." + self.ext))
        color_video: np.ndarray = np.stack(
            [dataio.read_img(placeholder.format(i)) for i in frames_to_read]
        )
        color_video = color_video.transpose(3, 0, 1, 2)  # change to channel first
        color_video = color_video.astype(np.float32) / 127.5 - 1.0  # change value range

        # read geometric infomation video
        if self.geometric_info == "depth":
            placeholder = str(path / self.geometric_info / ("{:03d}." + self.ext))
            geo_video: np.ndarray = np.stack(
                [
                    dataio.read_img(placeholder.format(i), grayscale=True)
                    for i in frames_to_read
                ]
            )
            geo_video = geo_video.transpose(3, 0, 1, 2)  # change to channel first
            geo_video = geo_video.astype(np.float32) / 127.5 - 1.0  # change value range
        elif self.geometric_info == "optical-flow":
            geo_video = np.load(
                str(path / (self.geometric_info + ".npy")), mmap_mode="r"
            )
            geo_video = geo_video[frames_to_read]
            geo_video = geo_video.transpose(3, 0, 1, 2)  # change to channel first
            geo_video = geo_video / float(self.image_size)
        else:
            raise NotImplementedError

        return {"color": color_video, self.geometric_info: geo_video}


def new_dataset(configs: Dict[str, Any]):
    if configs["dataset"]["name"] not in ["mug", "isogd", "surreal"]:
        raise NotImplementedError

    return VideoDataset(
        configs["dataset"]["name"],
        Path(configs["dataset"]["path"]),
        eval(f'preprocess_{configs["dataset"]["name"]}_dataset'),
        configs["video_length"],
        configs["image_size"],
        configs["dataset"]["number_limit"],
        geometric_info=configs["geometric_info"]["name"],
    )
