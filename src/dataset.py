import shutil
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

import dataio
import util
from datasets.isogd import preprocess_isogd_dataset
from datasets.mug import preprocess_mug_dataset
from datasets.surreal import preprocess_surreal_dataset

PROCESSED_PATH = Path("data/processed")


class VideoDataset(Dataset):
    def __init__(
        self,
        name,
        dataset_path,
        preprocess_func,
        video_length=16,
        image_size=64,
        number_limit=-1,
        mode="train",
    ):
        # TODO: Now only supporting mode 'train'.
        root_path = PROCESSED_PATH / name / mode
        if not root_path.exists():
            print(">> Preprocessing ... (->{})".format(root_path))
            root_path.mkdir(parents=True, exist_ok=True)
            try:
                preprocess_func(dataset_path, root_path, mode, video_length, image_size)
            except Exception as e:
                shutil.rmtree(str(root_path))
                raise e

        # collect video folder paths
        with open(root_path / "list.txt") as f:
            lines = f.readlines()

        if number_limit != -1:
            lines = lines[:number_limit]

        video_list = []
        for line in lines:
            # append [color_path, n_frames]
            video_path, n_frames = line.strip().split(" ")
            video_list.append([root_path / video_path, int(n_frames)])

        self.dataset_path = dataset_path
        self.root_path = root_path
        self.video_list = video_list
        self.video_length = video_length

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, i):
        path, n_frames = self.video_list[i]

        # if video length is longer than pre-defined max length, select subsequence randomly.
        if n_frames < self.video_length:
            raise Exception("Invalid Video Found! Video length is insufficient!")
        elif n_frames == self.video_length:
            frames_to_read = range(n_frames)
        else:
            t = np.random.randint(n_frames - self.video_length)
            frames_to_read = range(t, t + self.video_length)

        # read color video
        placeholder = str(path / "color" / "{:03d}.jpg")
        color_video = [dataio.read_img(placeholder.format(i)) for i in frames_to_read]
        color_video = np.stack(color_video)

        # read depth video
        placeholder = str(path / "depth" / "{:03d}.jpg")
        depth_video = [dataio.read_img(placeholder.format(i)) for i in frames_to_read]
        depth_video = np.stack(depth_video)
        depth_video = depth_video[..., 0:1]

        rgbd_video = np.concatenate([color_video, depth_video], axis=3)
        rgbd_video = rgbd_video.transpose(3, 0, 1, 2)  # change to channel first
        rgbd_video = rgbd_video / 128.0 - 1.0  # change value range

        return rgbd_video


if __name__ == "__main__":
    # isogd
    dataset = VideoDataset("isogd", Path("data/isogd/"), preprocess_isogd_dataset)
    print("isogd")
    print("dataset length:", len(dataset))
    print("The first video sample:", dataset[0].shape)

    # mug
    dataset = VideoDataset("mug", Path("data/mug/"), preprocess_mug_dataset)
    print("\nmug")
    print("dataset length:", len(dataset))
    print("The first video sample:", dataset[0].shape)

    # # surreal
    # dataset = VideoDataset("surreal", Path("data/surreal/"), preprocess_surreal_dataset)
    # print("\nsurreal")
    # print("dataset length:", len(dataset))
    # print("The first video sample:", dataset[0].shape)
