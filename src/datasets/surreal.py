import re
import shutil
from pathlib import Path

import numpy as np
import skvideo.io
from joblib import Parallel, delayed
from scipy.misc import imresize

import dataio
import utils

segm_part_colors = np.asarray(
    [
        [0.4500, 0.5470, 0.6410],
        [0.8500, 0.3250, 0.0980],
        [0.9290, 0.6940, 0.1250],
        [0.4940, 0.1840, 0.3560],
        [0.4660, 0.6740, 0.1880],
        [0.3010, 0.7450, 0.9330],
        [0.5142, 0.7695, 0.7258],
        [0.9300, 0.8644, 0.4048],
        [0.6929, 0.6784, 0.7951],
        [0.6154, 0.7668, 0.4158],
        [0.4668, 0.6455, 0.7695],
        [0.9227, 0.6565, 0.3574],
        [0.6528, 0.8096, 0.3829],
        [0.6856, 0.4668, 0.6893],
        [0.7914, 0.7914, 0.7914],
        [0.7440, 0.8571, 0.7185],
        [0.9191, 0.7476, 0.8352],
        [0.9300, 0.9300, 0.6528],
        [0.3686, 0.3098, 0.6353],
        [0.6196, 0.0039, 0.2588],
        [0.9539, 0.8295, 0.6562],
        [0.9955, 0.8227, 0.4828],
        [0.1974, 0.5129, 0.7403],
        [0.5978, 0.8408, 0.6445],
        [0.8877, 0.6154, 0.5391],
        [0.6206, 0.2239, 0.3094],
    ]
)
# segm_part_colors = (segm_part_colors * 255).astype(np.uint8)


def make_segmentation_video(video):
    T, H, W = video.shape
    N = len(segm_part_colors)
    segm_video = np.zeros((T, H, W, 3), dtype=np.uint8)
    for i in range(N):
        segm_video[video == i] = segm_part_colors[i]

    return segm_video


def preprocess_surreal_dataset(
    dataset_path, save_path, mode, length, img_size, n_jobs=-1
):
    """
    Preprocessing function for SURREAL Dataset
    https://www.di.ens.fr/willow/research/surreal/data/
    """

    frame_name_regex = re.compile(r"([0-9]+)_([0-9]+)_[a-z]([0-9]+).*")

    def frame_number(name):
        """
        Regex to sort files
        """
        match = re.search(frame_name_regex, str(name))
        video_number = match.group(1) + match.group(2) + match.group(3)

        return video_number

    # collect all video files
    video_paths, depth_paths = [], []
    segm_paths, info_paths = [], []

    video_folders = list((dataset_path / mode).glob("run*"))
    for folder in reversed(video_folders):
        _video_paths, _depth_paths = [], []
        _segm_paths, _info_paths = [], []

        for p in folder.iterdir():
            if not p.is_dir():
                continue

            if "ung_" in p.name:
                continue

            _video_paths.extend(list(p.glob("*.mp4")))
            _depth_paths.extend(list(p.glob("*_depth.mat")))
            _segm_paths.extend(list(p.glob("*_segm.mat")))
            _info_paths.extend(list(p.glob("*_info.mat")))

        # sort by name
        video_paths.extend(sorted(_video_paths, key=frame_number))
        depth_paths.extend(sorted(_depth_paths, key=frame_number))
        segm_paths.extend(sorted(_segm_paths, key=frame_number))
        info_paths.extend(sorted(_info_paths, key=frame_number))

    def _preprocess(paths, save_path, length, img_size, segm_part_colors):
        try:
            video_path, depth_path, segm_path, info_path = paths
            for p in paths:
                if not p.exists():
                    print("Sample Not found, skipped. {}".format(p.parents[0]))
                    return

            # read all videos
            color_video = dataio.read_video(video_path)
            depth_video = dataio.read_depth_mat(depth_path)
            segm_video = dataio.read_segm_mat(segm_path)
            joints = dataio.read_joints_mat(info_path)  # (n_frames, n_points, 2)

            if len(color_video) < 16:
                print("skipped: ", video_path)
                return

            if len(color_video) != len(depth_video) or len(color_video) != len(
                segm_video
            ):
                print("skipped: ", video_path)
                return

            # output path
            id_string = video_path.name[0:-4]  # without extension
            name = "{}_{}".format(video_path.parents[1].name, id_string)
            out_path = save_path / name
            out_path.mkdir(parents=True, exist_ok=True)

            # compute mean center points of human bbox from joints to crop video frames
            # TODO: make pose video
            center_points = []
            for f_joints in joints:
                bottom_left = np.amin(f_joints, axis=0)
                top_right = np.amax(f_joints, axis=0)
                center = (top_right - bottom_left) / 2.0
                center_points.append(center)
            center_point = np.array(center_points).mean(axis=0)

            T, H, W, C = color_video.shape
            cx = center_point[0]
            if cx + H > W:
                lx = W - H
            elif cx - H < 0:
                lx = 0
            else:
                lx = cx - H // 2

            # color

            # crop
            color_video = color_video[:, :, lx : lx + H]

            # resize
            color_video = [imresize(img, (img_size, img_size)) for img in color_video]
            color_video = np.stack(color_video)
            np.savez_compressed(str(out_path / "color.npz"), data=color_video)

            # save
            # dataio.save_video_as_images(color_video, save_path/name/'color')
            dataio.write_video(color_video, save_path / "color" / (name + ".mp4"))

            # depth

            # crop
            depth_video = depth_video[:, :, lx : lx + H]

            # resize
            depth_video = [
                imresize(img, (img_size, img_size), "nearest") for img in depth_video
            ]
            depth_video = np.stack(depth_video)

            # limit value range of depth video
            # fg_values = depth_video[depth_video!=10000000000.0]
            # depth_video_vis = np.clip(depth_video, fg_values.min(), fg_values.max())
            # depth_video_vis = utils.min_max_norm(depth_video) * 255.
            # depth_video_vis = depth_video_vis.astype(np.uint8)
            depth_video = np.clip(depth_video, 0, 15.0)
            # dataio.write_video(depth_video_vis, save_path/'depth'/(name+".mp4"))
            np.savez_compressed(str(out_path / "depth.npz"), data=depth_video)

            # semantic segmentation

            # crop
            segm_video = segm_video[:, :, lx : lx + H]

            # resize
            segm_video = [
                imresize(img, (img_size, img_size), "nearest") for img in segm_video
            ]
            segm_video = np.stack(segm_video)
            segm_video = np.eye(len(segm_part_colors))[segm_video]
            np.savez_compressed(str(out_path / "segm.npz"), data=segm_video)

            # give region color to segmentation video
            # segm_video_vis = make_segmentation_video(segm_video) * 255
            # dataio.save_video_as_images(segm_video_vis, save_path/name/'segm')

            return [name, T]
        except Exception:
            import traceback

            traceback.print_exc()
            print(video_path)
            return

    # perform preprocess with multi threads
    (save_path / "color").mkdir(parents=True, exist_ok=True)
    (save_path / "depth").mkdir(parents=True, exist_ok=True)
    (save_path / "segm").mkdir(parents=True, exist_ok=True)
    (save_path / "pose").mkdir(parents=True, exist_ok=True)

    # https://github.com/gulvarol/surreal/blob/8af8ae195e6b4bb39a0fb64524a15a434ea620f6/datageneration/main_part1.py#L34
    # pose_connections = np.asarray([ ])

    print(len(video_paths), "samples found")
    video_infos = Parallel(n_jobs=n_jobs, verbose=3)(
        [
            delayed(_preprocess)(paths, save_path, length, img_size, segm_part_colors)
            for paths in zip(video_paths, depth_paths, segm_paths, info_paths)
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
