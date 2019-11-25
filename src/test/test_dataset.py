import unittest
from typing import List, Union

import numpy as np

import context
from dataset import VideoDataset

COLOR_CHANNEL = 3
DEPTH_CHANNEL = 1
FLOW_CHANNEL = 2


def _is(img: np.ndarray, color: Union[int, List[int]]):
    return np.all(np.equal(color, img))


def new_mockdataset(video_length, image_size, geometric_info="depth"):
    inputs = {
        "name": "mock",
        "dataset_path": "data/raw/mock",
        "preprocess_func": None,
        "video_length": video_length,
        "image_size": image_size,
        "geometric_info": geometric_info,
        "extension": "png",
    }

    return VideoDataset(**inputs)


class TestDataset(unittest.TestCase):
    def test_batch_depth(self):
        size = 64
        length = 16
        dataset = new_mockdataset(length, size, "depth")

        self.assertEqual(3, len(dataset))
        self.assertEqual(["color", "depth"], list(dataset[0].keys()))

        for batch in dataset:
            color, depth = batch["color"], batch["depth"]

            self.assertEqual((COLOR_CHANNEL, length, size, size), color.shape)
            self.assertEqual(np.all(color >= -1.0) and np.all(color <= 1.0))

            self.assertEqual((DEPTH_CHANNEL, length, size, size), batch["depth"].shape)
            self.assertEqual(np.all(depth >= -1.0) and np.all(depth <= 1.0))

    def test_batch_optical_flow(self):
        size = 64
        length = 16
        dataset = new_mockdataset(length, size, "optical-flow")

        self.assertEqual(3, len(dataset))
        self.assertEqual(["color", "optical-flow"], list(dataset[0].keys()))

        for batch in dataset:
            color, flow = batch["color"], batch["optical-flow"]

            self.assertEqual((COLOR_CHANNEL, length, size, size), color.shape)
            self.assertEqual(np.all(color >= -1.0) and np.all(color <= 1.0))

            self.assertEqual((FLOW_CHANNEL, length, size, size), flow.shape)
            self.assertEqual(np.all(flow >= -1.0) and np.all(flow <= 1.0))

    def test_color_video_tensor(self):
        size = 64
        length = 16
        dataset = new_mockdataset(length, size)

        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        for i, batch in enumerate(dataset):
            # restore pixel value
            color_video = batch["color"].transpose(1, 2, 3, 0)
            color_video = (color_video + 1) / 2 * 255
            color_video = color_video.astype(np.uint8)

            for j, frame in enumerate(color_video):
                self.assertTrue(_is(frame, colors[(i + j) % len(colors)]))

    def test_depth_video_tensor(self):
        size = 64
        length = 16
        dataset = new_mockdataset(length, size, "depth")

        colors = [0, 127, 255]
        for i, batch in enumerate(dataset):
            # restore pixel value
            depth_video = batch["depth"].transpose(1, 2, 3, 0)
            depth_video = (depth_video + 1) / 2 * 255
            depth_video = depth_video.astype(np.uint8)

            for j, frame in enumerate(depth_video):
                self.assertTrue(_is(frame, colors[(i + j) % len(colors)]))
