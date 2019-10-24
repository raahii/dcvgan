import unittest
import numpy as np

from dataset import VideoDataset
from typing import Union, List

def _is(img: np.ndarray, color: Union[int, List[int]]):
    return np.all(np.equal(color, img))

def new_mock_dataset(video_length, image_size):
    inputs = {
        "name": "mock",
        "dataset_path": "data/raw/mock",
        "preprocess_func": None,
        "video_length": video_length,
        "image_size": image_size,
        "geometric_info": "depth",
        "extension": "png"
    }

    return VideoDataset(**inputs)


class TestDataset(unittest.TestCase):
    def test_batch(self):
        s = 64
        l = 16
        dataset = new_mock_dataset(l, s)

        self.assertEqual(3, len(dataset))
        self.assertEqual(["color", "depth"], list(dataset[0].keys()))
        
        for batch in dataset:
            self.assertTrue((3, l, s, s), batch["color"].shape)
            self.assertTrue((1, l, s, s), batch["depth"].shape)
        
    def test_color_video_tensor(self):
        s = 64
        l = 16
        dataset = new_mock_dataset(l, s)

        colors = [[255, 0, 0], [0, 255, 0], [0,0,255]]
        for i, batch in enumerate(dataset):
            # restore pixel value
            color_video = batch["color"].transpose(1,2,3,0)
            color_video = (color_video + 1) / 2 * 255
            color_video = color_video.astype(np.uint8)
            
            for j, frame in enumerate(color_video):
                self.assertTrue(_is(frame, colors[(i+j)%len(colors)]))

    def test_depth_video_tensor(self):
        s = 64
        l = 16
        dataset = new_mock_dataset(l, s)

        colors = [0, 127, 255]
        for i, batch in enumerate(dataset):
            # restore pixel value
            depth_video = batch["depth"].transpose(1,2,3,0)
            depth_video = (depth_video + 1) / 2 * 255
            depth_video = depth_video.astype(np.uint8)
            
            for j, frame in enumerate(depth_video):
                self.assertTrue(_is(frame, colors[(i+j)%len(colors)]))
