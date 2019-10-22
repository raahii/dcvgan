import unittest

import torch

from models import ColorVideoGenerator, DepthVideoGenerator

IMAGE_SIZE = 64
VIDEO_LENGTH = 16
BATCHSIZE = 2

DEPTH_CH = 1
COLOR_CH = 3


class TestModelForward(unittest.TestCase):
    def test_depth_video_generator(self):
        inputs = {"dim_z_content": 30, "dim_z_motion": 10, "video_length": VIDEO_LENGTH}
        mgen = DepthVideoGenerator(**inputs)
        videos = mgen.sample_videos(BATCHSIZE)

        expected = (BATCHSIZE, DEPTH_CH, VIDEO_LENGTH, IMAGE_SIZE, IMAGE_SIZE)
        self.assertEqual(expected, videos.shape)

    def test_color_video_generator(self):
        inputs = {"in_ch": DEPTH_CH, "out_ch": COLOR_CH, "dim_z": 10}
        cgen = ColorVideoGenerator(**inputs)

        input_shape = (BATCHSIZE, DEPTH_CH, IMAGE_SIZE, IMAGE_SIZE)
        x = torch.empty(input_shape, device=cgen.device).normal_()
        z = cgen.make_hidden(BATCHSIZE)

        output = cgen(x, z)
        expected = (BATCHSIZE, COLOR_CH, IMAGE_SIZE, IMAGE_SIZE)
        self.assertEqual(expected, output.shape)
