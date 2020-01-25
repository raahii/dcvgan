import unittest

import torch

from generator import ColorVideoGenerator, GeometricVideoGenerator

IMAGE_SIZE = 64
VIDEO_LENGTH = 16
BATCHSIZE = 2
COLOR_CH = 3

geometric_infos = {
    "depth": 1,
    "optical-flow": 2,
}


class TestModelForward(unittest.TestCase):
    def test_depth_video_generator(self):
        for name, n_channels in geometric_infos.items():
            inputs = {
                "dim_z_content": 30,
                "dim_z_motion": 10,
                "channel": n_channels,
                "geometric_info": name,
                "video_length": VIDEO_LENGTH,
            }
            ggen = GeometricVideoGenerator(**inputs)
            videos = ggen.sample_videos(BATCHSIZE)

            expected = (
                BATCHSIZE,
                n_channels,
                VIDEO_LENGTH,
                IMAGE_SIZE,
                IMAGE_SIZE,
            )
            self.assertEqual(expected, videos.shape)

    def test_color_video_generator(self):
        for name, n_channels in geometric_infos.items():
            inputs = {"in_ch": n_channels, "dim_z": 10, "geometric_info": name}
            cgen = ColorVideoGenerator(**inputs)

            input_shape = (BATCHSIZE, n_channels, IMAGE_SIZE, IMAGE_SIZE)
            x = torch.empty(input_shape, device=cgen.device).normal_()
            z = cgen.make_hidden(BATCHSIZE)

            output = cgen(x, z)
            expected = (BATCHSIZE, COLOR_CH, IMAGE_SIZE, IMAGE_SIZE)
            self.assertEqual(expected, output.shape)
