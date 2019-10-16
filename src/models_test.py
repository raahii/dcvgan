import unittest

import torch

from models import (
    ColorVideoGenerator,
    DepthVideoGenerator,
    ImageDiscriminator,
    VideoDiscriminator,
)

IMAGE_SIZE = 64
VIDEO_LENGTH = 16
BATCHSIZE = 2

DEPTH_CH = 1
COLOR_CH = 3


class TestModelForward(unittest.TestCase):
    def test_depth_video_generator(self):
        inputs = {
            "out_ch": DEPTH_CH,  # depth
            "dim_z_content": 30,
            "dim_z_motion": 10,
            "video_length": VIDEO_LENGTH,
        }
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

    def test_image_discriminator(self):
        inputs = {
            "ch1": COLOR_CH,
            "ch2": DEPTH_CH,
            "use_noise": True,
            "noise_sigma": 0.2,
        }
        idis = ImageDiscriminator(**inputs)

        input_shape = (BATCHSIZE, DEPTH_CH + COLOR_CH, IMAGE_SIZE, IMAGE_SIZE)
        x = torch.empty(input_shape, device=idis.device).normal_()

        output = idis(x)

        expected = (BATCHSIZE, 4, 4)
        self.assertEqual(expected, output.shape)

    def test_video_discriminator(self):
        inputs = {
            "ch1": COLOR_CH,
            "ch2": DEPTH_CH,
            "use_noise": True,
            "noise_sigma": 0.2,
        }
        vdis = VideoDiscriminator(**inputs)

        input_shape = (
            BATCHSIZE,
            DEPTH_CH + COLOR_CH,
            VIDEO_LENGTH,
            IMAGE_SIZE,
            IMAGE_SIZE,
        )
        x = torch.empty(input_shape, device=vdis.device).normal_()

        output = vdis(x)

        expected = (BATCHSIZE, 4, 4, 4)
        self.assertEqual(expected, output.shape)


if __name__ == "__main__":
    unittest.main()
