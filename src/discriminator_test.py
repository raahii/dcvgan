import unittest

import torch

from discriminator import ImageDiscriminator, VideoDiscriminator

IMAGE_SIZE = 64
VIDEO_LENGTH = 16
BATCHSIZE = 2

DEPTH_CH = 1
COLOR_CH = 3


class TestModelForward(unittest.TestCase):
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
