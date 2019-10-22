import unittest

import torch

from discriminator import ImageDiscriminator, VideoDiscriminator

IMAGE_SIZE = 64
VIDEO_LENGTH = 16
BATCHSIZE = 2

GEOMTRIC_INFO_CH = 1
COLOR_CH = 3


class TestModelForward(unittest.TestCase):
    def test_image_discriminator(self):
        inputs = {
            "ch1": GEOMTRIC_INFO_CH,
            "ch2": COLOR_CH,
            "use_noise": True,
            "noise_sigma": 0.2,
        }
        idis = ImageDiscriminator(**inputs)

        xg_shape = (BATCHSIZE, GEOMTRIC_INFO_CH, IMAGE_SIZE, IMAGE_SIZE)
        xg = torch.empty(xg_shape, device=idis.device).normal_()

        xc_shape = (BATCHSIZE, COLOR_CH, IMAGE_SIZE, IMAGE_SIZE)
        xc = torch.empty(xc_shape, device=idis.device).normal_()

        output = idis(xg, xc)

        expected = (BATCHSIZE, 4, 4)
        self.assertEqual(expected, output.shape)

    def test_video_discriminator(self):
        inputs = {
            "ch1": GEOMTRIC_INFO_CH,
            "ch2": COLOR_CH,
            "use_noise": True,
            "noise_sigma": 0.2,
        }
        vdis = VideoDiscriminator(**inputs)

        xg_shape = (BATCHSIZE, GEOMTRIC_INFO_CH, VIDEO_LENGTH, IMAGE_SIZE, IMAGE_SIZE)
        xg = torch.empty(xg_shape, device=vdis.device).normal_()

        xc_shape = (BATCHSIZE, COLOR_CH, VIDEO_LENGTH, IMAGE_SIZE, IMAGE_SIZE)
        xc = torch.empty(xc_shape, device=vdis.device).normal_()

        output = vdis(xg, xc)

        expected = (BATCHSIZE, 4, 4, 4)
        self.assertEqual(expected, output.shape)


if __name__ == "__main__":
    unittest.main()
