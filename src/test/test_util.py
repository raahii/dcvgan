import unittest

import numpy as np
import torch

from generator import ColorVideoGenerator, DepthVideoGenerator
from util import calc_optical_flow, current_device, generate_samples


class TestUtilities(unittest.TestCase):
    def test_current_device(self):
        self.assertEqual(torch.device("cpu"), current_device())

    def test_calc_optical_flow(self):
        video = np.random.randint(0, 255, size=(16, 64, 64, 3))
        video = video.astype(np.uint8)
        flow = calc_optical_flow(video)
        expected = (15, 64, 64, 2)

        self.assertEqual(expected, flow.shape)

    def test_generate_samples(self):
        IMAGE_SIZE = 64
        VIDEO_LENGTH = 16
        GEOMTRIC_INFO_CH = 1

        # init ggen
        inputs = {"dim_z_content": 30, "dim_z_motion": 10, "video_length": VIDEO_LENGTH}
        ggen = DepthVideoGenerator(**inputs)

        # init cgen
        inputs = {"in_ch": GEOMTRIC_INFO_CH, "dim_z": 10}
        cgen = ColorVideoGenerator(**inputs)

        # generate
        cases = [(3, 1), (3, 2), (3, 4)]
        for num, batchsize in cases:
            xg, xc = generate_samples(ggen, cgen, num, batchsize)

            # xg
            self.assertTrue(isinstance(xg, np.ndarray))
            self.assertEqual(len(xg), num)
            s = (num, VIDEO_LENGTH, IMAGE_SIZE, IMAGE_SIZE, 3)
            self.assertEqual(xg.shape, s)

            # xc
            self.assertTrue(isinstance(xc, np.ndarray))
            self.assertEqual(len(xc), num)
            s = (num, VIDEO_LENGTH, IMAGE_SIZE, IMAGE_SIZE, 3)
            self.assertEqual(xc.shape, s)


if __name__ == "__main__":
    unittest.main()
