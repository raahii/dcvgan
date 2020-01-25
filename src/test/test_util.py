import unittest

import numpy as np
import torch

from generator import ColorVideoGenerator, GeometricVideoGenerator
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

        for geometric_info, ch in zip(["depth", "optical-flow"], [1, 2]):
            # init ggen
            inputs = {
                "dim_z_content": 30,
                "dim_z_motion": 10,
                "channel": ch,
                "geometric_info": geometric_info,
                "video_length": VIDEO_LENGTH,
            }
            ggen = GeometricVideoGenerator(**inputs)

            # init cgen
            inputs = {"in_ch": ch, "dim_z": 10, "geometric_info": geometric_info}
            cgen = ColorVideoGenerator(**inputs)

            # generate
            cases = [(3, 1), (3, 2), (3, 4)]
            for num, batchsize in cases:
                xg, xc = generate_samples(ggen, cgen, num, batchsize)

                # xg
                self.assertTrue(isinstance(xg, np.ndarray))
                self.assertEqual(xg.dtype, np.uint8)
                self.assertEqual(len(xg), num)
                s = (num, 3, VIDEO_LENGTH, IMAGE_SIZE, IMAGE_SIZE)
                self.assertEqual(xg.shape, s)

                # xc
                self.assertTrue(isinstance(xc, np.ndarray))
                self.assertEqual(xc.dtype, np.uint8)
                self.assertEqual(len(xc), num)
                s = (num, 3, VIDEO_LENGTH, IMAGE_SIZE, IMAGE_SIZE)
                self.assertEqual(xc.shape, s)


if __name__ == "__main__":
    unittest.main()
