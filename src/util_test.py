import unittest

import numpy as np
import torch

from util import calc_optical_flow, current_device


class TestUtilities(unittest.TestCase):
    def test_current_device(self):
        self.assertEqual(torch.device("cpu"), current_device())

    def test_calc_optical_flow(self):
        video = np.random.randint(0, 255, size=(16, 64, 64, 3))
        video = video.astype(np.uint8)
        flow = calc_optical_flow(video)
        expected = (15, 64, 64, 2)

        self.assertEqual(expected, flow.shape)


if __name__ == "__main__":
    unittest.main()
