import unittest

import torch

from utils import current_device


class TestUtilities(unittest.TestCase):
    def test_current_device(self):
        self.assertEqual(torch.device("cpu"), current_device())


if __name__ == "__main__":
    unittest.main()
