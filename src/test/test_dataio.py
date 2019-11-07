import os
import tempfile
import unittest

import cv2
import numpy as np

import context
from dataio import read_img, read_video, resize_img, write_img, write_video

H = 64
W = 64
T = 16


class TestDataIO(unittest.TestCase):
    def test_resize_img(self):
        for s in [(H, W, 3), (H, W, 1)]:
            dummy = np.random.randint(0, 255, s, dtype=np.uint8)

            expected = (H // 2, W // 2)
            resized = resize_img(dummy, expected)

            self.assertEqual(expected, resized.shape[:2])

    def test_image_io(self):
        img = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)

        ext = ".png"
        with tempfile.NamedTemporaryFile(suffix=ext) as f:
            write_img(img, f.name)
            _img = read_img(f.name)

        self.assertTrue(np.array_equal(img, _img))

    def test_video_io(self):
        # do not use random tensor because of using lossy compression
        vid = np.zeros((T, H, W, 3), dtype=np.uint8)

        ext = ".mp4"
        with tempfile.NamedTemporaryFile(suffix=ext) as f:
            write_video(vid, f.name)
            _vid = read_video(f.name)

        self.assertTrue(np.allclose(vid, _vid, rtol=0, atol=2))
