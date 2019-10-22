import unittest
from pathlib import Path

import torch

from logger import Logger, MetricType
from util import current_device


def new_logger():
    out_path = Path("/tmp/log")
    tfb_path = Path("/tmp/log/tf")
    return Logger(out_path, tfb_path)


class TestLogger(unittest.TestCase):
    def test_new_logger(self):
        logger = new_logger()
        self.assertTrue(len(logger.metric_keys()) > 0)

    def test_define_metric(self):
        logger = new_logger()

        new_key = "foo"
        logger.define(new_key, MetricType.Number)
        keys = logger.metric_keys()
        self.assertEqual(new_key, keys[2])

        new_key = "bar"
        logger.define(new_key, MetricType.Number)
        keys = logger.metric_keys()
        self.assertEqual(new_key, keys[3])

        new_key = "hoge"
        logger.define(new_key, MetricType.Number, 1000)
        keys = logger.metric_keys()
        self.assertEqual(new_key, keys[0])

        new_key = "piyo"
        logger.define(new_key, MetricType.Number, -1000)
        keys = logger.metric_keys()
        self.assertEqual(new_key, keys[-1])

    def test_update_and_clear(self):
        logger = new_logger()

        logger.define("foo", MetricType.Number)
        logger.update("foo", 1)
        logger.update("foo", 2)
        self.assertEqual(2, logger.metrics["foo"].value)

        logger.define("bar", MetricType.Loss)
        logger.update("bar", 1)
        logger.update("bar", 2)
        self.assertEqual([1, 2], logger.metrics["bar"].value)

        logger.clear()
        self.assertEqual(0, logger.metrics["foo"].value)
        self.assertEqual([], logger.metrics["bar"].value)


if __name__ == "__main__":
    unittest.main()
