import datetime
import enum
import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

import colorlog
import numpy as np
import torch
from tensorboardX import SummaryWriter


class MetricType(enum.IntEnum):
    """
    Enum represents metric type
    """

    Integer = 1
    Float = 2
    Loss = 3
    Time = 4


class Metric(object):
    """
    Metric class for logger
    """

    mtype_list: List[int] = list(map(int, MetricType))

    def __init__(self, mtype: MetricType, priority: int):
        if mtype not in self.mtype_list:
            raise Exception("mtype is invalid, %s".format(self.mtype_list))

        self.mtype: MetricType = mtype
        self.params: Dict[str, Any] = {}
        self.priority: int = priority
        self.value: Any = 0


class Logger(object):
    """
    Logger for watchting some metrics involving training
    """

    def __init__(self, out_path: Path, tb_path: Path):
        # initialize logging module
        out_path.mkdir(parents=True, exist_ok=True)
        self.path = out_path

        self._logger: logging.Logger = self.new_logging_module(
            __name__, out_path / "log"
        )

        # logging metrics
        self.metrics: OrderedDict[str, Metric] = OrderedDict()

        # tensorboard writer
        tb_path.mkdir(parents=True, exist_ok=True)
        self.tb_path = tb_path
        self.tf_writer: SummaryWriter = SummaryWriter(str(tb_path))

        # automatically add elapsed_time metric
        self.define("epoch", MetricType.Integer, 100)
        self.define("iteration", MetricType.Integer, 99)
        self.define("elapsed_time", MetricType.Time, -1)

        self.indent = " " * 4

    def new_logging_module(self, name: str, log_file: Path) -> logging.Logger:
        # specify format
        log_format: str = "[%(asctime)s] %(message)s"
        date_format: str = "%Y-%m-%d %H:%M:%S"

        # init module
        logger: logging.Logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        # console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s" + log_format, datefmt=date_format
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # file handler
        fh = logging.FileHandler(str(log_file))
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(log_format, datefmt=date_format)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        return logger

    def define(self, name: str, mtype: MetricType, priority=0):
        """
        register a new metric
        """
        metric: Metric = Metric(mtype, priority)
        if mtype == MetricType.Integer:
            metric.value = 0
        elif mtype == MetricType.Float:
            metric.value = 0.0
        elif mtype == MetricType.Loss:
            metric.value = []
        elif mtype == MetricType.Time:
            metric.value = 0
            metric.params["start_time"] = time.time()
        self.metrics[name] = metric

        self.metrics = OrderedDict(
            sorted(self.metrics.items(), key=lambda m: m[1].priority, reverse=True)
        )

    def metric_keys(self) -> List[str]:
        """
        return all registerd metrics
        """
        return list(self.metrics.keys())

    def clear(self):
        """
        init registerd merics values
        """
        for _, metric in self.metrics.items():
            if metric.mtype == MetricType.Loss:
                metric.value = []
            elif metric.mtype == MetricType.Integer:
                metric.value = 0
            elif metric.mtype == MetricType.Float:
                metric.value = 0.0

    def update(self, name: str, value: Any):
        """
        add a new metric value
        """
        m = self.metrics[name]
        if m.mtype in [MetricType.Integer, MetricType.Float]:
            m.value = value
        elif m.mtype == MetricType.Loss:
            m.value.append(value)
        elif m.mtype == MetricType.Time:
            m.value = value - m.params["start_time"]

    def print_header(self):
        """
        add the title of logging
        """
        log_string = ""
        for name in self.metrics.keys():
            log_string += "{:>15} ".format(name)
        self.info(log_string)

    def log(self):
        """
        write logs to stdio and pre-defined file
        """
        self.update("elapsed_time", time.time())

        log_strings: List[str] = []
        for k, m in self.metrics.items():
            if m.mtype == MetricType.Integer:
                s = "{}".format(m.value)
            if m.mtype == MetricType.Float:
                s = "{:0.3f}".format(m.value)
            elif m.mtype == MetricType.Loss:
                if len(m.value) == 0:
                    s = " - "
                else:
                    s = "{:0.3f}".format(sum(m.value) / len(m.value))
            elif m.mtype == MetricType.Time:
                _value = int(m.value)
                s = str(datetime.timedelta(seconds=_value))

            log_strings.append(s)

        log_string: str = ""
        for s in log_strings:
            log_string += "{:>15} ".format(s)

        self.info(log_string)

    def tf_log_scalars(self, x_axis_metric: str):
        """
        plot loss to tensorboard
        automatically only plot MetricType.Loss metrics
        """
        if x_axis_metric not in self.metric_keys():
            raise Exception(f"No such metric: {x_axis_metric}")

        x_metric = self.metrics[x_axis_metric]
        if x_metric.mtype not in [MetricType.Integer, MetricType.Float]:
            raise Exception(f"Invalid metric type: {repr(x_metric.mtype)}")

        step = x_metric.value
        for name, metric in self.metrics.items():
            if metric.mtype != MetricType.Loss:
                continue

            if len(metric.value) == 0:
                raise Exception(f"Metric {name} has no values.")
            mean: float = sum(metric.value) / len(metric.value)
            self.tf_writer.add_scalar(name, mean, step)

    def tf_log_histgram(self, var, tag, step):
        """
        add a histgram data to tensorboard
        """
        var = var.clone().cpu().data.numpy()
        self.tf_writer.add_histogram(tag, var, step)

    def tf_log_image(self, x: torch.Tensor, step: int, tag: str):
        """
        add a image data to tensorboard
        """
        self.tf_writer.add_image(tag, x, step)

    def tf_log_video(self, name, videos, step):
        """
        add video data to tensorboard
        """
        self.tf_writer.add_video(name, videos, fps=8, global_step=step)

    def tf_hyperparams(self, values: Dict[str, Any]):
        self.tf_writer.add_hparams(values, {})

    def info(self, msg: str, level=0):
        self._logger.info(self.indent * level + msg)

    def debug(self, msg: str, level=0):
        self._logger.debug(self.indent * level + msg)

    def warning(self, msg: str, level=0):
        self._logger.warning(self.indent * level + msg)

    def error(self, msg: str, level=0):
        self._logger.error(self.indent * level + msg)

    def critical(self, msg: str, level=0):
        self._logger.critical(self.indent * level + msg)
