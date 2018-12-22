import time
from pathlib import Path

import logzero
from tensorboardX import SummaryWriter

class Logger(object):
    def __init__(self, log_folder, tensorboard_dir, log_interval):
        log_file = str(log_folder / 'log')

        self.logger = logzero.setup_logger(
                          name='main',
                          logfile=log_file,
                          level=20,
                          fileLoglevel=10,
                          formatter=None,
                      )

        self.metrics = {
            "epoch": 0,
            "iteration": 1,
            "loss_gen": 0.,
            "loss_idis": 0.,
            "loss_vdis": 0.,
            "elapsed_time": 0,
        }
        
        self.log_interval = log_interval
        
        self.writer = SummaryWriter(str(tensorboard_dir))
        
        self.start_time = time.time()
        self.display_metric_names()

    def display_metric_names(self):
        log_string = ""
        for name in self.metrics.keys():
            log_string += "{:>12} ".format(name)
        self.logger.info(log_string)

    def init(self):
        targets = ["loss_gen", "loss_idis", "loss_vdis"]
        for name in targets:
            self.metrics[name] = 0.

    def update(self, name, value):
        self.metrics[name] += value

    def log(self):
        # display and save logs
        self.metrics["elapsed_time"] = time.time() - self.start_time

        metric_strings = []
        for name, value in self.metrics.items():
            if name in ["epoch", "iteration"]:
                s = "{}".format(value)
            elif name in ["loss_gen", "loss_idis", "loss_vdis"]:
                s = "{:0.3f}".format(value/self.log_interval)
            elif name in ["elapsed_time"]:
                value = int(value)
                s = "{:02d}:{:02d}:{:02d}".format(value//3600, value//60, value%60)
            else:
                raise Exception("Unsupported mertic is added")

            metric_strings.append(s)
        
        log_string = ""
        for s in metric_strings:
            log_string += "{:>12} ".format(s)
        self.logger.info(log_string)

    def tf_log(self):
        step = self.metrics["iteration"]
        for name in ["loss_gen", "loss_idis", "loss_vdis"]:
            value = self.metrics[name]
            self.writer.add_scalar(name, value, step)

    def tf_log_video(self, name, videos, step):
        self.writer.add_video(name, videos, fps=8, global_step=step)

    def tf_log_histgram(self, var, tag, step):
        var = var.clone().cpu().data.numpy()
        self.writer.add_histogram(tag, var, step)

if __name__=="__main__":
    l = Logger(None, None, [1,2,3])
    time.sleep(3)
    l.print_log()
