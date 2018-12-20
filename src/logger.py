import time
import numpy as np
from tensorboardX import SummaryWriter

class Logger(object):
    def __init__(self, dataloader, configs):
        self.metrics ={
            "epoch": 0,
            "iteration": 1,
            "loss_gen": 0., "loss_idis": 0.,
            "loss_vdis": 0.,
            "elapsed_time": 0,
        }
        
        self.start_time = time.time()
        self.epoch_iters = len(dataloader)
        # self.writer = SummaryWriter(tensorboard_dir)
        self.log_interval = configs["log_interval"]

        self.display_metric_names()

    def update(self, new_metrics):
        for k, v in new_metrics.items():
            self.metrics[k] = v

    def display_metric_names(self):
        for name in self.metrics.keys():
            print("{:>12} ".format(name), end="")
        print("")

    def log_metrics(self):
        self.metrics["elapsed_time"] = time.time() - self.start_time

        metric_strings = []
        for name, value in self.metrics.items():
            if name in ["epoch", "iteration"]:
                s = "{}".format(value)
            elif name in ["loss_gen", "loss_idis", "loss_vdis"]:
                s = "{:0.3f}".format(value)
            elif name in ["elapsed_time"]:
                value = int(value)
                s = "{:02d}:{:02d}:{:02d}".format(value//3600, value//60, value%60)
            else:
                raise Exception("Unsupported mertic is added")

            metric_strings.append(s)
        
        for s in metric_strings:
            print("{:>12} ".format(s), end="")
        print("")
    
    def next_iter(self):
        # hook
        if self.metrics["iteration"] % self.log_interval == 0:
            self.log_metrics()
        
        # update iteration
        self.metrics['iteration'] += 1
        if self.metrics['iteration'] % self.epoch_iters == 0:
            self.metrics["epoch"] += 1

if __name__=="__main__":
    l = Logger(None, None, [1,2,3])
    time.sleep(3)
    l.log_metrics()
