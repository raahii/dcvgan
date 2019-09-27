import copy
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.optim as optim
from graphviz import Digraph
from torch import nn
from torch.utils.data import DataLoader
from torchviz.dot import make_dot

import utils
from logger import Logger, MetricType


class Trainer(object):
    def __init__(
        self,
        dataloader: DataLoader,
        logger: Logger,
        models: Dict[str, nn.Module],
        optimizers: Dict[str, Any],
        configs: Dict[str, Any],
    ):
        self.dataloader = dataloader
        self.logger = logger
        self.models = models
        self.optimizers = optimizers
        self.configs = configs
        self.device = utils.current_device()

        self.num_log, self.rows_log, self.cols_log = 25, 5, 5
        self.dataloader_log = DataLoader(
            self.dataloader.dataset,
            batch_size=self.num_log,
            num_workers=1,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

        self.gen_samples_path = self.logger.path / "videos"
        self.model_snapshots_path = self.logger.path / "models"
        for p in [self.gen_samples_path, self.model_snapshots_path]:
            p.mkdir(parents=True, exist_ok=True)

        self.adv_loss = nn.BCEWithLogitsLoss(reduction="sum")

        # copy config file to log directory
        shutil.copy(configs["config_path"], str(self.logger.path / "config.yml"))

        self.iteration = 0
        self.epoch = 0
        self.snapshot_models()
        self.fix_seed()

    def fix_seed(self):
        seed = self.configs["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def compute_dis_loss(self, y_real, y_fake):
        ones = torch.ones_like(y_real, device=self.device)
        zeros = torch.zeros_like(y_fake, device=self.device)

        loss = self.adv_loss(y_real, ones) / y_real.numel()
        loss += self.adv_loss(y_fake, zeros) / y_fake.numel()

        return loss

    def compute_gen_loss(self, y_fake_i, y_fake_v):
        ones_i = torch.ones_like(y_fake_i, device=self.device)
        ones_v = torch.ones_like(y_fake_v, device=self.device)

        loss = self.adv_loss(y_fake_i, ones_i) / y_fake_i.numel()
        loss += self.adv_loss(y_fake_v, ones_v) / y_fake_v.numel()

        return loss

    def snapshot_models(self):
        for name, _model in self.models.items():
            model: nn.Module = copy.deepcopy(_model.cpu())
            torch.save(model, self.model_snapshots_path / f"{name}_model.pth")

    def snapshot_params(self):
        for name, model in self.models.items():
            torch.save(
                model.state_dict(),
                str(
                    self.model_snapshots_path
                    / f"{name}_params_{self.iteration:05d}.pytorch"
                ),
            )

    def log_rgbd_videos(self, color_videos, depth_videos, tag, iteration):
        # (B, C, T, H, W)
        color_videos = utils.videos_to_numpy(color_videos)
        depth_videos = utils.videos_to_numpy(depth_videos)

        # (1, C, T, H*rows, W*cols)
        grid_c = utils.make_video_grid(color_videos, self.rows_log, self.cols_log)
        grid_d = utils.make_video_grid(depth_videos, self.rows_log, self.cols_log)

        # concat them in horizontal direction
        grid_video = np.concatenate([grid_d, grid_c], axis=-1)

        # (N, T, C, H, W)
        grid_video = grid_video.transpose(0, 2, 1, 3, 4)
        self.logger.tf_log_video(tag, grid_video, iteration)

    def generate_samples(self, dgen, cgen, iteration):
        dgen.eval()
        cgen.eval()

        with torch.no_grad():
            # fake samples
            d = dgen.sample_videos(self.num_log)
            c = cgen.forward_videos(d)
            d = d.repeat(1, 3, 1, 1, 1)  # to have 3-channels
            self.log_rgbd_videos(c, d, "fake_samples", iteration)
            self.logger.tf_log_histgram(d[:, :, 0], "depthspace_fake", iteration)
            self.logger.tf_log_histgram(c[:, :, 0], "colorspace_fake", iteration)

            # fake samples with fixed depth
            d = dgen.sample_videos(1)
            d = d.repeat(self.num_log, 1, 1, 1, 1)
            c = cgen.forward_videos(d)
            d = d.repeat(1, 3, 1, 1, 1)
            self.log_rgbd_videos(c, d, "fake_samples_fixed_depth", iteration)

            # real samples
            v = next(self.dataloader_log.__iter__())
            c, d = v[:, 0:3], v[:, 3:4].repeat(1, 3, 1, 1, 1)
            self.log_rgbd_videos(c, d, "real_samples", iteration)
            self.logger.tf_log_histgram(d[:, :, 0], "depthspace_real", iteration)
            self.logger.tf_log_histgram(c[:, :, 0], "colorspace_real", iteration)

    def train(self):
        # retrieve models and move them if necessary
        dgen, cgen = self.models["dgen"], self.models["cgen"]
        idis, vdis = self.models["idis"], self.models["vdis"]

        # move the models to proper device
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            dgen, cgen = nn.DataParallel(dgen), nn.DataParallel(cgen)
            idis, vdis = nn.DataParallel(idis), nn.DataParallel(vdis)

        dgen, cgen = dgen.to(self.device), cgen.to(self.device)
        idis, vdis = idis.to(self.device), vdis.to(self.device)

        # optimizers
        opt_gen = self.optimizers["gen"]
        opt_idis, opt_vdis = self.optimizers["idis"], self.optimizers["vdis"]

        # define metrics
        self.logger.define("iteration", MetricType.Number)
        self.logger.define("epoch", MetricType.Number)
        self.logger.define("loss_gen", MetricType.Loss)
        self.logger.define("loss_idis", MetricType.Loss)
        self.logger.define("loss_vdis", MetricType.Loss)

        # save hyperparams
        hparams = {}
        for key in ["seed", "batchsize"]:
            hparams[key] = self.configs[key]
        self.logger.tf_hyperparams(hparams)

        # training loop
        self.logger.warning(f"Start training, device: {self.device}, n_gpus: {n_gpus}")
        self.logger.print_header()
        for i in range(self.configs["n_epochs"]):
            self.epoch += 1
            for x_real in iter(self.dataloader):
                self.iteration += 1

                # --------------------
                # phase generator
                # --------------------
                dgen.train()
                cgen.train()
                opt_gen.zero_grad()

                # fake batch
                d = dgen.sample_videos(self.configs["batchsize"])
                c = cgen.forward_videos(d)
                x_fake = torch.cat([c.float(), d.float()], 1)
                t_rand = np.random.randint(self.configs["video_length"])
                y_fake_i = idis(x_fake[:, :, t_rand])
                y_fake_v = vdis(x_fake)

                # compute loss
                loss_gen = self.compute_gen_loss(y_fake_i, y_fake_v)

                # update weights
                loss_gen.backward()
                opt_gen.step()

                # --------------------
                # phase discriminator
                # --------------------
                idis.train()
                opt_idis.zero_grad()
                vdis.train()
                opt_vdis.zero_grad()

                # real batch
                x_real = x_real.float()
                x_real = x_real.to(self.device)

                y_real_i = idis(x_real[:, :, t_rand])
                y_real_v = vdis(x_real)

                x_fake = x_fake.detach()
                y_fake_i = idis(x_fake[:, :, t_rand])
                y_fake_v = vdis(x_fake)

                # compute loss
                loss_idis = self.compute_dis_loss(y_real_i, y_fake_i)
                loss_vdis = self.compute_dis_loss(y_real_v, y_fake_v)

                # update weights
                loss_idis.backward()
                opt_idis.step()
                loss_vdis.backward()
                opt_vdis.step()

                # --------------------
                # others
                # --------------------

                # update metrics
                self.logger.update("iteration", self.iteration)
                self.logger.update("epoch", self.epoch)
                self.logger.update("loss_gen", loss_gen.cpu().item())
                self.logger.update("loss_idis", loss_idis.cpu().item())
                self.logger.update("loss_vdis", loss_vdis.cpu().item())

                # log
                if self.iteration % self.configs["log_interval"] == 0:
                    self.logger.log()
                    self.logger.tf_log_scalars("iteration")
                    self.logger.clear()

                # snapshot models
                if self.iteration % self.configs["snapshot_interval"] == 0:
                    self.snapshot_params()

                # log samples
                if self.iteration % self.configs["log_samples_interval"] == 0:
                    self.generate_samples(dgen, cgen, self.iteration)

                # evaluate generated samples
                # if iteration % self.configs["evaluation_interval"] == 0:
                #    pass

        self.snapshot_models(dgen, cgen, idis, vdis, self.iteration)
        self.generate_samples(dgen, cgen, self.iteration)
