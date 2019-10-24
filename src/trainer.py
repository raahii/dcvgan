import copy
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

import util
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
        self.device = util.current_device()

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

        self.adv_loss = nn.BCELoss(reduction="sum")

        # copy config file to log directory
        shutil.copy(configs["config_path"], str(self.logger.path / "config.yml"))

        self.iteration = 0
        self.epoch = 0
        self.save_classobj()
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

    def save_classobj(self):
        for name, _model in self.models.items():
            model: nn.Module = copy.deepcopy(_model.cpu())
            torch.save(model, self.model_snapshots_path / f"{name}_model.pth")

    def save_params(self):
        for name, model in self.models.items():
            torch.save(
                model.state_dict(),
                self.model_snapshots_path / f"{name}_params_{self.iteration:05d}.pth",
            )

    def log_samples(self, ggen, cgen, iteration):
        def deform(video):
            # cast torch.tensor to np.ndarray, shape:(B,C,T,H,W)
            video = util.videos_to_numpy(video)

            # arrange videos in a grid:, shape(1,C,T,H*rows,W*cols)
            video = util.make_video_grid(video, self.rows_log, self.cols_log)

            return video

        ggen.eval()
        cgen.eval()

        with torch.no_grad():
            # generate fake samples
            xg_fake = ggen.sample_videos(self.num_log)
            xc_fake = cgen.forward_videos(xg_fake)
            if self.geometric_info == "depth":
                xg_fake = xg_fake.repeat(1, 3, 1, 1, 1)  # to have 3-channels

            # log histgram of fake samples
            self.logger.tf_log_histgram(xg_fake[:, :, 0], "depthspace_fake", iteration)
            self.logger.tf_log_histgram(xc_fake[:, :, 0], "colorspace_fake", iteration)

            # log fake samples
            xg_fake, xc_fake = deform(xg_fake), deform(xc_fake)
            x_fake = np.concatenate([xg_fake, xc_fake], axis=-1)  # concat
            x_fake = x_fake.transpose(0, 2, 1, 3, 4)  # (N, T, C, H, W)
            self.logger.tf_log_video("fake_samples", x_fake, iteration)

            # retrieve real samples
            batch = next(self.dataloader_log.__iter__())
            xg_real, xc_real = batch[self.geometric_info], batch["color"]
            if self.geometric_info == "depth":
                xg_real = xg_real.repeat(1, 3, 1, 1, 1)  # to have 3-channels

            # log histgram of real samples
            self.logger.tf_log_histgram(xg_real[:, :, 0], "depthspace_real", iteration)
            self.logger.tf_log_histgram(xc_real[:, :, 0], "colorspace_real", iteration)

            # log fake samples
            xg_real, xc_real = deform(xg_real), deform(xc_real)
            x_real = np.concatenate([xg_real, xc_real], axis=-1)  # concat
            x_real = x_real.transpose(0, 2, 1, 3, 4)  # (N, T, C, H, W)
            self.logger.tf_log_video("real_samples", x_real, iteration)

    def train(self):
        # retrieve models and move them if necessary
        ggen, cgen = self.models["ggen"], self.models["cgen"]
        idis, vdis = self.models["idis"], self.models["vdis"]

        # move the models to proper device
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            ggen, cgen = nn.DataParallel(ggen), nn.DataParallel(cgen)
            idis, vdis = nn.DataParallel(idis), nn.DataParallel(vdis)

        ggen, cgen = ggen.to(self.device), cgen.to(self.device)
        idis, vdis = idis.to(self.device), vdis.to(self.device)
        geometric_info = self.configs["geometric_info"]

        # optimizers
        opt_gen = self.optimizers["ggen"]
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
            for batch in iter(self.dataloader):
                self.iteration += 1

                # --------------------
                # phase generator
                # --------------------
                ggen.train()
                cgen.train()
                opt_gen.zero_grad()

                # fake batch
                xg_fake = ggen.sample_videos(self.configs["batchsize"])
                xc_fake = cgen.forward_videos(xg_fake)
                tg_rand = np.random.randint(ggen.video_length)
                tc_rand = np.random.randint(cgen.video_length)
                y_fake_i = idis(xg_fake[:, :, tg_rand], xc_fake[:, :, tc_rand])
                y_fake_v = vdis(xg_fake, xc_fake)

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
                xc_real = batch["color"].float()
                xc_real = xc_real.to(self.device)

                xg_real = batch[geometric_info].float()
                xg_real = xg_real.to(self.device)

                y_real_i = idis(xg_real[:, :, tg_rand], xc_real[:, :, tc_rand])
                y_real_v = vdis(xg_real, xc_real)

                xg_fake = xg_fake.detach()
                xc_fake = xc_fake.detach()
                y_fake_i = idis(xg_fake[:, :, tg_rand], xc_fake[:, :, tc_rand])
                y_fake_v = vdis(xg_fake, xc_fake)

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
                    self.save_params()

                # log samples
                if self.iteration % self.configs["log_samples_interval"] == 0:
                    self.generate_samples(ggen, cgen, self.iteration)

                # evaluate generated samples
                # if iteration % self.configs["evaluation_interval"] == 0:
                #    pass

        self.save_params()
        self.generate_samples(ggen, cgen, self.iteration)
