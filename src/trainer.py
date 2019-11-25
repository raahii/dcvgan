import copy
import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.optim as optim
from torch import nn

import dataio
import util
from dataset import VideoDataLoader
from evaluation import compute_conv_features
from evaluation import evaluate as eval_framework
from generator import ColorVideoGenerator, GeometricVideoGenerator
from logger import Logger, MetricType
from loss import Loss


class Trainer(object):
    def __init__(
        self,
        dataloader: VideoDataLoader,
        logger: Logger,
        models: Dict[str, nn.Module],
        optimizers: Dict[str, Any],
        loss: Loss,
        configs: Dict[str, Any],
    ):
        self.dataloader = dataloader
        self.logger = logger
        self.models = models
        self.optimizers = optimizers
        self.loss = loss
        self.configs = configs
        self.device = util.current_device()
        self.geometric_info = configs["geometric_info"]["name"]

        self.num_log, self.rows_log, self.cols_log = 25, 5, 5

        self.eval_batchsize = configs["evaluation"]["batchsize"]
        self.eval_num_smaples = configs["evaluation"]["num_samples"]
        self.eval_metrics = configs["evaluation"]["metrics"]

        # dataloader for logging real samples on tensorboard
        self.dataloader_log = VideoDataLoader(
            self.dataloader.dataset,
            batch_size=self.num_log,
            num_workers=4,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

        self.model_snapshots_path = self.logger.path / "models"
        self.model_snapshots_path.mkdir(parents=True, exist_ok=True)

        self.adv_loss = nn.BCEWithLogitsLoss(reduction="sum")

        # copy config file to log directory
        shutil.copy(configs["config_path"], str(self.logger.path / "config.yml"))

        self.iteration: int = 0
        self.epoch: int = 0
        self.save_classobj()

    def save_classobj(self):
        """
        Save nn.Module class object using torch.save
        """
        for name, _model in self.models.items():
            model: nn.Module = copy.deepcopy(_model.cpu())
            torch.save(model, self.model_snapshots_path / f"{name}_model.pth")

    def save_params(self):
        """
        Save nn.Module class object using torch.save
        """
        for name, model in self.models.items():
            torch.save(
                model.state_dict(),
                self.model_snapshots_path / f"{name}_params_{self.iteration:05d}.pth",
            )

    def log_hparams(self):
        """
        Save training configurations as hyperparameters to tensorboard.
        """

        def flat(item: Any, key: str) -> Dict[str, str]:
            if type(item) != dict:
                return {key: str(item)}

            _dict = {}
            for k, v in item.items():
                if key == "":
                    _dict = dict(_dict, **flat(v, k))
                else:
                    _dict = dict(_dict, **flat(v, key + "/" + k))

            return _dict

        _configs = flat(self.configs, "")
        self.logger.tf_log_hparams(_configs)

    def log_samples(
        self, ggen: GeometricVideoGenerator, cgen: ColorVideoGenerator, iteration: int
    ):
        """
        Log generator samples into TensorBoard.

        Parameters
        ----------
        ggen : GeometricVideoGenerator
            The geometric information video generator.

        cgen : ColorVideoGenerator
            The color video generator.

        iteration : int
            Iteration count.
        """
        ggen.eval()
        cgen.eval()

        # fake samples
        # generate sampels (dtype: int, axis: (B, C, T, H, W))
        xg_fake, xc_fake = util.generate_samples(ggen, cgen, self.num_log, self.num_log)

        # log histgram of fake samples
        self.logger.tf_log_histgram(xg_fake[:, 0], "geospace_fake", iteration)
        self.logger.tf_log_histgram(xc_fake[:, 0], "colorspace_fake", iteration)

        # make a grid video
        xg_fake = util.make_video_grid(xg_fake, self.rows_log, self.cols_log)
        xc_fake = util.make_video_grid(xc_fake, self.rows_log, self.cols_log)
        x_fake = np.concatenate([xg_fake, xc_fake], axis=-1)  # concat

        # log fake samples (dtype: int, axis: (B, T, C, H, W))
        x_fake = x_fake.transpose(0, 2, 1, 3, 4)
        self.logger.tf_log_video(x_fake, "fake_samples", iteration)

        # real samples
        # take next batch: (dtype: float, axis: (B, C, T, H, W))
        batch = next(self.dataloader_log.__iter__())
        xg_real, xc_real = batch[self.geometric_info], batch["color"]

        # convert xc to np.ndarray: (dtype: int, axis: (B, C, T, H, W))
        xc_real = util.videos_to_numpy(xc_real)

        # convert xg to np.ndarray: (dtype: int, axis: (B, C, T, H, W))
        xg_real = xg_real.data.cpu().numpy()
        xg_real = util.geometric_info_in_color_format(xg_real, ggen.geometric_info)

        # log histgram of real samples
        self.logger.tf_log_histgram(xg_real[:, 0], "geospace_real", iteration)
        self.logger.tf_log_histgram(xc_real[:, 0], "colorspace_real", iteration)

        # make a grid video
        xc_real = util.make_video_grid(xc_real, self.rows_log, self.cols_log)
        xg_real = util.make_video_grid(xg_real, self.rows_log, self.cols_log)
        x_real = np.concatenate([xg_real, xc_real], axis=-1)

        # log fake samples (dtype: int, axis: (B, T, C, H, W))
        x_real = x_real.transpose(0, 2, 1, 3, 4)
        self.logger.tf_log_video(x_real, "real_samples", iteration)

    def evaluate(self, ggen: GeometricVideoGenerator, cgen: ColorVideoGenerator):
        """
        Evaluate generated samples using evaluation metrics for GANs.
        Currently supporing InceptionScore, Frechet Inception Distance.

        Parameters
        ----------
        ggen : nn.Module
            The geometric information video generator.

        cgen : nn.Module
            The color video generator.
        """
        # directory contains all color videos (.mp4) of the dataset
        dataset_dir = Path(self.dataloader.dataset.root_path) / "color"
        if not (
            (dataset_dir / "features.npy").exists()
            and (dataset_dir / "probs.npy").exists()
        ):
            # convert to convolutional features with inpception model
            f, p = compute_conv_features.convert(self.eval_batchsize, dataset_dir)
            compute_conv_features.save(f, p, dataset_dir)

        # generate fake samples
        _, xc = util.generate_samples(
            ggen,
            cgen,
            self.eval_num_smaples,
            self.eval_batchsize,
            desc=f"sampling {self.eval_num_smaples} videos for evalaution",
            verbose=False,
        )
        ggen, cgen = ggen.to("cpu"), cgen.to("cpu")

        # save them in a temporary directory
        temp = tempfile.TemporaryDirectory()
        temp_dir = Path(temp.name)
        samples_dir = temp_dir / "samples"
        samples_dir.mkdir()
        for i, x in enumerate(xc):
            dataio.write_video(x, samples_dir / f"{i}.mp4")

        # convert to convolutional features with inpception model
        f, p = compute_conv_features.convert(self.eval_batchsize, samples_dir)
        compute_conv_features.save(f, p, temp_dir)

        for m in self.eval_metrics:
            # calculate the score
            r = eval_framework.compute_metric(m, [temp_dir, dataset_dir])
            self.logger.update(m, r["score"])

        temp.cleanup()
        ggen, cgen = ggen.to(self.device), cgen.to(self.device)

    def train(self):
        """
        Start training.
        """
        # retrieve models and move them if necessary
        ggen, cgen = self.models["ggen"], self.models["cgen"]
        idis, vdis = self.models["idis"], self.models["vdis"]

        ggen, cgen = ggen.to(self.device), cgen.to(self.device)
        idis, vdis = idis.to(self.device), vdis.to(self.device)

        # optimizers
        opt_ggen, opt_cgen = self.optimizers["ggen"], self.optimizers["cgen"]
        opt_idis, opt_vdis = self.optimizers["idis"], self.optimizers["vdis"]

        # define metrics
        self.logger.define("loss_gen", MetricType.Loss)
        self.logger.define("loss_idis", MetricType.Loss)
        self.logger.define("loss_vdis", MetricType.Loss)
        for m in self.configs["evaluation"]["metrics"]:
            self.logger.define(m, MetricType.Float)

        # log configurations
        self.log_hparams()

        # training loop
        self.logger.debug("(trainer)")
        self.logger.debug(f"epochs: {self.configs['n_epochs']}", 1)
        self.logger.debug(f"device: {self.device}", 1)

        self.logger.debug("(evaluation)")
        self.logger.debug(f"batchsize: {self.eval_batchsize}", 1)
        self.logger.debug(f"num_samples: {self.eval_num_smaples}", 1)
        self.logger.debug(f"metrics: {self.eval_metrics}", 1)
        self.logger.debug("(start training)")
        self.logger.print_header()
        for i in range(self.configs["n_epochs"]):
            self.epoch += 1
            for batch in iter(self.dataloader):
                self.iteration += 1
                self.logger.update("iteration", self.iteration)
                self.logger.update("epoch", self.epoch)

                # random video frame index for image discriminator
                # the value is commonly used for both generator phase
                # and discriminator phase in a iteration
                t_rand = np.random.randint(ggen.video_length)
                tg_rand, tc_rand = t_rand, t_rand

                # --------------------
                # phase discriminator
                # --------------------
                idis.train()
                vdis.train()
                idis.zero_grad()
                vdis.zero_grad()

                # real batch
                xc_real = batch["color"]
                xc_real = xc_real.to(self.device)

                xg_real = batch[self.geometric_info]
                xg_real = xg_real.to(self.device)

                y_real_i = idis(xg_real[:, :, tg_rand], xc_real[:, :, tc_rand])
                y_real_v = vdis(xg_real, xc_real)

                # fake batch
                xg_fake = ggen.sample_videos(self.configs["batchsize"])
                xc_fake = cgen.forward_videos(xg_fake)

                y_fake_i = idis(xg_fake[:, :, tg_rand], xc_fake[:, :, tc_rand])
                y_fake_v = vdis(xg_fake, xc_fake)

                # compute loss
                loss_idis = self.loss.compute_dis_loss(y_real_i, y_fake_i)
                loss_vdis = self.loss.compute_dis_loss(y_real_v, y_fake_v)
                loss_dis = loss_idis + loss_vdis

                # update weights
                if self.iteration % self.configs["num_gen_update"] == 0:
                    loss_dis.backward()
                    opt_idis.step()
                    opt_vdis.step()
                else:
                    loss_dis.detach_()

                self.logger.update("loss_idis", loss_idis.cpu().item())
                self.logger.update("loss_vdis", loss_vdis.cpu().item())

                # free grads
                y_fake_i.detach()
                y_fake_v.detach()

                # --------------------
                # phase generator
                # --------------------
                ggen.train()
                cgen.train()
                ggen.zero_grad()
                cgen.zero_grad()

                # fake batch
                xg_fake = ggen.sample_videos(self.configs["batchsize"])
                xc_fake = cgen.forward_videos(xg_fake)

                y_fake_i = idis(xg_fake[:, :, tg_rand], xc_fake[:, :, tc_rand])
                y_fake_v = vdis(xg_fake, xc_fake)

                # compute loss
                loss_gen = self.loss.compute_gen_loss(y_fake_i, y_fake_v)

                # update weights
                if self.iteration % self.configs["num_dis_update"] == 0:
                    loss_gen.backward()
                    opt_ggen.step()
                    opt_cgen.step()
                else:
                    loss_gen.detach_()

                self.logger.update("loss_gen", loss_gen.cpu().item())

                # free grads
                y_real_i.detach()
                y_real_v.detach()

                # --------------------
                # others
                # --------------------

                # snapshot models
                if self.iteration % self.configs["snapshot_interval"] == 0:
                    self.save_params()

                # log samples
                if self.iteration % self.configs["log_samples_interval"] == 0:
                    self.log_samples(ggen, cgen, self.iteration)

                # evaluation
                if self.iteration % self.configs["evaluation_interval"] == 0:
                    self.evaluate(ggen, cgen)

                # log
                if self.iteration % self.configs["log_interval"] == 0:
                    self.logger.log()
                    self.logger.clear()

        self.save_params()
        self.log_samples(ggen, cgen, self.iteration)
