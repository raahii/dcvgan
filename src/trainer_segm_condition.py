import pickle
import random
import shutil
import time
from pathlib import Path

import numpy as np

import torch
import torch.optim as optim
import utils
from datasets.surreal import segm_part_colors
from graphviz import Digraph
from logger import Logger
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchviz.dot import make_dot


class Trainer(object):
    def __init__(self, dataloader, configs):
        self.batchsize = configs["batchsize"]
        self.n_epochs = configs["n_epochs"]
        self.video_length = configs["video_length"]

        self.dataloader = dataloader

        self.num_log, self.rows_log, self.cols_log = 25, 5, 5
        self.dataloader_log = DataLoader(
            self.dataloader.dataset,
            batch_size=self.num_log,
            num_workers=1,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

        self.log_dir = Path(configs["log_dir"]) / configs["experiment_name"]
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.tensorboard_dir = (
            Path(configs["tensorboard_dir"]) / configs["experiment_name"]
        )
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)

        self.logger = Logger(
            self.log_dir, self.tensorboard_dir, configs["log_interval"]
        )

        self.evaluation_interval = configs["evaluation_interval"]
        self.log_samples_interval = configs["log_samples_interval"]

        self.gan_criterion = nn.BCEWithLogitsLoss(reduction="sum")
        self.use_cuda = torch.cuda.is_available()
        self.device = self.use_cuda and torch.device("cuda") or torch.device("cpu")
        self.configs = configs

        # copy config file to log directory
        shutil.copy(configs["config_path"], str(self.log_dir / "config.yml"))

        self.fix_seed()

        if configs["dataset"]["cond"] == "segm":
            self.segm_colors = torch.from_numpy(segm_part_colors).float()
            self.segm_colors = self.segm_colors.to(self.device)

    def fix_seed(self):
        seed = self.configs["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def make_segmentation_video(self, video):
        N, T, H, W = video.shape

        segm_video = torch.zeros((T, H, W, 3), dtype=torch.float32, device=self.device)

        for i in range(N):
            n = int((video[i] == 1).sum().data)
            if n != 0:
                segm_video[video[i] == 1] = self.segm_colors[i].repeat(n, 1)

        return segm_video

    def create_optimizer(self, params, lr, decay):
        return optim.Adam(params, lr=lr, betas=(0.5, 0.999), weight_decay=decay)

    def compute_dis_loss(self, y_real, y_fake):
        ones = torch.ones_like(y_real, device=self.device)
        zeros = torch.zeros_like(y_fake, device=self.device)

        loss = self.gan_criterion(y_real, ones) / y_real.numel()
        loss += self.gan_criterion(y_fake, zeros) / y_fake.numel()

        return loss

    def compute_gen_loss(self, y_fake_i, y_fake_v):
        ones_i = torch.ones_like(y_fake_i, device=self.device)
        ones_v = torch.ones_like(y_fake_v, device=self.device)

        loss = self.gan_criterion(y_fake_i, ones_i) / y_fake_i.numel()
        loss += self.gan_criterion(y_fake_v, ones_v) / y_fake_v.numel()

        return loss

    def log_rgbd_videos(self, cond_videos, color_videos, tag, iteration):
        # (B, C, T, H, W)
        cond_videos = utils.videos_to_numpy(cond_videos)
        color_videos = utils.videos_to_numpy(color_videos)

        # (1, C, T, H*rows, W*cols)
        grid_g = utils.make_video_grid(cond_videos, self.rows_log, self.cols_log)
        grid_c = utils.make_video_grid(color_videos, self.rows_log, self.cols_log)

        # concat them in horizontal direction
        grid_video = np.concatenate([grid_g, grid_c], axis=-1)
        self.logger.tf_log_video(tag, grid_video, iteration)

    def save_graph(self, model, name):
        graph_dir = self.log_dir / "graph"
        graph_dir.mkdir(parents=True, exist_ok=True)
        graph = make_dot(model.forward_dummy().mean(), dict(model.named_parameters()))
        graph.render(str(graph_dir / name))

    def generate_samples(self, ggen, cgen, iteration):
        ggen.eval()
        cgen.eval()

        with torch.no_grad():
            # fake samples
            g = ggen.sample_videos(self.num_log)
            c = cgen.forward_videos(g)
            g = torch.stack([self.make_segmentation_video(_g) for _g in g])
            g = g.permute(0, 4, 1, 2, 3)
            self.log_rgbd_videos(g, c, "fake_samples", iteration)
            self.logger.tf_log_histgram(g[:, :, 0], "segmspace_fake", iteration)
            self.logger.tf_log_histgram(c[:, :, 0], "colorspace_fake", iteration)

            # real samples
            v = next(self.dataloader_log.__iter__())
            c, g = v[:, :3], v[:, 3:]
            g = torch.stack([self.make_segmentation_video(_g) for _g in g])
            g = g.permute(0, 4, 1, 2, 3)
            self.log_rgbd_videos(g, c, "real_samples", iteration)
            self.logger.tf_log_histgram(g[:, :, 0], "segmspace_real", iteration)
            self.logger.tf_log_histgram(c[:, :, 0], "colorspace_real", iteration)

    def snapshot_models(self, ggen, cgen, idis, vdis, i):
        torch.save(
            ggen.state_dict(), str(self.log_dir / "ggen_{:05d}.pytorch".format(i))
        )
        torch.save(
            cgen.state_dict(), str(self.log_dir / "cgen_{:05d}.pytorch".format(i))
        )
        torch.save(
            idis.state_dict(), str(self.log_dir / "idis_{:05d}.pytorch".format(i))
        )
        torch.save(
            vdis.state_dict(), str(self.log_dir / "vdis_{:05d}.pytorch".format(i))
        )

    def train(self, ggen, cgen, idis, vdis):
        if self.use_cuda:
            ggen.cuda()
            cgen.cuda()
            idis.cuda()
            vdis.cuda()

        # save the graphs
        self.save_graph(ggen, "ggen")
        self.save_graph(cgen, "cgen")
        self.save_graph(idis, "idis")
        self.save_graph(vdis, "vdis")

        # create optimizers
        configs = self.configs
        gen_params = list(ggen.parameters()) + list(cgen.parameters())
        opt_gen = self.create_optimizer(gen_params, **configs["gen"]["optimizer"])
        opt_idis = self.create_optimizer(
            idis.parameters(), **configs["idis"]["optimizer"]
        )
        opt_vdis = self.create_optimizer(
            vdis.parameters(), **configs["vdis"]["optimizer"]
        )

        # training loop
        logger = self.logger
        iteration = 1
        for i in range(self.n_epochs):
            for x_real in iter(self.dataloader):
                # --------------------
                # phase generator
                # --------------------
                ggen.train()
                cgen.train()
                opt_gen.zero_grad()

                # fake batch
                g = ggen.sample_videos(self.batchsize)
                c = cgen.forward_videos(g)
                x_fake = torch.cat([c.float(), g.float()], 1)
                t_rand = np.random.randint(self.video_length)
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
                x_real = x_real.cuda() if self.use_cuda else x_real
                x_real = Variable(x_real)

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
                logger.update("loss_gen", loss_gen.cpu().item())
                logger.update("loss_idis", loss_idis.cpu().item())
                logger.update("loss_vdis", loss_vdis.cpu().item())

                # log
                if iteration % configs["log_interval"] == 0:
                    self.logger.log()
                    self.logger.tf_log()
                    self.logger.init()

                # snapshot models
                if iteration % configs["snapshot_interval"] == 0:
                    self.snapshot_models(ggen, cgen, idis, vdis, iteration)

                # log samples
                if iteration % configs["log_samples_interval"] == 0:
                    self.generate_samples(ggen, cgen, iteration)

                # evaluate generated samples
                # if iteration % configs["evaluation_interval"] == 0:
                #    pass

                iteration += 1
                logger.update("iteration", 1)
            logger.update("epoch", 1)

        self.snapshot_models(ggen, cgen, idis, vdis, iteration)
        self.generate_samples(ggen, cgen, iteration)
