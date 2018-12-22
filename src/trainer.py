import time
from pathlib import Path
import pickle

import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim

from torch.utils.data import DataLoader

from logger import Logger
import utils

class Trainer(object):
    def __init__(self, dataloader, configs):
        self.batchsize = configs["batchsize"]
        self.epoch_iters = len(dataloader)
        self.max_iteration = configs["iterations"]
        self.video_length = configs["video_length"]

        self.dataloader = dataloader

        self.num_log, self.rows_log, self.cols_log = 36, 6, 6
        self.dataloader_log = DataLoader(
                                self.dataloader.dataset, 
                                batch_size=self.num_log,
                                num_workers=1,
                                shuffle=True,
                                drop_last=True,
                                )
        
        self.log_dir = Path(configs["log_dir"]) / configs["experiment_name"]
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.tensorboard_dir = Path(configs["tensorboard_dir"]) / configs["experiment_name"]
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)

        self.logger = Logger(self.log_dir, self.tensorboard_dir,\
                             configs["log_interval"], self.epoch_iters)

        self.evaluation_interval  = configs["evaluation_interval"]
        self.log_samples_interval = configs["log_samples_interval"]

        self.gan_criterion = nn.BCEWithLogitsLoss(reduction='sum')
        self.use_cuda = torch.cuda.is_available()
        self.device = self.use_cuda and torch.device('cuda') or torch.device('cpu')
        self.configs = configs


    def create_optimizer(self, model, lr, decay):
        return optim.Adam(
                model.parameters(),
                lr=lr,
                betas=(0.5, 0.999),
                weight_decay=decay,
                )

    def compute_dis_loss(self, y_real, y_fake):
        ones = torch.ones_like(y_real, device=self.device)
        zeros = torch.zeros_like(y_fake, device=self.device)

        loss  = self.gan_criterion(y_real, ones)  / y_real.numel()
        loss += self.gan_criterion(y_fake, zeros) / y_fake.numel()

        return loss

    def compute_gen_loss(self, y_fake_i, y_fake_v):
        ones_i = torch.ones_like(y_fake_i, device=self.device)
        ones_v = torch.ones_like(y_fake_v, device=self.device)

        loss  = self.gan_criterion(y_fake_i, ones_i) / y_fake_i.numel()
        loss += self.gan_criterion(y_fake_v, ones_v) / y_fake_v.numel()

        return loss

    def log_rgbd_videos(self, color_videos, depth_videos, tag, iteration):
        # (B, C, T, H, W)
        color_videos = utils.videos_to_numpy(color_videos)
        depth_videos = utils.videos_to_numpy(depth_videos)

        # (1, C, T, H*rows, W*cols)
        grid_c = utils.make_video_grid(color_videos, self.rows_log, self.cols_log)
        grid_d = utils.make_video_grid(depth_videos, self.rows_log, self.cols_log)

        grid_video = np.concatenate([grid_d, grid_c], axis=-1)
        self.logger.log_video(tag, grid_video, iteration)

    def train(self, dgen, cgen, idis, vdis):
        def snapshot_models(dgen, cgen, idis, vdis, i):
            torch.save(dgen.state_dict(), str(self.log_dir/'dgen_{:05d}.pytorch'.format(i)))
            torch.save(cgen.state_dict(), str(self.log_dir/'cgen_{:05d}.pytorch'.format(i)))
            torch.save(idis.state_dict(), str(self.log_dir/'idis_{:05d}.pytorch'.format(i)))
            torch.save(vdis.state_dict(), str(self.log_dir/'vdis_{:05d}.pytorch'.format(i)))

        if self.use_cuda:
            dgen.cuda()
            cgen.cuda()
            idis.cuda()
            vdis.cuda()
        
        # create optimizers
        configs = self.configs
        opt_dgen = self.create_optimizer(dgen, **configs["dgen"]["optimizer"])
        opt_cgen = self.create_optimizer(cgen, **configs["cgen"]["optimizer"])
        opt_idis = self.create_optimizer(idis, **configs["idis"]["optimizer"])
        opt_vdis = self.create_optimizer(vdis, **configs["vdis"]["optimizer"])

        # training loop
        logger = self.logger
        dataiter = iter(self.dataloader)
        while True:
            #--------------------
            # phase generator
            #--------------------
            dgen.train();  opt_dgen.zero_grad()
            cgen.train();  opt_cgen.zero_grad()

            # fake batch
            d = dgen.sample_videos(self.batchsize)
            c = cgen.forward_videos(d)
            x_fake = torch.cat([c.float(), d.float()], 1)
            t_rand = np.random.randint(self.video_length)
            y_fake_i = idis(x_fake[:,:,t_rand])
            y_fake_v = vdis(x_fake)

            # compute loss
            loss_gen = self.compute_gen_loss(y_fake_i, y_fake_v)

            # update weights
            loss_gen.backward(); opt_dgen.step(); opt_cgen.step()


            #--------------------
            # phase discriminator
            #--------------------
            
            idis.train(); opt_idis.zero_grad()
            vdis.train(); opt_vdis.zero_grad()

            x_fake = x_fake.detach()

            # real batch
            x_real = next(dataiter).float()
            x_real = x_real.cuda() if self.use_cuda else x_fake
            x_real = Variable(x_real)

            y_real_i = idis(x_real[:,:,t_rand])
            y_real_v = vdis(x_real)
            
            y_fake_i = idis(x_fake[:,:,t_rand])
            y_fake_v = vdis(x_fake)
            
            # compute loss
            loss_idis = self.compute_dis_loss(y_real_i, y_fake_i)
            loss_vdis = self.compute_dis_loss(y_real_v, y_fake_v)

            # update weights
            loss_idis.backward(); opt_idis.step()
            loss_vdis.backward(); opt_vdis.step()

            #--------------------
            # logging
            #--------------------
            logger.update("loss_gen", loss_gen.cpu().item())
            logger.update("loss_idis", loss_idis.cpu().item())
            logger.update("loss_vdis", loss_vdis.cpu().item())

            iteration = self.logger.metrics["iteration"]

            if iteration % (self.epoch_iters-1) == 0:
                dataiter = iter(self.dataloader)

            # snapshot models
            if iteration % configs["snapshot_interval"] == 0:
                snapshot_models(dgen, cgen, idis, vdis, iteration)

            # log samples
            if iteration % configs["log_samples_interval"] == 0:
                dgen.eval(); cgen.eval()

                with torch.no_grad():
                    # fake samples
                    d = dgen.sample_videos(self.num_log)
                    c = cgen.forward_videos(d)
                    d = d.repeat(1,3,1,1,1) # to have 3-channels
                    self.log_rgbd_videos(c, d, 'fake_samples', iteration)
                    self.logger.log_histgram(c[:,0:3,0], 'colorspace_fake', iteration)

                    # fake samples with fixed depth
                    d = dgen.sample_videos(1)
                    d = d.repeat(self.num_log,1,1,1,1)
                    c = cgen.forward_videos(d)
                    d = d.repeat(1,3,1,1,1)
                    self.log_rgbd_videos(c, d, 'fake_samples_fixed_depth', iteration)

                    # real samples
                    v = next(self.dataloader_log.__iter__())
                    c, d = v[:, 0:3], v[:, 3:4].repeat(1,3,1,1,1)
                    self.log_rgbd_videos(c, d, 'real_samples', iteration)
                    self.logger.log_histgram(c[:,0:3,0], 'colorspace_real', iteration)
            
            # evaluate generated samples
            # if iteration % configs["evaluation_interval"] == 0:
            #    pass
            
            if iteration >= self.max_iteration:
                snapshot_models(dgen, cgen, idis, vdis, iteration)
                break

            logger.next_iter()
