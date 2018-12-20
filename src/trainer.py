import time
from pathlib import Path

import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim

# from logger import Logger

if torch.cuda.is_available():
    array_module = torch.cuda
else:
    array_module = torch

def images_to_numpy(tensor):
    imgs = tensor.data.cpu().numpy()
    imgs = imgs.transpose(0, 2, 3, 1) # (B, C, H, W) -> (B, H, W, C)
    imgs = np.clip(imgs, -1, 1)
    imgs = (imgs + 1) / 2 * 255
    imgs = imgs.astype('uint8')

    return imgs

def videos_to_numpy(tensor):
    videos = tensor.data.cpu().numpy()
    import pdb; pdb.set_trace()
    videos = videos.transpose(0, 1, 2, 3, 4)
    videos = np.clip(videos, -1, 1)
    videos = (videos + 1) / 2 * 255
    videos = videos.astype('uint8')

    return videos


class Trainer(object):
    def __init__(self, dataloader, configs):

        self.batchsize = configs["batchsize"]
        self.max_iteration = configs["iterations"]
        self.video_length = configs["video_length"]

        self.dataloader = dataloader
        self.data_enumerater = enumerate(dataloader)

        self.log_folder = Path(configs["result_path"]) / configs["experiment_name"]

        self.display_interval     = configs["display_interval"]
        self.evaluation_interval  = configs["evaluation_interval"]
        self.log_samples_interval = configs["log_samples_interval"]

        self.gan_criterion = nn.BCEWithLogitsLoss()
        self.use_cuda = torch.cuda.is_available()
        self.configs = configs

    def sample_real_batch(self):
        batch_idx, batch = next(self.data_enumerater)
        if self.use_cuda:
            batch = batch.cuda()

        if batch_idx == len(self.dataloader) - 1:
            self.data_enumerator = enumerate(self.dataloader)

        return batch.float()

    def create_optimizer(self, model, lr, decay):
        return optim.Adam(
                model.parameters(),
                lr=lr,
                betas=(0.5, 0.999),
                weight_decay=decay,
                )

    def init_logs(self):
        return 

    def compute_dis_loss(self, y_real, y_fake):
        ones = array_module.ones_like(y_real)
        zeros = array_module.zeros_like(y_fake)

        loss = self.gan_criterion(y_real, ones) + \
                  self.gan_criterion(y_fake, zeros)

        return loss

    def compute_gen_loss(self, y_fake_i, y_fake_v):
        ones_i = array_module.ones_like(y_fake_i)
        ones_v = array_module.ones_like(y_fake_v)

        loss = self.gan_criterion(y_fake_i, ones_i) + \
                self.gan_criterion(y_fake_v, ones_v)

        return loss

    def train(self, gen, idis, vdis):
        if self.use_cuda:
            gen.cuda()
            idis.cuda()
            vdis.cuda()
        
        # logger = Logger(self.log_folder)
        configs = self.configs

        # create optimizers
        opt_gen  = self.create_optimizer(gen,  **configs["gen"]["optimizer"])
        opt_idis = self.create_optimizer(idis, **configs["idis"]["optimizer"])
        opt_vdis = self.create_optimizer(vdis, **configs["vdis"]["optimizer"])

        # training loop
        iteration = 0
        logs = {'loss_gen': 0, 'loss_idis': 0, 'loss_vdis': 0}
        start_time = time.time()

        while True:
            gen.train();  opt_gen.zero_grad()
            idis.train(); opt_idis.zero_grad()
            vdis.train(); opt_vdis.zero_grad()

            #--------------------
            # phase generator
            #--------------------

            # fake batch
            x_fake = gen.sample_videos(self.batchsize).float()

            t_rand = np.random.randint(self.video_length)
            y_fake_i = idis(x_fake[:,:,t_rand])
            y_fake_v = vdis(x_fake)

            # compute loss
            loss_gen = self.compute_gen_loss(y_fake_i, y_fake_v)

            # update weights
            loss_gen.backward(); opt_gen.step()


            #--------------------
            # phase discriminator
            #--------------------
            
            # real batch
            x_real = Variable(self.sample_real_batch(), requires_grad=False)

            y_real_i = idis(x_real[:,:,t_rand])
            y_real_v = vdis(x_real)

            y_fake_i = idis(x_fake[:,:,t_rand].detach())
            y_fake_v = vdis(x_fake.detach())
            
            # compute loss
            loss_idis = self.compute_dis_loss(y_real_i, y_fake_i)
            loss_vdis = self.compute_dis_loss(y_real_v, y_fake_v)

            # update weights
            loss_idis.backward(); opt_idis.step()
            loss_vdis.backward(); opt_vdis.step()

            #--------------------
            # logging
            #--------------------
            
            logs['loss_gen']  += loss_gen.cpu().item()
            logs['loss_idis'] += loss_idis.cpu().item()
            logs['loss_vdis'] += loss_vdis.cpu().item()

