import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

import util


class Noise(nn.Module):
    def __init__(self, use_noise, sigma=0.2):
        super(Noise, self).__init__()
        self.use_noise = use_noise
        self.sigma = sigma
        self.device = util.current_device()

    def forward(self, x):
        if self.use_noise:
            return (
                x
                + self.sigma
                * torch.empty(
                    x.size(), device=self.device, requires_grad=False
                ).normal_()
            )
        return x


class ImageDiscriminator(nn.Module):
    def __init__(self, ch1, ch2, use_noise=False, noise_sigma=None, ndf=64):
        super(ImageDiscriminator, self).__init__()

        self.ch1, self.ch2 = ch1, ch2
        self.use_noise = use_noise

        self.conv_c = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(self.ch1, ndf // 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_d = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(self.ch2, ndf // 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.main = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False),
        )

        self.device = utils.current_device()

    def forward(self, x):
        hc = self.conv_c(x[:, 0 : self.ch1])
        hd = self.conv_d(x[:, self.ch1 : self.ch1 + self.ch2])
        h = torch.cat([hc, hd], 1)
        h = self.main(h).squeeze()

        return h


class VideoDiscriminator(nn.Module):
    def __init__(self, ch1, ch2, use_noise=False, noise_sigma=None, ndf=64):
        super(VideoDiscriminator, self).__init__()

        self.ch1, self.ch2 = ch1, ch2
        self.use_noise = use_noise

        self.conv_c = nn.Sequential(
            nn.Conv3d(
                ch1, ndf // 2, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_d = nn.Sequential(
            nn.Conv3d(
                ch2, ndf // 2, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.main = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf, ndf * 2, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(
                ndf * 2, ndf * 4, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False
            ),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf * 4, 1, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
        )
        self.device = util.current_device()

    def forward(self, x):
        hc = self.conv_c(x[:, 0 : self.ch1])
        hd = self.conv_d(x[:, self.ch1 : self.ch1 + self.ch2])
        h = torch.cat([hc, hd], 1)
        h = self.main(h).squeeze()

        return h
