import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

import util


class Noise(nn.Module):
    """
    Adding noise layer.

    Parameters
    ----------
    use_noise : bool
        If true, this layer is activated.

    sigma : float
        Standard deviation of the input gaussian noise.
    """

    def __init__(self, use_noise: bool, sigma: float = 0.2):
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
    """
    The image discriminator.

    Parameters
    ----------
    ch1 : int
        Num channels for input geometric information image.

    ch2 : int
        Num channels for input color image.

    use_noise : bool
        A parameter for the noise layer

    noise_sigma : float
        A parameter for the noise layer

    ndf : int
        Standard number of the kernel filters
    """

    def __init__(
        self,
        ch1: int,
        ch2: int,
        use_noise: bool = False,
        noise_sigma: float = 0,
        ndf: int = 64,
    ):
        super(ImageDiscriminator, self).__init__()

        self.ch1, self.ch2 = ch1, ch2
        self.use_noise = use_noise
        self.noise_sigma = noise_sigma
        self.ndf = ndf

        self.conv_g = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(self.ch1, ndf // 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_c = nn.Sequential(
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

        self.device = util.current_device()

    def forward(self, xg, xc):
        """
        Parameters
        ----------
        xg : torch.Tensor
            The input geometric infomation image
        xc : torch.Tensor
            The input color image

        Returns
        -------
        h : torch.Tensor
            A feature map about the fidelity of input image pair.
            The outputs are not probability values yet.
            (C: 1, H: 4, W: 4)
        """
        hg = self.conv_g(xg)
        hc = self.conv_c(xc)
        h = torch.cat([hc, hg], 1)
        h = self.main(h).squeeze()

        return h

    def __str__(self, name: str = "idis") -> str:
        return json.dumps(
            {
                name: {
                    "ch_g": self.ch1,
                    "ch_c": self.ch2,
                    "ndf": self.ndf,
                    "use_noise": self.use_noise,
                    "noise_sigma": self.noise_sigma,
                }
            }
        )


class VideoDiscriminator(nn.Module):
    """
    The video discriminator.

    Parameters
    ----------
    ch1 : int
        Num channels for input geometric information video.

    ch2 : int
        Num channels for input color video.

    use_noise : bool
        A parameter for the noise layer

    noise_sigma : float
        A parameter for the noise layer

    ndf : int
        Standard number of the kernel filters
    """

    def __init__(
        self,
        ch1: int,
        ch2: int,
        use_noise: bool = False,
        noise_sigma: float = 0,
        ndf: int = 64,
    ):
        super(VideoDiscriminator, self).__init__()

        self.ch1, self.ch2 = ch1, ch2
        self.use_noise = use_noise
        self.noise_sigma = noise_sigma
        self.ndf = ndf

        self.conv_g = nn.Sequential(
            nn.Conv3d(
                ch1, ndf // 2, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_c = nn.Sequential(
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

    def forward(self, xg, xc):
        """
        Parameters
        ----------
        xg : torch.Tensor
            The input geometric infomation video
        xc : torch.Tensor
            The input color video

        Returns
        -------
        h : torch.Tensor
            A feature map about the fidelity of input video pair.
            The outputs are not probability values yet.
            (C:1, T: 4, H: 4, W: 4)
        """
        hg = self.conv_g(xg)
        hc = self.conv_c(xc)
        h = torch.cat([hc, hg], 1)
        h = self.main(h).squeeze()

        return h

    def __str__(self, name: str = "vdis") -> str:
        return json.dumps(
            {
                name: {
                    "ch_g": self.ch1,
                    "ch_c": self.ch2,
                    "ndf": self.ndf,
                    "use_noise": self.use_noise,
                    "noise_sigma": self.noise_sigma,
                }
            }
        )


class GradientDiscriminator(nn.Module):
    """
    The gradient discriminator.

    Parameters
    ----------
    ch1 : int
        Num channels for input geometric information video.

    ch2 : int
        Num channels for input color video.

    use_noise : bool
        A parameter for the noise layer

    noise_sigma : float
        A parameter for the noise layer

    ndf : int
        Standard number of the kernel filters
    """

    def __init__(
        self,
        ch1: int,
        ch2: int,
        use_noise: bool = False,
        noise_sigma: float = 0,
        ndf: int = 64,
    ):
        super(GradientDiscriminator, self).__init__()

        # do not use ch2 now
        self.ch1, self.ch2 = ch1, ch2
        self.use_noise = use_noise
        self.noise_sigma = noise_sigma
        self.ndf = ndf

        self.main = nn.Sequential(
            # 1st
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ch1, ndf, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # 2nd
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf, ndf * 2, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 3rd
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(
                ndf * 2, ndf * 4, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False
            ),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 4th
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf * 4, 1, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
        )
        self.device = util.current_device()

    def forward(self, xg, xc):
        """
        Parameters
        ----------
        xg : torch.Tensor
            The input geometric infomation video

        xc : torch.Tensor
            The input color video

        Returns
        -------
        h : torch.Tensor
            A feature map about the fidelity of input video pair.
            The outputs are not probability values yet.
            (C:1, T: 4, H: 4, W: 4)
        """
        # hg = self.conv_g(xg)
        # hc = self.conv_c(xc)
        # h = torch.cat([hc, hg], 1)
        # h = self.main(h).squeeze()
        _, _, L, _, _ = xg.shape
        h = self.main(xg[:, :, 1:L] - xg[:, :, 0 : L - 1]).squeeze()

        return h

    def __str__(self, name: str = "vdis") -> str:
        return json.dumps(
            {
                name: {
                    "ch_g": self.ch1,
                    "ch_c": self.ch2,
                    "ndf": self.ndf,
                    "use_noise": self.use_noise,
                    "noise_sigma": self.noise_sigma,
                }
            }
        )
