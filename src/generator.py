import json

import numpy as np
import torch
import torch.nn as nn

import util


class GeometricVideoGenerator(nn.Module):
    """
    The geometric information video generator.

    Parameters
    ----------
    dim_z_content : int
        Number of dimension of the content latent variable.

    dim_z_motion : int
        Number of dimension of the motion latent variable.

    channel : int
        Number of channels of output video

    geometric_info : str
        Type of the geometric infomation.
        The choices are "depth" or "flow"

    ngf : int
        Standard number of the kernel filters

    video_length : int
        The length of output video
    """

    def __init__(
        self,
        dim_z_content: int,
        dim_z_motion: int,
        channel: int,
        geometric_info: str,
        ngf: int = 64,
        video_length: int = 16,
    ):
        super(GeometricVideoGenerator, self).__init__()

        self.dim_z_content = dim_z_content
        self.dim_z_motion = dim_z_motion
        self.channel = channel
        self.geometric_info = geometric_info
        self.video_length = video_length
        self.ngf = ngf

        dim_z = dim_z_motion + dim_z_content
        self.dim_z = dim_z

        self.recurrent: nn.Module = nn.GRUCell(dim_z_motion, dim_z_motion)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(dim_z, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf, self.channel, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

        self.device = util.current_device()

    def get_gru_initial_state(self, batchsize: int) -> torch.Tensor:
        return torch.empty((batchsize, self.dim_z_motion), device=self.device).normal_()

    def get_iteration_noise(self, batchsize: int) -> torch.Tensor:
        return torch.empty((batchsize, self.dim_z_motion), device=self.device).normal_()

    def sample_z_m(self, batchsize: int) -> torch.Tensor:
        h_t = [self.get_gru_initial_state(batchsize)]
        for frame_num in range(self.video_length):
            e_t = self.get_iteration_noise(batchsize)
            h_t.append(self.recurrent(e_t, h_t[-1]))

        # (batchsize, dim_z_motion*self.video_length)
        z_m = torch.stack(h_t[1:], 1)
        # (batchsize*self.video_length, dim_z_motion)
        z_m = z_m.view(batchsize * self.video_length, -1)

        return z_m

    def sample_z_content(self, batchsize: int) -> torch.Tensor:
        z_c = torch.empty((batchsize, self.dim_z_content), device=self.device).normal_()
        z_c = z_c.repeat(1, self.video_length).view(
            batchsize * self.video_length, -1
        )  # same operation as np.repeat
        return z_c

    def sample_z_video(self, batchsize: int) -> torch.Tensor:
        z_content = self.sample_z_content(batchsize)
        z_motion = self.sample_z_m(batchsize)

        z = torch.cat([z_content, z_motion], dim=1)

        return z

    def sample_videos(self, batchsize: int) -> torch.Tensor:
        """
        Sample geometric infomation videos. Equivalent to 'forward' method.

        Parameters
        ----------
        batchsize : int
            Number of videos.

        Returns
        -------
        h : torch.Tensor
            Geometric infomation videos.
            (shape: (batchsize, self.video_length, self.channel, 64, 64),
             value range: [-1, 1])
        """
        z = self.sample_z_video(batchsize)

        h = self.main(z.view(batchsize * self.video_length, self.dim_z, 1, 1))
        h = h.view(batchsize, self.video_length, self.channel, 64, 64)

        h = h.permute(0, 2, 1, 3, 4)

        return h

    def __str__(self, name: str = "ggen") -> str:
        return json.dumps(
            {
                name: {
                    "dim_zc": self.dim_z_content,
                    "dim_zm": self.dim_z_motion,
                    "channel": self.channel,
                    "geometric_info": self.geometric_info,
                    "vlen": self.video_length,
                    "ngf": self.ngf,
                }
            }
        )


class Inconv(nn.Module):
    """
    The first layer of the color video generator.

    Parameters
    ----------
    in_ch : int
        Number of input channels of convolution layer.

    out_ch : int
        Number of output channels of convolution layer.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super(Inconv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.main(x)

        return x


class DownBlock(nn.Module):
    """
    A convolutional block of the color video generator.

    Parameters
    ----------
    in_ch : int
        Number of input channels of convolution layer.

    out_ch : int
        Number of output channels of convolution layer.

    dropout : bool
        If true, dropout layer is added after batch norm layer.
    """

    def __init__(self, in_ch: int, out_ch: int, dropout=False):
        super(DownBlock, self).__init__()

        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        if dropout:
            # between batchnorm and activation
            layers.insert(2, nn.Dropout2d(0.5, inplace=True))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class UpBlock(nn.Module):
    """
    A transposed convolutional block of the color video generator.

    Parameters
    ----------
    in_ch : int
        Number of input channels of convolution layer.

    out_ch : int
        Number of output channels of convolution layer.

    dropout : bool
        If true, dropout layer is added after batch norm layer.
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: bool = False):
        super(UpBlock, self).__init__()

        layers = [
            nn.ConvTranspose2d(
                in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]

        if dropout:
            # between batchnorm and activation
            layers.insert(2, nn.Dropout2d(0.5, inplace=True))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class Outconv(nn.Module):
    """
    The last layer of the color video generator.

    Parameters
    ----------
    in_ch : int
        Number of input channels of convolution layer.

    out_ch : int
        Number of output channels of convolution layer.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super(Outconv, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(
                in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.main(x)

        return x


class ColorVideoGenerator(nn.Module):
    """
    The color video generator.

    Parameters
    ----------
    in_ch : int
        Number of channels of the input geometric information video.

    dim_z : int
        Number of dimension of the color latent variable

    ngf : int
        Standard number of the kernel filters

    video_length : int
        The length of input/output video
    """

    def __init__(self, in_ch: int, dim_z: int, ngf: int = 64, video_length: int = 16):
        super(ColorVideoGenerator, self).__init__()

        self.in_ch = in_ch
        self.out_ch = 3
        self.dim_z = dim_z

        self.inconv = Inconv(in_ch, ngf * 1)
        self.down_blocks = nn.ModuleList(
            [
                DownBlock(ngf * 1, ngf * 1),
                DownBlock(ngf * 1, ngf * 2),
                DownBlock(ngf * 2, ngf * 4),
                DownBlock(ngf * 4, ngf * 4),
                DownBlock(ngf * 4, ngf * 4),
                DownBlock(ngf * 4, ngf * 4),
            ]
        )

        self.up_blocks = nn.ModuleList(
            [
                UpBlock(ngf * 4 + dim_z, ngf * 4, dropout=True),
                UpBlock(ngf * 8, ngf * 4, dropout=True),
                UpBlock(ngf * 8, ngf * 4),
                UpBlock(ngf * 8, ngf * 2),
                UpBlock(ngf * 4, ngf * 1),
                UpBlock(ngf * 2, ngf * 1),
            ]
        )
        self.outconv = Outconv(ngf * 2, self.out_ch)

        self.n_down_blocks = len(self.down_blocks)
        self.n_up_blocks = len(self.up_blocks)

        self.device = util.current_device()

        self.channel = 3
        self.video_length = video_length

    def make_hidden(self, batchsize: int) -> torch.Tensor:
        z = torch.empty((batchsize, self.dim_z), device=self.device).normal_()
        z = z.unsqueeze(-1).unsqueeze(-1)  # (B, dim_z, 1, 1)

        return z

    def forward(self, x, z):
        """
        Foward method to convert a geometric infomation image to color image.

        Parameters
        ----------
        xs : torch.Tensor
            Input geometric infomation videos.

        Returns
        -------
        h : torch.Tensor
            Color videos.
        """
        # video to images
        B, C, H, W = x.shape

        # down
        hs = [self.inconv(x)]
        for i in range(self.n_down_blocks):
            hs.append(self.down_blocks[i](hs[-1]))

        # concat latent variable
        h = torch.cat([hs[-1], z], 1)

        # up
        h = self.up_blocks[0](h)
        for i in range(1, self.n_up_blocks):
            h = torch.cat([h, hs[-i - 1]], 1)
            h = self.up_blocks[i](h)
        h = self.outconv(torch.cat([h, hs[0]], 1))

        return h

    def forward_videos(self, xs: torch.Tensor) -> torch.Tensor:
        """
        Forward method for videos but not images.

        Parameters
        ----------
        xs : torch.Tensor
            Input geometric infomation videos.

        Returns
        -------
        h : torch.Tensor
            Color videos.
        """
        # create z_color for video frames batches
        B, C, T, H, W = xs.shape
        zs = self.make_hidden(B)  # (B,    C, 1, 1)
        zs = zs.unsqueeze(1).repeat(1, T, 1, 1, 1)  # (B, T, C, 1, 1)
        zs = zs.view(B * T, -1, 1, 1)

        # consider video batches as image batches
        xs = xs.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        xs = xs.view(B * T, C, H, W)

        # forward
        ys = self(xs, zs)

        # undo tensor to video batches.
        ys = ys.view(B, T, 3, H, W)
        ys = ys.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)

        return ys

    def __str__(self, name: str = "cgen") -> str:
        return json.dumps(
            {
                name: {
                    "in_ch": self.in_ch,
                    "out_ch": self.out_ch,
                    "dim_z": self.dim_z,
                    "n_down_blocks": self.n_down_blocks,
                    "n_up_blocks": self.n_up_blocks,
                }
            }
        )
