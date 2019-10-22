import numpy as np
import torch
import torch.nn as nn

import utils


class BaseMidVideoGenerator(nn.Module):
    def __init__(self, out_ch, dim_z_content, dim_z_motion, ngf=64, video_length=16):
        super(BaseMidVideoGenerator, self).__init__()

        self.out_ch = out_ch
        self.dim_z_content = dim_z_content
        self.dim_z_motion = dim_z_motion
        self.video_length = video_length
        self.ngf = ngf

        dim_z = dim_z_motion + dim_z_content
        self.dim_z = dim_z

        self.recurrent = nn.GRUCell(dim_z_motion, dim_z_motion)

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
            nn.ConvTranspose2d(ngf, self.out_ch, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

        self.apply(utils.init_normal)
        self.device = utils.current_device()

    def sample_z_m(self, batchsize):
        h_t = [self.get_gru_initial_state(batchsize)]
        for frame_num in range(self.video_length):
            e_t = self.get_iteration_noise(batchsize)
            h_t.append(self.recurrent(e_t, h_t[-1]))

        # (batchsize, dim_z_motion*self.video_length)
        z_m = torch.stack(h_t[1:], 1)
        # (batchsize*self.video_length, dim_z_motion)
        z_m = z_m.view(batchsize * self.video_length, -1)

        return z_m

    def sample_z_content(self, batchsize):
        content = torch.empty(
            (batchsize, self.dim_z_content), device=self.device
        ).normal_()
        content = content.repeat(1, self.video_length).view(
            batchsize * self.video_length, -1
        )  # same operation as np.repeat
        return content

    def sample_z_video(self, batchsize):
        z_content = self.sample_z_content(batchsize)
        z_motion = self.sample_z_m(batchsize)

        z = torch.cat([z_content, z_motion], dim=1)

        return z

    def sample_videos(self, batchsize):
        z = self.sample_z_video(batchsize)

        h = self.main(z.view(batchsize * self.video_length, self.dim_z, 1, 1))
        h = h.view(batchsize, self.video_length, self.out_ch, 64, 64)

        h = h.permute(0, 2, 1, 3, 4)

        return h

    def get_gru_initial_state(self, batchsize):
        return torch.empty((batchsize, self.dim_z_motion), device=self.device).normal_()

    def get_iteration_noise(self, batchsize):
        return torch.empty((batchsize, self.dim_z_motion), device=self.device).normal_()

    def forward_dummy(self):
        return self.sample_videos(2)


class DepthVideoGenerator(BaseMidVideoGenerator):
    """
    Generator for depth videos
    output channel: 3
    """

    def __init__(self, dim_z_content, dim_z_motion, ngf=64, video_length=16):
        super(DepthVideoGenerator, self).__init__(
            1, dim_z_content, dim_z_motion, ngf, video_length
        )


class Inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Inconv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.main(x)

        return x


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=False):
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
        x = self.main(x)

        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=False):
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
        x = self.main(x)

        return x


class Outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
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
    def __init__(self, in_ch, out_ch, dim_z, ngf=64):
        super(ColorVideoGenerator, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
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

        self.outconv = Outconv(ngf * 2, out_ch)

        self.n_down_blocks = len(self.down_blocks)
        self.n_up_blocks = len(self.up_blocks)

        self.apply(utils.init_normal)
        self.device = utils.current_device()

    def make_hidden(self, batchsize):
        z = torch.empty((batchsize, self.dim_z), device=self.device).normal_()
        z = z.unsqueeze(-1).unsqueeze(-1)  # (B, dim_z, 1, 1)

        return z

    def forward(self, x, z):
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

    def forward_videos(self, xs):
        B, C, T, H, W = xs.shape
        zs = self.make_hidden(B)  # (B,    C, 1, 1)
        zs = zs.unsqueeze(1).repeat(1, T, 1, 1, 1)  # (B, T, C, 1, 1)
        zs = zs.view(B * T, -1, 1, 1)

        xs = xs.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        xs = xs.view(B * T, C, H, W)
        ys = self(xs, zs)
        ys = ys.view(B, T, 3, H, W)
        ys = ys.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)

        return ys


if __name__ == "__main__":
    # print(dict(ColorVideoGenerator(10).named_parameters()).keys())
    # print(DepthVideoGenerator(1, 40, 10).forward_dummy().shape)
    # print(ColorVideoGenerator(10).forward_dummy().shape)
    # print(ImageDiscriminator(1, 3).forward_dummy().shape)
    # print(VideoDiscriminator(1, 3).forward_dummy().shape)
    pass
