import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.init as init
from torch.autograd import Variable

import numpy as np

if torch.cuda.is_available():
    T = torch.cuda
else:
    T = torch


class Noise(nn.Module):
    def __init__(self, use_noise, sigma=0.2):
        super(Noise, self).__init__()
        self.use_noise = use_noise
        self.sigma = sigma

    def forward(self, x):
        if self.use_noise:
            return x + self.sigma * Variable(T.FloatTensor(x.size()).normal_(), requires_grad=False)
        return x

class DepthVideoGenerator(nn.Module):
    def __init__(self, dim_z_content, dim_z_motion, video_length, ngf=64):
        super(DepthVideoGenerator, self).__init__()

        self.dim_z_content = dim_z_content
        self.dim_z_motion = dim_z_motion
        self.video_length = video_length

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
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        self.apply(init_normal)

    def forward_dummy(self):
        return self.sample_videos(2)

    def sample_z_m(self, batchsize):
        h_t = [self.get_gru_initial_state(batchsize)]
        for frame_num in range(self.video_length):
            e_t = self.get_iteration_noise(batchsize)
            h_t.append(self.recurrent(e_t, h_t[-1]))
        
        # (batchsize, dim_z_motion*self.video_length)
        z_m = torch.stack(h_t[1:], 1)
        # (batchsize*self.video_length, dim_z_motion)
        z_m = z_m.view(batchsize*self.video_length, -1)

        return z_m

    def sample_z_content(self, batchsize):
        content = Variable(T.FloatTensor(batchsize, self.dim_z_content).normal_())
        content = content.repeat(1, self.video_length)\
                         .view(batchsize*self.video_length, -1) # same operation as np.repeat
        return content

    def sample_z_video(self, batchsize):
        z_content = self.sample_z_content(batchsize)
        z_motion = self.sample_z_m(batchsize)
        
        z = torch.cat([z_content, z_motion], dim=1)

        return z

    def sample_videos(self, batchsize):
        z = self.sample_z_video(batchsize)

        h = self.main(z.view(batchsize*self.video_length, self.dim_z, 1, 1))
        h = h.view(batchsize, self.video_length, 1, 64, 64)

        h = h.permute(0, 2, 1, 3, 4)

        return h 

    def sample_images(self, batchsize):
        z = self.sample_z_video(batchsize * self.video_length * 2)

        z = z.view(z.size(0), z.size(1), 1, 1)
        h = self.main(z)

        return h

    def get_gru_initial_state(self, batchsize):
        return Variable(T.FloatTensor(batchsize, self.dim_z_motion).normal_())

    def get_iteration_noise(self, batchsize):
        return Variable(T.FloatTensor(batchsize, self.dim_z_motion).normal_())

class Inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Inconv, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3,
                      stride=1, padding=1, bias=False),
        )
        self.main.apply(init_normal)

    def forward(self, x):
        x = self.main(x)
        return x

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownBlock, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.main.apply(init_normal)

    def forward(self, x):
        x = self.main(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpBlock, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.main.apply(init_normal)

    def forward(self, x):
        x = self.main(x)
        return x

class Outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Outconv, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3,
                                stride=1, padding=1, bias=False),
            nn.Tanh(),
        )
        self.main.apply(init_normal)

    def forward(self, x):
        x = self.main(x)
        return x

class ColorVideoGenerator(nn.Module):
    n_down_blocks = 6
    n_up_blocks   = 6
    use_cuda = torch.cuda.is_available()
    def __init__(self, dim_z, ngf=64):
        super(ColorVideoGenerator, self).__init__()

        self.dim_z = dim_z

        self.inconv = Inconv(1, ngf*1)
        self.down1  = DownBlock(ngf*1, ngf*1)
        self.down2  = DownBlock(ngf*1, ngf*2)
        self.down3  = DownBlock(ngf*2, ngf*4)
        self.down4  = DownBlock(ngf*4, ngf*4)
        self.down5  = DownBlock(ngf*4, ngf*4)
        self.down6  = DownBlock(ngf*4, ngf*4)

        self.up1     = UpBlock(ngf*4+dim_z, ngf*4)
        self.up2     = UpBlock(ngf*8      , ngf*4)
        self.up3     = UpBlock(ngf*8      , ngf*4)
        self.up4     = UpBlock(ngf*8      , ngf*2)
        self.up5     = UpBlock(ngf*4      , ngf*1)
        self.up6     = UpBlock(ngf*2      , ngf*1)
        self.outconv = Outconv(ngf*2, 3)

    def forward(self, x):
        # video to images
        T, C, H, W = x.shape
        
        # prepare z
        z = np.random.normal(0, 1, (self.dim_z))
        z = torch.from_numpy(z).float()
        z = z.unsqueeze(0).repeat(T, 1) # (T, dim_z)
        z = z.unsqueeze(-1).unsqueeze(-1) # (T, dim_z, 1, 1)
        if self.use_cuda:
            z = z.cuda(non_blocking=True)

        # down
        hs = [self.inconv(x)]
        for i in range(1, self.n_down_blocks+1):
            layer = eval(f"self.down{i}")
            hs.append(layer(hs[-1]))
        
        # concat latent variable
        hs[-1] = torch.cat([hs[-1], z], 1)

        # up
        h = self.up1(hs[-1])
        for i in range(2, self.n_up_blocks+1):
            layer = eval(f"self.up{i}")
            h = torch.cat([h, hs[-i]], 1)
            h = layer(h)
        h = self.outconv(torch.cat([h, hs[0]], 1))

        return h

    def forward_videos(self, xs):
        xs = xs.permute(0, 2, 1, 3, 4) #(B,C,T,H,W)->(B,T,C,H,W)
        ys = torch.stack([self(x) for x in xs])
        ys = ys.permute(0, 2, 1, 3, 4) #(B,T,C,H,W)->(B,C,T,H,W)

        return ys

class ImageDiscriminator(nn.Module):
    def __init__(self, use_noise=False, noise_sigma=None, ndf=64):
        super(ImageDiscriminator, self).__init__()

        self.use_noise = use_noise

        self.conv_c = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(3, ndf//2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_d = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(1, ndf//2, 4, 2, 1, bias=False),
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

    def forward(self, x):
        hc = self.conv_c(x[:,0:3])
        hd = self.conv_d(x[:,3:4])
        h = torch.cat([hc, hd], 1)
        h = self.main(h).squeeze()

        return h


class VideoDiscriminator(nn.Module):
    def __init__(self, use_noise=False, noise_sigma=None, ndf=64):
        super(VideoDiscriminator, self).__init__()

        self.use_noise = use_noise

        self.conv_c = nn.Sequential(
            nn.Conv3d(3, ndf//2, 4, stride=(1,2,2), padding=(0,1,1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_d = nn.Sequential(
            nn.Conv3d(1, ndf//2, 4, stride=(1,2,2), padding=(0,1,1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.main = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf, ndf * 2, 4, stride=(1,2,2), padding=(0,1,1), bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf * 2, ndf * 4, 4, stride=(1,2,2), padding=(0,1,1), bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf * 4, 1, 4, stride=(1,2,2), padding=(0,1,1), bias=False),
        )

    def forward(self, x):
        hc = self.conv_c(x[:,0:3])
        hd = self.conv_d(x[:,3:4])
        h = torch.cat([hc, hd], 1)
        h = self.main(h).squeeze()
        return h

if __name__=="__main__":
    input_img_shape = (1, 16, 64, 64)
    out_img_shape   = (3, 16, 64, 64)
    net = ColorVideoGenerator(1, 3, 10)

    # forward
    x = torch.ones(input_img_shape)
    x = net(x)
    print(x.shape, out_img_shape)

    # video batch forward
    input_img_shape = (20, 1, 16, 64, 64)
    out_img_shape   = (20, 3, 16, 64, 64)

    x = torch.ones(input_img_shape)
    x = net.forward_videos(x)
    print(x.shape, out_img_shape)

    print(ColorVideoGenerator.parameters())
