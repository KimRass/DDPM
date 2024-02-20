# References:
    # https://github.com/KimRass/Transformer/blob/main/model.py
    # https://nn.labml.ai/diffusion/ddpm/unet.html
    # https://nn.labml.ai/diffusion/ddpm/index.html
    # https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/blob/main/notebooks/08_diffusion/01_ddm/ddm.ipynb
    # https://huggingface.co/blog/annotated-diffusion

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import imageio
import math
from tqdm import tqdm
from pathlib import Path

from utils import image_to_grid, save_image
from model_labml import labmlUNet


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedder(nn.Module):
    # "Parameters are shared across time, which is specified to the network using the Transformer
    # sinusoidal position embedding."
    def __init__(self, n_diffusion_steps, time_channels):
        super().__init__()

        self.d_model = time_channels // 4

        pos = torch.arange(n_diffusion_steps).unsqueeze(1)
        i = torch.arange(self.d_model // 2).unsqueeze(0)
        angle = pos / (10_000 ** (2 * i / self.d_model))

        self.pe_mat = torch.zeros(size=(n_diffusion_steps, self.d_model))
        self.pe_mat[:, 0:: 2] = torch.sin(angle)
        self.pe_mat[:, 1:: 2] = torch.cos(angle)

        self.register_buffer("pos_enc_mat", self.pe_mat)

        self.layers = nn.Sequential(
            nn.Linear(self.d_model, time_channels),
            Swish(),
            nn.Linear(time_channels, time_channels),
        )

    def forward(self, diffusion_step):
        x = torch.index_select(
            self.pe_mat.to(diffusion_step.device), dim=0, index=diffusion_step,
        )
        return self.layers(x)


class ResConvSelfAttnBlock(nn.Module):
    def __init__(self, channels, n_groups=32):
        super().__init__()

        self.gn = nn.GroupNorm(num_groups=n_groups, num_channels=channels)
        self.qkv_proj = nn.Conv2d(channels, channels * 3, 1, 1, 0)
        self.out_proj = nn.Conv2d(channels, channels, 1, 1, 0)
        self.scale = channels ** (-0.5)

    def forward(self, x):
        b, c, h, w = x.shape
        skip = x

        x = self.gn(x)
        x = self.qkv_proj(x)
        q, k, v = torch.chunk(x, chunks=3, dim=1)
        attn_score = torch.einsum(
            "bci,bcj->bij", q.view((b, c, -1)), k.view((b, c, -1)),
        ) * self.scale
        attn_weight = F.softmax(attn_score, dim=2)        
        x = torch.einsum("bij,bcj->bci", attn_weight, v.view((b, c, -1)))
        x = x.view(b, c, h, w)
        x = self.out_proj(x)
        return x + skip


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, n_groups=32, drop_prob=0.1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layers1 = nn.Sequential(
            # "We replaced weight normalization with group normalization
            # to make the implementation simpler."
            nn.GroupNorm(num_groups=n_groups, num_channels=in_channels),
            Swish(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        )
        self.time_proj = nn.Sequential(
            Swish(),
            nn.Linear(time_channels, out_channels),
        )
        self.layers2 = nn.Sequential(
            nn.GroupNorm(num_groups=n_groups, num_channels=out_channels),
            Swish(),
            nn.Dropout(drop_prob),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

        if in_channels != out_channels:
            self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x, t):
        x1 = self.layers1(x)
        # "Diffusion time $t$ is specified by adding the Transformer sinusoidal position embedding
        # into each residual block."
        # "We condition all layers on $t$ by adding in the Transformer sinusoidal position embedding."
        x1 = x1 + self.time_proj(t)[:, :, None, None]
        x1 = self.layers2(x1)
        if self.in_channels != self.out_channels:
            return x1 + self.conv(x)
        return x1 + x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, attn, n_groups=32):
        super().__init__()

        self.attn = attn

        self.res_block = ResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            time_channels=time_channels,
            n_groups=n_groups,
        )
        if attn:
            self.attn_block = ResConvSelfAttnBlock(channels=out_channels)

    def forward(self, x, t):
        x = self.res_block(x, t)
        if self.attn:
            x = self.attn_block(x)
        return x


class MidBlock(nn.Module):
    def __init__(self, channels, time_channels, n_groups=32):
        super().__init__()

        self.res_block1 = ResBlock(
            in_channels=channels,
            out_channels=channels,
            time_channels=time_channels,
            n_groups=n_groups,
        )
        self.attn_block = ResConvSelfAttnBlock(channels=channels)
        self.res_block2 = ResBlock(
            in_channels=channels,
            out_channels=channels,
            time_channels=time_channels,
            n_groups=n_groups,
        )

    def forward(self, x, t):
        x = self.res_block1(x, t)
        x = self.attn_block(x)
        x = self.res_block2(x, t)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, attn, n_groups=32):
        super().__init__()

        self.attn = attn

        self.res_block = ResBlock(
            in_channels=in_channels + out_channels,
            out_channels=out_channels,
            time_channels=time_channels,
            n_groups=n_groups,
        )
        if attn:
            self.attn_block = ResConvSelfAttnBlock(channels=out_channels)

    def forward(self, x, t):
        x = self.res_block(x, t)
        if self.attn:
            x = self.attn_block(x)
        return x


class Downsample(nn.Conv2d):
    def __init__(self, channels):
        super().__init__(channels, channels, 3, 2, 1)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    def __init__(
        self,
        # "Our 32 × 32 models use four feature map resolutions (32 × 32 to 4 × 4),
        # and our 256 × 256 models use six."
        # 4 8 16 32: 4
        # 8 16 32 64: 4
        # 8 16 32 64 128 256: 6
        n_diffusion_steps=1000,
        init_channels=32,
        channels=(64, 128, 256, 512),
        # "All models have self-attention blocks at the 16 × 16 resolution
        # between the convolutional blocks."
        # "We use self-attention at the 16 × 16 feature map resolution."
        attns=(False, False, True, False),
        # "All models have two convolutional residual blocks per resolution level."
        n_blocks=2,
        n_groups=32,
    ):
        super().__init__()

        assert len(attns) == len(channels)

        self.init_conv = nn.Conv2d(3, init_channels, 3, 1, 1)

        self.time_channels = init_channels * 4
        self.time_embed = TimeEmbedder(
            n_diffusion_steps=n_diffusion_steps, time_channels=self.time_channels,
        )

        channels = (init_channels, *channels)
        self.down_blocks = nn.ModuleList()
        for idx in range(len(channels) - 1):
            in_channels = channels[idx]
            out_channels = channels[idx + 1]
            attn=attns[idx]
            # print(in_channels, out_channels, attn)
            for _ in range(n_blocks):
                self.down_blocks.append(
                    DownBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        time_channels=self.time_channels,
                        attn=attn,
                        n_groups=n_groups,
                    )
                )
                in_channels = out_channels

            if idx < len(channels) - 2:
                self.down_blocks.append(Downsample(out_channels))

        self.mid_block = MidBlock(
            channels=out_channels, time_channels=self.time_channels, n_groups=n_groups,
        )

        self.up_blocks = nn.ModuleList()
        for idx in list(reversed(range(1, len(channels)))):
            out_channels = in_channels
            attn = attns[idx - 1]
            # print(in_channels, out_channels, attn)
            for _ in range(n_blocks):
                self.up_blocks.append(
                    UpBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        time_channels=self.time_channels,
                        attn=attn,
                        n_groups=n_groups,
                    )
                )
            in_channels = channels[idx]
            out_channels = channels[idx - 1]
            self.up_blocks.append(
                UpBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    time_channels=self.time_channels,
                    attn=attn,
                    n_groups=n_groups,
                )
            )
            in_channels = out_channels

            if idx > 1:
                self.up_blocks.append(Upsample(out_channels))

        self.fin_block = nn.Sequential(
            nn.GroupNorm(n_groups, out_channels),
            Swish(),
            nn.Conv2d(out_channels, 3, 3, 1, 1)
        )

    def forward(self, noisy_image, diffusion_step):
        x = self.init_conv(noisy_image)
        # print(x.shape)
        t = self.time_embed(diffusion_step)
        # print(t.shape)

        xs = [x]
        for layer in self.down_blocks:
            if isinstance(layer, Downsample):
                x = layer(x)
            else:
                x = layer(x, t)
            # print(x.shape)
            xs.append(x)

        x = self.mid_block(x, t)
        # print(x.shape)

        for layer in self.up_blocks:
            if isinstance(layer, Upsample):
                x = layer(x)
            else:
                x = layer(torch.cat([x, xs.pop()], dim=1), t)
            # print(x.shape)
        assert len(xs) == 0

        x = self.fin_block(x)
        # print(x.shape)
        return x


class DDPM(nn.Module):
    def get_linear_beta_schdule(self):
        # "We set the forward process variances to constants increasing linearly."
        # return torch.linspace(init_beta, fin_beta, n_diffusion_steps) # "$\beta_{t}$"
        return torch.linspace(
            self.init_beta,
            self.fin_beta,
            self.n_diffusion_steps,
            device=self.device,
        ) # "$\beta_{t}$"

    # "We set T = 1000 without a sweep."
    # "We chose a linear schedule from $\beta_{1} = 10^{-4}$ to  $\beta_{T} = 0:02$."
    def __init__(
        self,
        img_size,
        init_channels,
        channels,
        attns,
        device,
        n_blocks,
        n_channels=3,
        n_diffusion_steps=1000,
        init_beta=0.0001,
        fin_beta=0.02,
    ):
        super().__init__()

        self.img_size = img_size
        self.device = device
        self.n_channels = n_channels
        self.n_diffusion_steps = n_diffusion_steps
        self.init_beta = init_beta
        self.fin_beta = fin_beta

        self.beta = self.get_linear_beta_schdule()
        self.alpha = 1 - self.beta # "$\alpha_{t} = 1 - \beta_{t}$"
        # "$\bar{\alpha_{t}} = \prod^{t}_{s=1}{\alpha_{s}}$"
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        self.net = UNet(
            n_diffusion_steps=n_diffusion_steps,
            init_channels=init_channels,
            channels=channels,
            attns=attns,
            n_blocks=n_blocks,
        ).to(device)
        # self.net = labmlUNet().to(device)

    @staticmethod
    def index(x, diffusion_step):
        return torch.index_select(x, dim=0, index=diffusion_step)[:, None, None, None]

    def sample_noise(self, batch_size):
        return torch.randn(
            size=(batch_size, self.n_channels, self.img_size, self.img_size),
            device=self.device,
        )

    def sample_diffusion_step(self, batch_size):
        return torch.randint(
            0, self.n_diffusion_steps, size=(batch_size,), device=self.device,
        )

    def batchify_diffusion_steps(self, cur_diffusion_step, batch_size):
        return torch.full(
            size=(batch_size,),
            fill_value=cur_diffusion_step,
            dtype=torch.long,
            device=self.device,
        )

    # Forward (diffusion) process
    def forward(self, ori_image, diffusion_step, random_noise=None):
        # "$\bar{\alpha_{t}}$"
        alpha_bar_t = self.index(self.alpha_bar, diffusion_step=diffusion_step)
        # $\sqrt{\bar{\alpha_{t}}}x_{0}$
        mean = (alpha_bar_t ** 0.5) * ori_image
        # $(1 - \bar{\alpha_{t}})\mathbf{I}$
        var = 1 - alpha_bar_t
        if random_noise is None:
            random_noise = self.sample_noise(batch_size=ori_image.size(0))
        noisy_image = mean + (var ** 0.5) * random_noise
        return noisy_image

    def predict_noise(self, noisy_image, diffusion_step):
        return self.net(noisy_image=noisy_image, diffusion_step=diffusion_step)

    @torch.inference_mode()
    def denoise(self, noisy_image, cur_diffusion_step):
        diffusion_step = self.batchify_diffusion_steps(
            cur_diffusion_step=cur_diffusion_step, batch_size=noisy_image.size(0),
        )
        alpha_t = self.index(self.alpha, diffusion_step=diffusion_step)
        beta_t = self.index(self.beta, diffusion_step=diffusion_step)
        alpha_bar_t = self.index(self.alpha_bar, diffusion_step=diffusion_step)
        pred_noise = self.net(noisy_image=noisy_image.detach(), diffusion_step=diffusion_step)
        # # ["Algorithm 2-4:
        # $x_{t - 1} = \frac{1}{\sqrt{\alpha_{t}}}
        # \Big(x_{t} - \frac{\beta_{t}}{\sqrt{1 - \bar{\alpha_{t}}}}z_{\theta}(x_{t}, t)\Big)
        # + \sigma_{t}z"$
        mean = (1 / (alpha_t ** 0.5)) * (
            noisy_image - ((beta_t / ((1 - alpha_bar_t) ** 0.5)) * pred_noise)
        )
        # mean = (1 / (alpha_t ** 0.5)) * (noisy_image - (1 - alpha_t) / ((1 - alpha_bar_t) ** 0.5) * pred_noise)
        if cur_diffusion_step > 0:
            var = beta_t
            random_noise = self.sample_noise(batch_size=noisy_image.size(0))
            denoised_image = mean + (var ** 0.5) * random_noise
        else:
            denoised_image = mean
        # denoised_image.clamp_(-1, 1)
        return denoised_image

    @torch.inference_mode()
    def sample(self, batch_size): # Reverse (denoising) process
        x = self.sample_noise(batch_size=batch_size) # "$x_{T}$"
        pbar = tqdm(range(self.n_diffusion_steps - 1, -1, -1), leave=False)
        for cur_diffusion_step in pbar:
            pbar.set_description("Sampling...")

            x = self.denoise(x, cur_diffusion_step=cur_diffusion_step)
        return x