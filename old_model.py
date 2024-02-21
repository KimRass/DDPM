# References:
    # https://github.com/w86763777/pytorch-ddpm/blob/master/model.py
    # https://huggingface.co/blog/annotated-diffusion

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import imageio
import math
from tqdm import tqdm
from pathlib import Path

from utils import image_to_grid, print_n_params


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
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
    def __init__(self, in_channels, out_channels, time_channels, drop_prob, attn=False, n_groups=32):
        super().__init__()

        self.layers1 = nn.Sequential(
            # "We replaced weight normalization with group normalization
            # to make the implementation simpler."
            nn.GroupNorm(n_groups, in_channels),
            Swish(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        )
        self.time_proj = nn.Sequential(
            Swish(),
            nn.Linear(time_channels, out_channels),
        )
        self.layers = nn.Sequential(
            nn.GroupNorm(n_groups, out_channels),
            Swish(),
            nn.Dropout(drop_prob),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )
        if in_channels != out_channels:
            self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        else:
            self.conv = nn.Identity()
            
        if attn:
            self.attn = ResConvSelfAttnBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x, temb):
        x1 = self.layers1(x)
        # "Diffusion time $t$ is specified by adding the Transformer sinusoidal position embedding
        # into each residual block."
        # "We condition all layers on $t$ by adding in the Transformer sinusoidal position embedding."
        x1 = x1 + self.time_proj(temb)[:, :, None, None]
        x1 = self.layers(x1)
        x = x1 + self.conv(x)
        x = self.attn(x)
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


class OldUNet(nn.Module):
    def __init__(
        self,
        channels=128,
        channel_mults=(1, 2, 2, 2),
        # "All models have two convolutional residual blocks per resolution level."
        attns=(False, True, False, False),
        n_res_blocks=2,
        drop_prob=0.1,
        n_groups=32,
        n_diffusion_steps=1000,
    ):
        super().__init__()

        assert len(attns) == len(channel_mults)

        time_channels = channels * 4
        self.time_embedding = TimeEmbedding(
            n_diffusion_steps=n_diffusion_steps, time_channels=time_channels,
        )

        self.init_conv = nn.Conv2d(3, channels, 3, 1, 1)
        self.downblocks = nn.ModuleList()
        cxs = [channels]  # record output channel when dowmsample for upsample
        cur_channels = channels
        for i, mult in enumerate(channel_mults):
            out_channels = channels * mult
            for _ in range(n_res_blocks):
                # print("Res", cur_channels, out_channels)
                self.downblocks.append(
                    ResBlock(
                        in_channels=cur_channels,
                        out_channels=out_channels,
                        time_channels=time_channels,
                        drop_prob=drop_prob,
                        attn=attns[i],
                    )
                )
                cur_channels = out_channels
                cxs.append(cur_channels)
            if i != len(channel_mults) - 1:
                # print("Down", cur_channels)
                self.downblocks.append(Downsample(cur_channels))
                cxs.append(cur_channels)

        self.middleblocks = nn.ModuleList([
            ResBlock(cur_channels, cur_channels, time_channels, drop_prob, attn=True),
            ResBlock(cur_channels, cur_channels, time_channels, drop_prob, attn=False),
        ])
        # print("Mid")

        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = channels * mult
            for _ in range(n_res_blocks + 1):
                tt = cxs.pop() + cur_channels
                # print("Res", tt, out_channels)
                self.up_blocks.append(
                    ResBlock(
                        in_channels=cxs.pop() + cur_channels,
                        # in_channels=tt,
                        out_channels=out_channels,
                        time_channels=time_channels,
                        drop_prob=drop_prob,
                        attn=attns[i],
                    )
                )
                cur_channels = out_channels
            if i != 0:
                # print("Up", cur_channels)
                self.up_blocks.append(Upsample(cur_channels))
        assert len(cxs) == 0

        self.fin_block = nn.Sequential(
            nn.GroupNorm(n_groups, cur_channels),
            Swish(),
            nn.Conv2d(cur_channels, 3, 3, 1, 1)
        )

    def forward(self, noisy_image, diffusion_step):
        temb = self.time_embedding(diffusion_step)
        x = self.init_conv(noisy_image)
        xs = [x]
        for layer in self.downblocks:
            if isinstance(layer, Downsample):
                x = layer(x)
            else:
                x = layer(x, temb)
            # print(x.shape)
            xs.append(x)

        for layer in self.middleblocks:
            x = layer(x, temb)
            # print(x.shape)

        for layer in self.up_blocks:
            if isinstance(layer, Upsample):
                x = layer(x)
            else:
                x = torch.cat([x, xs.pop()], dim=1)
                x = layer(x, temb)
                # print(x.shape)
        x = self.fin_block(x)
        # print(x.shape)
        assert len(xs) == 0
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
        channels,
        channel_mults,
        attns,
        n_res_blocks,
        device,
        img_channels=3,
        n_diffusion_steps=1000,
        init_beta=0.0001,
        fin_beta=0.02,
    ):
        super().__init__()

        self.img_size = img_size
        self.device = device
        self.img_channels = img_channels
        self.n_diffusion_steps = n_diffusion_steps
        self.init_beta = init_beta
        self.fin_beta = fin_beta

        self.beta = self.get_linear_beta_schdule()
        self.alpha = 1 - self.beta # "$\alpha_{t} = 1 - \beta_{t}$"
        # "$\bar{\alpha_{t}} = \prod^{t}_{s=1}{\alpha_{s}}$"
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        self.net = OldUNet(
            channels=channels,
            channel_mults=channel_mults,
            attns=attns,
            # "All models have two convolutional residual blocks per resolution level."
            n_res_blocks=n_res_blocks,
        ).to(device)

    @staticmethod
    def index(x, diffusion_step):
        return torch.index_select(x, dim=0, index=diffusion_step)[:, None, None, None]

    def sample_noise(self, batch_size):
        return torch.randn(
            size=(batch_size, self.img_channels, self.img_size, self.img_size),
            device=self.device,
        )

    def sample_diffusion_step(self, batch_size):
        return torch.randint(
            0, self.n_diffusion_steps, size=(batch_size,), device=self.device,
        )
        # return torch.randint(
        #     0, self.n_diffusion_steps, size=(1,), device=self.device,
        # ).repeat(batch_size)

    def batchify_diffusion_steps(self, cur_diffusion_step, batch_size):
        return torch.full(
            size=(batch_size,),
            fill_value=cur_diffusion_step,
            dtype=torch.long,
            device=self.device,
        )  

    def perform_diffusion_process(self, ori_image, diffusion_step, random_noise=None):
        # "$\bar{\alpha_{t}}$"
        alpha_bar_t = self.index(self.alpha_bar, diffusion_step=diffusion_step)
        mean = (alpha_bar_t ** 0.5) * ori_image # $\sqrt{\bar{\alpha_{t}}}x_{0}$
        var = 1 - alpha_bar_t # $(1 - \bar{\alpha_{t}})\mathbf{I}$
        if random_noise is None:
            random_noise = self.sample_noise(batch_size=ori_image.size(0))
        noisy_image = mean + (var ** 0.5) * random_noise
        return noisy_image

    def forward(self, noisy_image, diffusion_step):
        # return self.net(noisy_image=noisy_image, diffusion_step=diffusion_step)
        return self.net(noisy_image, diffusion_step)

    def get_loss(self, ori_image):
        # "Algorithm 1-3: $t \sim Uniform(\{1, \ldots, T\})$"
        diffusion_step = self.sample_diffusion_step(batch_size=ori_image.size(0))
        random_noise = self.sample_noise(batch_size=ori_image.size(0))
        # "Algorithm 1-4: $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$"
        noisy_image = self.perform_diffusion_process(
            ori_image=ori_image, diffusion_step=diffusion_step, random_noise=random_noise,
        )
        pred_noise = self(noisy_image=noisy_image, diffusion_step=diffusion_step)
        # recon_image = self.reconstruct(
        #     noisy_image=noisy_image, noise=pred_noise.detach(), diffusion_step=diffusion_step,
        # )
        # image_to_grid(recon_image, n_cols=int(recon_image.size(0) ** 0.5)).show()
        return F.mse_loss(pred_noise, random_noise, reduction="mean")

    @torch.inference_mode()
    def reconstruct(self, noisy_image, noise, diffusion_step):
        alpha_bar_t = self.index(self.alpha_bar, diffusion_step=diffusion_step)
        return (noisy_image - ((1 - alpha_bar_t) ** 0.5) * noise) / (alpha_bar_t ** 0.5)

    @torch.inference_mode()
    def take_denoising_step(self, noisy_image, cur_diffusion_step):
        diffusion_step = self.batchify_diffusion_steps(
            cur_diffusion_step=cur_diffusion_step, batch_size=noisy_image.size(0),
        )
        alpha_t = self.index(self.alpha, diffusion_step=diffusion_step)
        beta_t = self.index(self.beta, diffusion_step=diffusion_step)
        alpha_bar_t = self.index(self.alpha_bar, diffusion_step=diffusion_step)
        pred_noise = self(noisy_image=noisy_image.detach(), diffusion_step=diffusion_step)
        # # "Algorithm 2-4:
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
    def perform_denoising_process(self, noisy_image, cur_diffusion_step):
        x = noisy_image
        pbar = tqdm(range(cur_diffusion_step, -1, -1), leave=False)
        for trg_diffusion_step in pbar:
            pbar.set_description("Denoising...")

            x = self.take_denoising_step(x, cur_diffusion_step=trg_diffusion_step)
        return x

    @torch.inference_mode()
    def sample(self, batch_size):
        random_noise = self.sample_noise(batch_size=batch_size) # "$x_{T}$"
        return self.perform_denoising_process(
            noisy_image=random_noise, cur_diffusion_step=self.n_diffusion_steps - 1,
        )


if __name__ == "__main__":
    old = OldUNet()
    print_n_params(old)

    x = torch.randn(1, 3, 32, 32)
    t = torch.randint(0, 1000, (1,))
    old(x, t)
