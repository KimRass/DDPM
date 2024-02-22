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

        self.n_diffusion_steps = n_diffusion_steps

        time_channels = channels * 4
        self.time_embedding = TimeEmbedding(
            n_diffusion_steps=n_diffusion_steps, time_channels=time_channels,
        )

        self.init_conv = nn.Conv2d(3, channels, 3, 1, 1)
        self.downblocks = nn.ModuleList()
        channels_ls = [channels]
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
                channels_ls.append(cur_channels)
            if i != len(channel_mults) - 1:
                # print("Down", cur_channels)
                self.downblocks.append(Downsample(cur_channels))
                channels_ls.append(cur_channels)

        self.middleblocks = nn.ModuleList([
            ResBlock(cur_channels, cur_channels, time_channels, drop_prob, attn=True),
            ResBlock(cur_channels, cur_channels, time_channels, drop_prob, attn=False),
        ])
        # print("Mid")

        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = channels * mult
            for _ in range(n_res_blocks + 1):
                # tt = channels_ls.pop() + cur_channels
                # print("Res", tt, out_channels)
                self.up_blocks.append(
                    ResBlock(
                        in_channels=channels_ls.pop() + cur_channels,
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
        assert len(channels_ls) == 0

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


if __name__ == "__main__":
    old = OldUNet()
    print_n_params(old)

    x = torch.randn(1, 3, 32, 32)
    t = torch.randint(0, 1000, (1,))
    old(x, t)
