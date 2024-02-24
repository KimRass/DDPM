# References:
    # https://github.com/w86763777/pytorch-ddpm/blob/master/model.py

import torch
from torch import nn
from torch.nn import functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    # "Parameters are shared across time, which is specified to the network using the Transformer
    # sinusoidal position embedding."
    def __init__(self, time_channels, max_len=4000):
        super().__init__()

        self.d_model = time_channels // 4

        pos = torch.arange(max_len).unsqueeze(1)
        i = torch.arange(self.d_model // 2).unsqueeze(0)
        angle = pos / (10_000 ** (2 * i / self.d_model))

        self.pe_mat = torch.zeros(size=(max_len, self.d_model))
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
    def __init__(
        self, in_channels, out_channels, time_channels, attn=False, n_groups=32, drop_prob=0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attn = attn

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
        else:
            self.conv = nn.Identity()

        if attn:
            self.attn_block = ResConvSelfAttnBlock(out_channels)
        else:
            self.attn_block = nn.Identity()

    def forward(self, x, t):
        skip = x
        x = self.layers1(x)
        # "Diffusion time $t$ is specified by adding the Transformer sinusoidal position embedding
        # into each residual block."
        # "We condition all layers on $t$ by adding in the Transformer sinusoidal position embedding."
        x = x + self.time_proj(t)[:, :, None, None]
        x = self.layers2(x)
        x = x + self.conv(skip)
        return self.attn_block(x)


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
        channels=128,
        channel_mults=[1, 2, 2, 2],
        attns=[False, True, False, False],
        n_res_blocks=2,
        n_groups=32,
        drop_prob=0.1,
    ):
        super().__init__()

        assert len(channel_mults) == len(attns)

        self.time_channels = channels * 4
        self.time_embed = TimeEmbedding(time_channels=self.time_channels)

        self.init_conv = nn.Conv2d(3, channels, 3, 1, 1)
        self.down_blocks = nn.ModuleList()
        cxs = [channels]  # record output channel when dowmsample for upsample
        cur_channels = channels
        for i, mult in enumerate(channel_mults):
            out_channels = channels * mult
            for _ in range(n_res_blocks):
                self.down_blocks.append(
                    ResBlock(
                        in_channels=cur_channels,
                        out_channels=out_channels,
                        time_channels=self.time_channels,
                        attn=attns[i],
                        n_groups=n_groups,
                        drop_prob=drop_prob,
                    )
                )
                cur_channels = out_channels
                cxs.append(cur_channels)
            if i != len(channel_mults) - 1:
                self.down_blocks.append(Downsample(cur_channels))
                cxs.append(cur_channels)

        self.mid_blocks = nn.ModuleList(
            [
                ResBlock(
                    in_channels=cur_channels,
                    out_channels=cur_channels,
                    time_channels=self.time_channels,
                    attn=True,
                    n_groups=n_groups,
                    drop_prob=drop_prob,
                ),
                ResBlock(
                    in_channels=cur_channels,
                    out_channels=cur_channels,
                    time_channels=self.time_channels,
                    attn=False,
                    n_groups=n_groups,
                    drop_prob=drop_prob,
                ),
            ]
        )

        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = channels * mult
            for _ in range(n_res_blocks + 1):
                self.up_blocks.append(
                    ResBlock(
                        in_channels=cxs.pop() + cur_channels,
                        out_channels=out_channels,
                        time_channels=self.time_channels,
                        attn=attns[i],
                        n_groups=n_groups,
                        drop_prob=drop_prob,
                    )
                )
                cur_channels = out_channels
            if i != 0:
                self.up_blocks.append(Upsample(cur_channels))
        assert len(cxs) == 0

        self.fin_block = nn.Sequential(
            nn.GroupNorm(n_groups, cur_channels),
            Swish(),
            nn.Conv2d(cur_channels, 3, 3, 1, 1)
        )

    def forward(self, noisy_image, diffusion_step):
        x = self.init_conv(noisy_image)
        t = self.time_embed(diffusion_step)

        xs = [x]
        for layer in self.down_blocks:
            if isinstance(layer, Downsample):
                x = layer(x)
            else:
                x = layer(x, t)
            xs.append(x)

        for layer in self.mid_blocks:
            x = layer(x, t)

        for layer in self.up_blocks:
            if isinstance(layer, Upsample):
                x = layer(x)
            else:
                x = torch.cat([x, xs.pop()], dim=1)
                x = layer(x, t)

        x = self.fin_block(x)
        assert len(xs) == 0
        return x
