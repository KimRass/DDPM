# References:
    # https://github.com/KimRass/Transformer/blob/main/model.py
    # https://nn.labml.ai/diffusion/ddpm/unet.html
    # https://nn.labml.ai/diffusion/ddpm/index.html
    # https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/blob/main/notebooks/08_diffusion/01_ddm/ddm.ipynb
    # https://huggingface.co/blog/annotated-diffusion

import torch
from torch import nn
from torch.nn import functional as F


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
    def __init__(self, in_channels, out_channels, time_channels, attn=False, n_groups=32, drop_prob=0.1):
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
        # "Our 32 × 32 models use four feature map resolutions (32 × 32 to 4 × 4),
        # and our 256 × 256 models use six."
        # 4 8 16 32: 4
        # 8 16 32 64: 4
        # 8 16 32 64 128 256: 6
        init_channels,
        channels,
        # "All models have self-attention blocks at the 16 × 16 resolution
        # between the convolutional blocks."
        # "We use self-attention at the 16 × 16 feature map resolution."
        attns,
        # "All models have two convolutional residual blocks per resolution level."
        n_blocks=2,
        n_groups=32,
        n_diffusion_steps=1000,
    ):
        super().__init__()

        assert len(attns) == len(channels)

        self.n_diffusion_steps = n_diffusion_steps

        self.init_conv = nn.Conv2d(3, init_channels, 3, 1, 1)

        self.time_channels = init_channels * 4
        self.time_embed = TimeEmbedding(
            n_diffusion_steps=n_diffusion_steps, time_channels=self.time_channels,
        )

        channels = (init_channels, *channels)
        self.down_block = nn.ModuleList()
        for idx in range(len(channels) - 1):
            in_channels = channels[idx]
            out_channels = channels[idx + 1]
            attn=attns[idx]
            for _ in range(n_blocks):
                # print("Res", in_channels, out_channels)
                self.down_block.append(
                    ResBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        time_channels=self.time_channels,
                        attn=attn,
                        n_groups=n_groups,
                    )
                )
                in_channels = out_channels

            if idx < len(channels) - 2:
                # print("Down", out_channels)
                self.down_block.append(Downsample(out_channels))

        self.mid_block = nn.ModuleList(
            [
                ResBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    time_channels=self.time_channels,
                    attn=True,
                    n_groups=n_groups,
                ),
                ResBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    time_channels=self.time_channels,
                    attn=False,
                    n_groups=n_groups,
                ),
            ]
        )
        # print("Mid")

        self.up_block = nn.ModuleList()
        for idx in list(reversed(range(1, len(channels)))):
            out_channels = in_channels
            attn = attns[idx - 1]
            for _ in range(n_blocks):
                # print("Res", in_channels, out_channels)
                self.up_block.append(
                    ResBlock(
                        in_channels=in_channels + out_channels,
                        out_channels=out_channels,
                        time_channels=self.time_channels,
                        attn=attn,
                        n_groups=n_groups,
                    )
                )
            in_channels = channels[idx]
            out_channels = channels[idx - 1]
            # print("Res", in_channels, out_channels)
            self.up_block.append(
                ResBlock(
                    in_channels=in_channels + out_channels,
                    out_channels=out_channels,
                    time_channels=self.time_channels,
                    attn=attn,
                    n_groups=n_groups,
                )
            )
            in_channels = out_channels

            if idx > 1:
                # print("Up", out_channels)
                self.up_block.append(Upsample(out_channels))

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
        for layer in self.down_block:
            if isinstance(layer, Downsample):
                x = layer(x)
            else:
                x = layer(x, t)
            # print(x.shape)
            xs.append(x)

        for layer in self.mid_block:
            x = layer(x, t)
        # print(x.shape)

        for layer in self.up_block:
            if isinstance(layer, Upsample):
                x = layer(x)
            else:
                x = torch.cat([x, xs.pop()], dim=1)
                x = layer(x, t)
            # print(x.shape)
        assert len(xs) == 0

        x = self.fin_block(x)
        # print(x.shape)
        return x


if __name__ == "__main__":
    new = UNet(
        n_diffusion_steps=1000,
        init_channels=128,
        channels=(128, 256, 256, 256),
        attns=(True, True, True, True),
        n_blocks=2,
    )
    # print_n_params(new)
    x = torch.randn(1, 3, 32, 32)
    t = torch.randint(0, 1000, (1,))
    new(x, t)
