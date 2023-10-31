# References:
    # https://medium.com/mlearning-ai/enerating-images-with-ddpms-a-pytorch-implementation-cef5a2ba8cb1
    # https://nn.labml.ai/diffusion/ddpm/unet.html

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, shape, kernel_size=3, stride=1, padding=1, activ=None, normalize=True,
    ):
        super().__init__()

        self.normalize = normalize

        if normalize:
            self.norm = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.activ = nn.SiLU() if activ is None else activ

    def forward(self, x):
        if self.normalize:
            x = self.norm(x)
        x = self.conv1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.activ(x)
        return x


class TimeEmbedding(nn.Module):
    def __init__(self, n_timesteps, dim):
        super().__init__()

        self.n_timesteps = n_timesteps
        self.dim = dim

        self.time_embed = nn.Embedding(n_timesteps, dim)
        self.time_embed.weight.data = self._sinusoidal_embedding()
        self.time_embed.requires_grad = False

    def _sinusoidal_embedding(self):
        pos = torch.arange(self.n_timesteps).unsqueeze(1)
        i = torch.arange(self.dim // 2).unsqueeze(0)
        angle = pos / (10_000 ** (2 * i / self.dim))

        pe_mat = torch.zeros(size=(self.n_timesteps, self.dim))
        pe_mat[:, 0:: 2] = torch.sin(angle)
        pe_mat[:, 1:: 2] = torch.cos(angle)
        return pe_mat

    def forward(self, x):
        return self.time_embed(x)


class MLPBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.proj1 = nn.Linear(in_features, out_features)
        self.silu = nn.SiLU()
        self.proj2 = nn.Linear(out_features, out_features)

    def forward(self, x):
        x = self.proj1(x)
        x = self.silu(x)
        x = self.proj2(x)
        return x


class UNetForDDPM(nn.Module):
    def __init__(self, n_timesteps, time_embed_dim=100):
        super().__init__()

        # Sinusoidal embedding
        self.time_embed = TimeEmbedding(n_timesteps=n_timesteps, dim=time_embed_dim)

        self.mlp_block1 = MLPBlock(time_embed_dim, 1)
        self.mlp_block2 = MLPBlock(time_embed_dim, 10)
        self.mlp_block3 = MLPBlock(time_embed_dim, 20)
        self.mid_mlp_block = MLPBlock(time_embed_dim, 40)
        self.mlp_block4 = MLPBlock(time_embed_dim, 80)
        self.mlp_block5 = MLPBlock(time_embed_dim, 40)
        self.mlp_block6 = MLPBlock(time_embed_dim, 20)

        # First half
        self.down1 = nn.Conv2d(10, 10, 4, 2, 1)
        self.down2 = nn.Conv2d(20, 20, 4, 2, 1)
        self.down3 = nn.Sequential(
            nn.Conv2d(40, 40, 2, 1),
            nn.SiLU(),
            nn.Conv2d(40, 40, 4, 2, 1),
        )

        self.b1 = nn.Sequential(
            ConvBlock(1, 10, shape=(1, 28, 28)),
            ConvBlock(10, 10, shape=(10, 28, 28)),
            ConvBlock(10, 10, shape=(10, 28, 28)),
        )
        self.b2 = nn.Sequential(
            ConvBlock(10, 20, shape=(10, 14, 14)),
            ConvBlock(20, 20, shape=(20, 14, 14)),
            ConvBlock(20, 20, shape=(20, 14, 14)),
        )
        self.b3 = nn.Sequential(
            ConvBlock(20, 40, shape=(20, 7, 7),),
            ConvBlock(40, 40, shape=(40, 7, 7),),
            ConvBlock(40, 40, shape=(40, 7, 7),),
        )

        # Bottleneck
        self.b_mid = nn.Sequential(
            ConvBlock(40, 20, shape=(40, 3, 3)),
            ConvBlock(20, 20, shape=(20, 3, 3)),
            ConvBlock(20, 40, shape=(20, 3, 3)),
        )

        # Second half
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(40, 40, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(40, 40, 2, 1),
        )
        self.up2 = nn.ConvTranspose2d(20, 20, 4, 2, 1)
        self.up3 = nn.ConvTranspose2d(10, 10, 4, 2, 1)

        self.b4 = nn.Sequential(
            ConvBlock(80, 40, shape=(80, 7, 7)),
            ConvBlock(40, 20, shape=(40, 7, 7)),
            ConvBlock(20, 20, shape=(20, 7, 7)),
        )
        self.b5 = nn.Sequential(
            ConvBlock(40, 20, shape=(40, 14, 14)),
            ConvBlock(20, 10, shape=(20, 14, 14)),
            ConvBlock(10, 10, shape=(10, 14, 14)),
        )
        self.b_out = nn.Sequential(
            ConvBlock(20, 10, shape=(20, 28, 28)),
            ConvBlock(10, 10, shape=(10, 28, 28)),
            ConvBlock(10, 10, shape=(10, 28, 28), normalize=False),
        )

        self.conv_out = nn.Conv2d(10, 1, 3, 1, 1)

    # `x`: (b, 1, 28, 28), `t`: (b, 1)
    def forward(self, x, t):
        b, _, _, _ = x.shape
        t = self.time_embed(t)

        x1 = self.b1(x + self.mlp_block1(t).reshape(b, -1, 1, 1))  # (N, 10, 28, 28)
        x2 = self.b2(self.down1(x1) + self.mlp_block2(t).reshape(b, -1, 1, 1))  # (N, 20, 14, 14)
        x3 = self.b3(self.down2(x2) + self.mlp_block3(t).reshape(b, -1, 1, 1))  # (N, 40, 7, 7)

        x_mid = self.b_mid(self.down3(x3) + self.mid_mlp_block(t).reshape(b, -1, 1, 1))  # (N, 40, 3, 3)

        x4 = torch.cat((x3, self.up1(x_mid)), dim=1)  # (N, 80, 7, 7)
        x4 = self.b4(x4 + self.mlp_block4(t).reshape(b, -1, 1, 1))  # (N, 20, 7, 7)

        x5 = torch.cat((x2, self.up2(x4)), dim=1)  # (N, 40, 14, 14)
        x5 = self.b5(x5 + self.mlp_block5(t).reshape(b, -1, 1, 1))  # (N, 10, 14, 14)

        x = torch.cat((x1, self.up3(x5)), dim=1)  # (N, 20, 28, 28)
        x = self.b_out(x + self.mlp_block6(t).reshape(b, -1, 1, 1))  # (N, 1, 28, 28)

        x = self.conv_out(x)
        return x


if __name__ == "__main__":
    model = UNetForDDPM(n_timesteps=200, time_embed_dim=100)
    x = torch.randn(4, 1, 28, 28)
    t = torch.full(size=(4, 1), fill_value=30, dtype=torch.long)
    out = model(x, t=t)
    out.shape
