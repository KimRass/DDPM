# References:
    # https://medium.com/mlearning-ai/enerating-images-with-ddpms-a-pytorch-implementation-cef5a2ba8cb1

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self, shape, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activ=None, normalize=True,
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


def sinusoidal_embedding(n_timesteps, time_embed_dim):
    pos = torch.arange(n_timesteps).unsqueeze(1)
    i = torch.arange(time_embed_dim // 2).unsqueeze(0)
    angle = pos / (10_000 ** (2 * i / time_embed_dim))

    pe_mat = torch.zeros(size=(n_timesteps, time_embed_dim))
    pe_mat[:, 0:: 2] = torch.sin(angle)
    pe_mat[:, 1:: 2] = torch.cos(angle)
    return pe_mat


class UNetForDDPM(nn.Module):

    def __init__(self, n_timesteps, time_emb_dim=100):
        super().__init__()

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_timesteps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_timesteps, time_emb_dim)
        self.time_embed.requires_grad = False

        # First half
        self.te1 = self._make_time_embedding(time_emb_dim, 1)
        self.b1 = nn.Sequential(
            ConvBlock((1, 28, 28), 1, 10),
            ConvBlock((10, 28, 28), 10, 10),
            ConvBlock((10, 28, 28), 10, 10)
        )
        self.down1 = nn.Conv2d(10, 10, 4, 2, 1)

        self.te2 = self._make_time_embedding(time_emb_dim, 10)
        self.b2 = nn.Sequential(
            ConvBlock((10, 14, 14), 10, 20),
            ConvBlock((20, 14, 14), 20, 20),
            ConvBlock((20, 14, 14), 20, 20)
        )
        self.down2 = nn.Conv2d(20, 20, 4, 2, 1)

        self.te3 = self._make_time_embedding(time_emb_dim, 20)
        self.b3 = nn.Sequential(
            ConvBlock((20, 7, 7), 20, 40),
            ConvBlock((40, 7, 7), 40, 40),
            ConvBlock((40, 7, 7), 40, 40)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(40, 40, 2, 1),
            nn.SiLU(),
            nn.Conv2d(40, 40, 4, 2, 1)
        )

        # Bottleneck
        self.te_mid = self._make_time_embedding(time_emb_dim, 40)
        self.b_mid = nn.Sequential(
            ConvBlock((40, 3, 3), 40, 20),
            ConvBlock((20, 3, 3), 20, 20),
            ConvBlock((20, 3, 3), 20, 40)
        )

        # Second half
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(40, 40, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(40, 40, 2, 1)
        )

        self.te4 = self._make_time_embedding(time_emb_dim, 80)
        self.b4 = nn.Sequential(
            ConvBlock((80, 7, 7), 80, 40),
            ConvBlock((40, 7, 7), 40, 20),
            ConvBlock((20, 7, 7), 20, 20)
        )

        self.up2 = nn.ConvTranspose2d(20, 20, 4, 2, 1)
        self.te5 = self._make_time_embedding(time_emb_dim, 40)
        self.b5 = nn.Sequential(
            ConvBlock((40, 14, 14), 40, 20),
            ConvBlock((20, 14, 14), 20, 10),
            ConvBlock((10, 14, 14), 10, 10)
        )

        self.up3 = nn.ConvTranspose2d(10, 10, 4, 2, 1)
        self.te_out = self._make_time_embedding(time_emb_dim, 20)
        self.b_out = nn.Sequential(
            ConvBlock((20, 28, 28), 20, 10),
            ConvBlock((10, 28, 28), 10, 10),
            ConvBlock((10, 28, 28), 10, 10, normalize=False)
        )

        self.conv_out = nn.Conv2d(10, 1, 3, 1, 1)

    # `x`: (b, 1, 28, 28), `t`: (b, 1)
    def forward(self, x, t):
        b, _, _, _ = x.shape
        t = self.time_embed(t)

        x1 = self.b1(x + self.te1(t).reshape(b, -1, 1, 1))  # (N, 10, 28, 28)
        x2 = self.b2(self.down1(x1) + self.te2(t).reshape(b, -1, 1, 1))  # (N, 20, 14, 14)
        x3 = self.b3(self.down2(x2) + self.te3(t).reshape(b, -1, 1, 1))  # (N, 40, 7, 7)

        out_mid = self.b_mid(self.down3(x3) + self.te_mid(t).reshape(b, -1, 1, 1))  # (N, 40, 3, 3)

        x4 = torch.cat((x3, self.up1(out_mid)), dim=1)  # (N, 80, 7, 7)
        x4 = self.b4(x4 + self.te4(t).reshape(b, -1, 1, 1))  # (N, 20, 7, 7)

        x5 = torch.cat((x2, self.up2(x4)), dim=1)  # (N, 40, 14, 14)
        x5 = self.b5(x5 + self.te5(t).reshape(b, -1, 1, 1))  # (N, 10, 14, 14)

        x = torch.cat((x1, self.up3(x5)), dim=1)  # (N, 20, 28, 28)
        x = self.b_out(x + self.te_out(t).reshape(b, -1, 1, 1))  # (N, 1, 28, 28)

        x = self.conv_out(x)
        return x

    def _make_time_embedding(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )

if __name__ == "__main__":
    model = UNetForDDPM(n_timesteps=200, time_emb_dim=100)
    x = torch.randn(16, 2, 28, 28)
    t = torch.tensor(30)
    model(x, t=t)