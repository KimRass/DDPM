# References:
    # https://github.com/KimRass/Transformer/blob/main/model.py
    # https://nn.labml.ai/diffusion/ddpm/unet.html

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
import numpy as np
import imageio
import math
from tqdm import tqdm
from pathlib import Path

from utils import image_to_grid, save_image


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# "We use self-attention at the 16 × 16 feature map resolution."

# "We replaced weight normalization with group normalization to make the implementation simpler."

# "Our 32 × 32 models use four feature map resolutions (32 × 32 to 4 × 4),
# and our 256 × ×256 models use six."

# 256 128 64 32 16 8
# 32 16 8 4

# Our CIFAR10 model has 35.7 million parameters, and our LSUN and CelebA-HQ models have 114 million parameters. We also trained a larger variant of the LSUN Bedroom model with approximately 256 million parameters by increasing filter count."

# "All models have two convolutional residual blocks per resolution level
# and self-attention blocks at the 16 × 16 resolution between the convolutional blocks."


class SelfAttnBlock(nn.Module):
    def __init__(self, dim, n_heads=1):
        super().__init__()
    
        self.dim = dim
        self.n_heads = n_heads

        self.init_conv_dim = dim // n_heads

        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def _rearrange(self, x):
        return rearrange(x, pattern="b n (h d) -> b h n d", h=self.n_heads, d=self.init_conv_dim)

    def forward(self, x):
        ori_shape = x.shape

        x = rearrange(x, pattern="b c h w -> b (h w) c")
        q, k, v = torch.chunk(self.qkv_proj(x), chunks=3, dim=2)
        q = self._rearrange(q)
        k = self._rearrange(k)
        v = self._rearrange(v)

        attn_score = torch.einsum("bnid,bnjd->bnij", q, k)
        attn_score /= (self.init_conv_dim ** 0.5)
        attn_weight = F.softmax(attn_score, dim=3)

        x = torch.einsum("bnij,bnjd->bnid", attn_weight, v)
        x = rearrange(x, pattern="b n i d -> b i (n d)")

        x = self.out_proj(x)
        return x.view(ori_shape), attn_weight


class TimeEmbedder(nn.Module):
    # "Diffusion time $t$ is specified by adding the Transformer sinusoidal position embedding
    # into each residual block."
    # "Parameters are shared across time, which is specified to the network using the Transformer
    # sinusoidal position embedding."
    # "We condition all layers on $t$ by adding in the Transformer sinusoidal position embedding,"
    def __init__(self, n_diffusion_steps, time_dim):
        super().__init__()

        pos = torch.arange(n_diffusion_steps).unsqueeze(1)
        i = torch.arange(time_dim // 2).unsqueeze(0)
        angle = pos / (10_000 ** (2 * i / time_dim))

        self.pe_mat = torch.zeros(size=(n_diffusion_steps, time_dim))
        self.pe_mat[:, 0:: 2] = torch.sin(angle)
        self.pe_mat[:, 1:: 2] = torch.cos(angle)

        self.register_buffer("pos_enc_mat", self.pe_mat)

    def forward(self, diffusion_step):
        return self.pe_mat[diffusion_step, :]        


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, drop_prob, n_gropus=32):
        super().__init__()
        self.layers1 = nn.Sequential(
            nn.GroupNorm(num_groups=n_gropus, num_channels=in_channels),
            Swish(),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
        )
        self.time_proj = nn.Sequential(
            Swish(),
            nn.Linear(time_dim, out_channels),
        )
        self.layers2 = nn.Sequential(
            nn.GroupNorm(num_groups=n_gropus, num_channels=in_channels),
            Swish(),
            nn.Dropout(drop_prob),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()


    def forward(self, x, t):
        x1 = self.layers1(x)
        x1 = x1 + self.time_proj(t)[:, :, None, None]
        x1 = self.layers2(x1)
        x2 = self.shortcut(x)
        return x1 + x2


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, attn):
        super().__init__()

        self.attn = attn

        self.layers = nn.ModuleList()
        self.layers.append(
            ResBlock(in_channels=in_channels, out_channels=out_channels, time_dim=time_dim)
        )
        if attn:
            self.layers.append(SelfAttnBlock(dim=out_channels))

    def forward(self, x):
        return self.layers(x)


class MidBlock(nn.Module):
    def __init__(self, channels, time_dim):
        super().__init__()

        self.layers = nn.Sequential(
            ResBlock(
                in_channels=channels, out_channels=channels, time_dim=time_dim,
            ),
            SelfAttnBlock(dim=channels),
            ResBlock(
                in_channels=channels, out_channels=channels, time_dim=time_dim,
            ),
        )

    def forward(self, x):
        return self.layers(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, attn):
        super().__init__()

        self.attn = attn

        self.layers = nn.ModuleList()
        self.layers.append(
            ResBlock(
                in_channels=in_channels + out_channels,
                out_channels=out_channels,
                time_dim=time_dim,
            )
        )
        if attn:
            self.layers.append(SelfAttnBlock(dim=out_channels))

    def forward(self, x):
        return self.layers(x)


class Downsample(nn.Conv2d):
    def __init__(self, channels):
        super().__init__(channels, channels, 3, 2, 1)


# class Upsample(nn.ConvTranspose2d):
#     def __init__(self, channels):
#         super().__init__(channels, channels, 4, 2, 1)
class UpSample(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(channels, channels, 3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    def __init__(
        self,
        n_diffusion_steps=1000,
        channels=128,
        channel_mults=[1, 2, 2, 4],
        attns=[False, False, True, True],
        n_blocks=2,
        drop_prob=0.1,
    ):
        super().__init__()

        assert all([i < len(channel_mults) for i in attns]), "attn index out of bound"

        self.init_conv = nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1)

        time_dim = channels * 4
        self.time_embed = TimeEmbedder(
            n_diffusion_steps=n_diffusion_steps, dim=time_dim,
        )

        self.down_blocks = nn.ModuleList()
        in_channels = channels
        for idx, (channel_mult, attn) in enumerate(zip(channel_mults, attns)):
            out_channels = in_channels * channel_mult
            for _ in range(n_blocks):
                self.down_blocks.append(
                    ResBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        time_dim=time_dim,
                        drop_prob=drop_prob,
                        attn=attn,
                    )
                )
                in_channels = out_channels

            if idx < len(channel_mults) - 1:
                self.down_blocks.append(Downsample(in_channels))

        # self.mid_blocks = nn.ModuleList([
        #     ResBlock(cur_channels, cur_channels, time_dim, drop_prob, attn=True),
        #     ResBlock(cur_channels, cur_channels, time_dim, drop_prob, attn=False),
        # ])

        # self.up_blocks = nn.ModuleList()
        # for i, mult in reversed(list(enumerate(channel_mults))):
        #     out_ch = channels * mult
        #     for _ in range(n_blocks + 1):
        #         self.up_blocks.append(ResBlock(
        #             in_channels=cxs.pop() + cur_channels, out_ch=out_ch, time_dim=time_dim,
        #             drop_prob=drop_prob, attn=(i in attns)))
        #         cur_channels = out_ch
        #     if i != 0:
        #         self.up_blocks.append(UpSample(cur_channels))
        # assert len(cxs) == 0

        # self.tail = nn.Sequential(
        #     nn.GroupNorm(32, cur_channels),
        #     Swish(),
        #     nn.Conv2d(cur_channels, 3, kernel_size=3, stride=1, padding=1)
        # )

    def forward(self, noisy_image, diffusion_step):
        x = self.init_conv(noisy_image)
        t = self.time_embed(diffusion_step)

        xs = [x]
        for layer in self.down_blocks:
            x = layer(x, t)
            xs.append(x)

        # for layer in self.mid_blocks:
        #     x = layer(x, t)

        # for layer in self.up_blocks:
        #     if isinstance(layer, ResBlock):
        #         x = torch.cat([x, xs.pop()], dim=1)

        #     if isinstance(layer, UpSample):
        #         x = layer(x)
        #     else:
        #         x = layer(x, t)
        # x = self.tail(x)
        # assert len(xs) == 0
        return x
model = UNet()


class DDPM(nn.Module):
    # "We set T = 1000 without a sweep."
    # "We chose a linear schedule from $\beta_{1} = 10^{-4}$ to  $\beta_{T} = 0:02$."
    def __init__(self, device, n_diffusion_steps=1000, init_beta=0.0001, fin_beta=0.02):
        super().__init__()

        self.device = device
        self.n_diffusion_steps = n_diffusion_steps
        self.init_beta = init_beta
        self.fin_beta = fin_beta

        self.beta = self.get_linear_beta_schdule()
        self.alpha = 1 - self.beta # "$\alpha_{t} = 1 - \beta_{t}$"
        # "$\bar{\alpha_{t}} = \prod^{t}_{s=1}{\alpha_{s}}$"
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        self.net = UNet(n_diffusion_steps=n_diffusion_steps).to(device)

    def get_linear_beta_schdule(self):
        # "We set the forward process variances to constants increasing linearly."
        # return torch.linspace(init_beta, fin_beta, n_diffusion_steps) # "$\beta_{t}$"
        return torch.linspace(self.init_beta, self.fin_beta, self.n_diffusion_steps + 1, device=self.device) # "$\beta_{t}$"

    @staticmethod
    def index(x, diffusion_step):
        return x[diffusion_step].view(-1, 1, 1, 1)

    def sample_noise(self, batch_size, n_channels, img_size):
        return torch.randn(size=(batch_size, n_channels, img_size, img_size), device=self.device)

    def sample_diffusion_step(self, batch_size):
        return torch.randint(0, self.n_diffusion_steps, size=(batch_size,), device=self.device)

    def batchify_diffusion_steps(self, diffusion_step, batch_size):
        return torch.full(
            size=(batch_size,), fill_value=diffusion_step, dtype=torch.long, device=self.device,
        )

    def get_noisy_image(self, ori_image, diffusion_step, random_noise=None): # Forward (diffusion) process
        b, c, h, _ = ori_image.shape
        if random_noise is not None:
            random_noise = self.sample_noise(batch_size=b, n_channels=c, img_size=h)
        alpha_bar_t = self.index(self.alpha_bar, diffusion_step=diffusion_step) # "$\bar{\alpha_{t}}$"
        mean = (alpha_bar_t ** 0.5) # $\sqrt{\bar{\alpha_{t}}}$
        var = 1 - alpha_bar_t # $(1 - \bar{\alpha_{t}})\mathbf{I}$
        return mean * ori_image + (var ** 0.5) * random_noise

    def predict_noise(self, noisy_image, diffusion_step):
        return self.net(noisy_image=noisy_image, diffusion_step=diffusion_step)

    @staticmethod
    def norm(image):
        return (image - torch.mean(image)) / torch.std(image)

    def get_loss(self, ori_image):
        b, c, h, _ = ori_image.shape
        # "Algorithm 1-3: $t \sim Uniform(\{1, \ldots, T\})$"
        diffusion_step = self.sample_diffusion_step(batch_size=b)
        # "Algorithm 1-4: $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$"
        random_noise = self.sample_noise(batch_size=b, n_channels=c, img_size=h)
        norm_ori_image = self.norm(ori_image)
        noisy_image = self.get_noisy_image(
            ori_image=norm_ori_image, diffusion_step=diffusion_step, random_noise=random_noise,
        )
        # print(noisy_image.shape)
        pred_noise = self.predict_noise(noisy_image=noisy_image, diffusion_step=diffusion_step)
        return F.mse_loss(pred_noise, random_noise, reduction="mean")

    @torch.no_grad()
    def denoise(self, noisy_image, cur_diffusion_step):
        self.net.eval()

        b, c, h, _ = noisy_image.shape
        diffusion_step = self.batchify_diffusion_steps(cur_diffusion_step, batch_size=b)
        alpha_t = self.index(self.alpha, diffusion_step=diffusion_step)
        alpha_bar_t = self.index(self.alpha_bar, diffusion_step=diffusion_step)
        pred_noise = self.predict_noise(noisy_image=noisy_image, diffusion_step=diffusion_step)
        # # ["Algorithm 2-4: $$\mu_{\theta}(x_{t}, t) =
        # \frac{1}{\sqrt{\alpha_{t}}}\Big(x_{t} - \frac{\beta_{t}}{\sqrt{1 - \bar{\alpha_{t}}}}z_{\theta}(x_{t}, t)\Big)$$
        # + \sigma_{t}z"
        denoised_image = (1 / (alpha_t ** 0.5)) * (
            noisy_image - (1 - alpha_t) / ((1 - alpha_bar_t) ** 0.5) * pred_noise
        )

        if cur_diffusion_step != 0:
            beta_t = self.index(self.beta, diffusion_step=diffusion_step)
            random_noise = self.sample_noise(batch_size=b, n_channels=c, img_size=h) # "$z$"
            denoised_image += (beta_t ** 0.5) * random_noise # "$\sigma_{t}z$"

        self.net.train()
        return denoised_image

    def sample(self, batch_size, n_channels, img_size): # Reverse (denoising) process
        x = self.sample_noise(batch_size=batch_size, n_channels=n_channels, img_size=img_size) # "$x_{T}$"
        for cur_diffusion_step in range(self.n_diffusion_steps - 1, -1, -1):
            x = self.denoise(noisy_image=x, cur_diffusion_step=cur_diffusion_step)
        return x

    @staticmethod
    def _get_frame(x):
        b, _, _, _ = x.shape
        grid = image_to_grid(x, n_cols=int(b ** 0.5))
        frame = np.array(grid)
        return frame

    def progressively_sample(self, batch_size, n_channels, img_size, save_path, n_frames=100):
        with imageio.get_writer(save_path, mode="I") as writer:
            x = self.sample_noise(batch_size=batch_size, n_channels=n_channels, img_size=img_size)
            for cur_diffusion_step in range(self.n_diffusion_steps - 1, -1, -1):
                x = self.denoise(noisy_image=x, cur_diffusion_step=cur_diffusion_step)

                if cur_diffusion_step % (self.n_diffusion_steps // n_frames) == 0:
                    frame = self._get_frame(x)
                    writer.append_data(frame)

    @staticmethod
    def _linearly_interpolate(x, y, n_points):
        _, b, c, d = x.shape
        lambs = torch.linspace(start=0, end=1, steps=n_points)
        lambs = lambs[:, None, None, None].expand(n_points, b, c, d)
        return ((1 - lambs) * x + lambs * y)

    def interpolate(self, ori_image1, ori_image2, interpolate_at=500, n_points=10, image_to_grid=True):
        diffusion_step = self.batchify_diffusion_steps(interpolate_at, batch_size=1)
        noisy_image1 = self.get_noisy_image(ori_image=ori_image1, diffusion_step=diffusion_step)
        noisy_image2 = self.get_noisy_image(ori_image=ori_image2, diffusion_step=diffusion_step)

        x = self._linearly_interpolate(noisy_image1, noisy_image2, n_points=n_points)
        for cur_diffusion_step in range(interpolate_at - 1, -1, -1):
            x = self.denoise(x, cur_diffusion_step=cur_diffusion_step)
        gen_image = torch.cat([ori_image1, x, ori_image2], dim=0)
        if not image_to_grid:
            return x
        gen_grid = image_to_grid(gen_image, n_cols=n_points + 2)
        return gen_grid

    def coarse_to_fine_interpolate(self, ori_image1, ori_image2, n_rows=9, n_points=10):
        rows = list()
        for interpolate_at in range(self.n_diffusion_steps, -1, - self.n_diffusion_steps // (n_rows - 1)):
            row = self.interpolate(
                ori_image1=ori_image1,
                ori_image2=ori_image2,
                interpolate_at=interpolate_at,
                n_points=n_points,
                image_to_grid=False,
            )
            rows.append(row)
        gen_image = torch.cat(rows, dim=0)
        gen_grid = image_to_grid(gen_image, n_cols=n_points + 2)
        return gen_grid

    def sample_eval_images(self, n_eval_imgs, batch_size, n_channels, img_size, save_dir):
        for idx in tqdm(range(1, math.ceil(n_eval_imgs / batch_size) + 1)):
            gen_image = self.sample(
                batch_size=batch_size,
                n_channels=n_channels,
                img_size=img_size,
            )
            gen_grid = image_to_grid(gen_image, n_cols=int(batch_size ** 0.5))
            save_image(
                gen_grid, path=Path(save_dir)/f"""{str(idx).zfill(len(str(n_eval_imgs)))}.jpg""",
            )


if __name__ == "__main__":
    DEVICE = torch.device("mps")
    model = UNet()
    x = torch.randn(4, 3, 64, 64)
    y = torch.randint(0, 1000, size=(4,))
    model(x, y)

    model = DDPM(device=DEVICE)
    x = torch.randn(4, 3, 64, 64).to(DEVICE)
    model.get_loss(x)

    batch_size = 4
    n_channels = 3
    img_size = 32
    sampled_image = model.sample(batch_size=batch_size, n_channels=n_channels, img_size=img_size)
    print(sampled_image.shape)

    ori_image = torch.randn(batch_size, n_channels, img_size, img_size)
    loss = model.get_loss(ori_image)
    print(loss)
