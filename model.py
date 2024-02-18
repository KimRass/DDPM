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

from utils import image_to_grid, save_image


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, n_diffusion_steps, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()

        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(n_diffusion_steps).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [n_diffusion_steps, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [n_diffusion_steps, d_model // 2, 2]
        emb = emb.view(n_diffusion_steps, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def _sinusoidal_embedding(self):
        pos = torch.arange(self.n_diffusion_steps).unsqueeze(1)
        i = torch.arange(self.dim // 2).unsqueeze(0)
        angle = pos / (10_000 ** (2 * i / self.dim))

        pe_mat = torch.zeros(size=(self.n_diffusion_steps, self.dim))
        pe_mat[:, 0:: 2] = torch.sin(angle)
        pe_mat[:, 1:: 2] = torch.cos(angle)
        return pe_mat

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        nn.init.xavier_uniform_(self.main.weight)
        nn.init.zeros_(self.main.bias)

    def forward(self, x, temb):
        x = self.main(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        nn.init.xavier_uniform_(self.main.weight)
        nn.init.zeros_(self.main.bias)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.main(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        h = self.block1(x)
        h = h + self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class OldUNet(nn.Module):
    def __init__(self, n_diffusion_steps=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1], num_res_blocks=2, dropout=0.1):
        super().__init__()

        assert all([i < len(ch_mult) for i in attn]), "attn index out of bound"

        tdim = ch * 4
        self.time_embedding = TimeEmbedding(
            n_diffusion_steps=n_diffusion_steps, d_model=ch, dim=tdim,
        )

        self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        cxs = [ch]  # record output channel when dowmsample for upsample
        cur_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(
                    ResBlock(
                        in_ch=cur_ch,
                        out_ch=out_ch,
                        tdim=tdim,
                        dropout=dropout,
                        attn=(i in attn)
                    )
                )
                cur_ch = out_ch
                cxs.append(cur_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(cur_ch))
                cxs.append(cur_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(cur_ch, cur_ch, tdim, dropout, attn=True),
            ResBlock(cur_ch, cur_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=cxs.pop() + cur_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                cur_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(cur_ch))
        assert len(cxs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, cur_ch),
            Swish(),
            nn.Conv2d(cur_ch, 3, kernel_size=3, stride=1, padding=1)
        )
        self.initialize()

    def initialize(self):
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        nn.init.zeros_(self.tail[-1].bias)

    def forward(self, noisy_image, diffusion_step):
        temb = self.time_embedding(diffusion_step)
        x = self.head(noisy_image)
        xs = [x]
        for layer in self.downblocks:
            x = layer(x, temb)
            xs.append(x)

        for layer in self.middleblocks:
            x = layer(x, temb)

        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                x = torch.cat([x, xs.pop()], dim=1)

            if isinstance(layer, UpSample):
                x = layer(x)
            else:
                x = layer(x, temb)
        x = self.tail(x)
        assert len(xs) == 0
        return x


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
