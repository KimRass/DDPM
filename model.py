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

from utils import (
    sample_noise,
    index,
    image_to_grid,
    get_linear_beta_schdule,
    save_image,
    batchify_timesteps,
)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, n_timesteps, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()

        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(n_timesteps).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [n_timesteps, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [n_timesteps, d_model // 2, 2]
        emb = emb.view(n_timesteps, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def _sinusoidal_embedding(self):
        pos = torch.arange(self.n_timesteps).unsqueeze(1)
        i = torch.arange(self.dim // 2).unsqueeze(0)
        angle = pos / (10_000 ** (2 * i / self.dim))

        pe_mat = torch.zeros(size=(self.n_timesteps, self.dim))
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
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class UNet(nn.Module):
    def __init__(self, n_timesteps=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1], num_res_blocks=2, dropout=0.1):
        super().__init__()

        assert all([i < len(ch_mult) for i in attn]), "attn index out of bound"

        tdim = ch * 4
        self.time_embedding = TimeEmbedding(
            n_timesteps=n_timesteps, d_model=ch, dim=tdim,
        )

        self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        cxs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                cxs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                cxs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=cxs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(cxs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 3, kernel_size=3, stride=1, padding=1)
        )
        self.initialize()

    def initialize(self):
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        nn.init.zeros_(self.tail[-1].bias)

    def forward(self, x, t):
        # Timestep embedding
        temb = self.time_embedding(t)
        # Downsampling
        x = self.head(x)
        xs = [x]
        for layer in self.downblocks:
            x = layer(x, temb)
            xs.append(x)
        # Middle
        for layer in self.middleblocks:
            x = layer(x, temb)
        # Upsampling
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
    def __init__(self, n_timesteps=1000, init_beta=0.0001, fin_beta=0.02):
        super().__init__()

        self.n_timesteps = n_timesteps
        self.init_beta = init_beta
        self.fin_beta = fin_beta

        self.beta = get_linear_beta_schdule(
            init_beta=init_beta, fin_beta=fin_beta, n_timesteps=n_timesteps,
        )
        self.alpha = 1 - self.beta # "$\alpha_{t} = 1 - \beta_{t}$"
        # "$\bar{\alpha_{t}} = \prod^{t}_{s=1}{\alpha_{s}}$"
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        self.model = UNet(n_timesteps=n_timesteps)

    def _q(self, t): # $q(x_{t} \vert x_{0})$
        alpha_bar_t = index(self.alpha_bar.to(t.device), t=t) # "$\bar{\alpha_{t}}$"
        mean = (alpha_bar_t ** 0.5) # $\sqrt{\bar{\alpha_{t}}}$
        var = 1 - alpha_bar_t # $(1 - \bar{\alpha_{t}})\mathbf{I}$
        return mean, var

    def _sample_from_q(self, x0, t, eps):
        b, c, h, _ = x0.shape
        if eps is None:
            # "$\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$"
            eps = sample_noise(batch_size=b, n_channels=c, img_size=h, device=x0.device)
        mean, var = self._q(t)
        return mean * x0 + (var ** 0.5) * eps

    def forward(self, x0, t, eps=None): # Forward (diffusion) process
        noisy_image = self._sample_from_q(x0=x0, t=t, eps=eps)
        return noisy_image

    def predict_noise(self, x, t):
        # The model returns its estimation of the noise that was added.
        return self.model(x, t=t)

    def get_loss(self, x0, t, eps):
        noisy_image = self(x0, t=t, eps=eps)
        pred_eps = self.predict_noise(x=noisy_image, t=t)
        loss = F.mse_loss(pred_eps, eps)
        return loss

    @torch.no_grad()
    def _sample_from_p(self, x, timestep):
        self.model.eval()

        b, c, h, _ = x.shape
        t = batchify_timesteps(timestep, batch_size=b, device=x.device)
        alpha_t = index(self.alpha.to(x.device), t=t)
        alpha_bar_t = index(self.alpha_bar.to(x.device), t=t)
        pred_noise = self.predict_noise(x.detach(), t=t)
        # # ["Algorithm 2"] "4: $$\mu_{\theta}(x_{t}, t) =
        # \frac{1}{\sqrt{\alpha_{t}}}\Big(x_{t} - \frac{\beta_{t}}{\sqrt{1 - \bar{\alpha_{t}}}}z_{\theta}(x_{t}, t)\Big)$$
        # + \sigma_{t}z"
        model_mean = (1 / (alpha_t ** 0.5)) * (x - (1 - alpha_t) / ((1 - alpha_bar_t) ** 0.5) * pred_noise)

        if timestep > 0:
            beta_t = index(self.beta.to(x.device), t=t)
            eps = sample_noise(batch_size=b, n_channels=c, img_size=h, device=x.device)
            model_mean += (beta_t ** 0.5) * eps

        self.model.train()
        return model_mean

    def sample(self, batch_size, n_channels, img_size, device, return_image=True): # Reverse (denoising) process
        x = sample_noise(batch_size=batch_size, n_channels=n_channels, img_size=img_size, device=device)
        for timestep in range(self.n_timesteps - 1, -1, -1):
            x = self._sample_from_p(x, timestep=timestep)
        if not return_image:
            return x

        image = image_to_grid(x, n_cols=int(batch_size ** 0.5))
        return image

    def _get_frame(self, x):
        b, _, _, _ = x.shape
        grid = image_to_grid(x, n_cols=int(b ** 0.5))
        frame = np.array(grid)
        return frame

    def progressively_sample(self, batch_size, n_channels, img_size, device, save_path, n_frames=100):
        with imageio.get_writer(save_path, mode="I") as writer:
            x = sample_noise(batch_size=batch_size, n_channels=n_channels, img_size=img_size, device=device)
            for timestep in range(self.n_timesteps - 1, -1, -1):
                x = self._sample_from_p(x, timestep=timestep)

                if timestep % (self.n_timesteps // n_frames) == 0:
                    frame = self._get_frame(x)
                    writer.append_data(frame)

    def _linearly_interpolate(self, x, y, n):
        _, b, c, d = x.shape
        lambs = torch.linspace(start=0, end=1, steps=n)
        lambs = lambs[:, None, None, None].expand(n, b, c, d)
        return ((1 - lambs) * x + lambs * y)

    def interpolate(self, image1, image2, timestep, n=10, return_image=True):
        t = batchify_timesteps(timestep, batch_size=1, device=image1.device)
        noisy_image1 = self(image1, t=t)
        noisy_image2 = self(image2, t=t)

        x = self._linearly_interpolate(noisy_image1, noisy_image2, n=n)
        for timestep in range(timestep - 1, -1, -1):
            x = self._sample_from_p(x, timestep=timestep)
        x = torch.cat([image1, x, image2], dim=0)
        if not return_image:
            return x
        image = image_to_grid(x, n_cols=n + 2)
        return image

    def coarse_to_fine_interpolate(self, image1, image2, n_rows=9, n=10):
        rows = list()
        for timestep in range(self.n_timesteps, -1, - self.n_timesteps // (n_rows - 1)):
            row = self.interpolate(image1, image2, timestep=timestep, n=n, return_image=False)
            rows.append(row)
        x = torch.cat(rows, dim=0)
        image = image_to_grid(x, n_cols=n + 2)
        return image

    def sample_eval_images(self, n_eval_imgs, batch_size, n_channels, img_size, save_dir, device):
        for idx in tqdm(range(1, math.ceil(n_eval_imgs / batch_size) + 1)):
            gen_image = self.sample(
                batch_size=batch_size,
                n_channels=n_channels,
                img_size=img_size,
                device=device,
            )
            save_image(
                gen_image, path=Path(save_dir)/f"""{str(idx).zfill(len(str(n_eval_imgs)))}.jpg""",
            )



if __name__ == "__main__":
    model = UNet(n_timesteps=1000)
    batch_size = 2
    img_size = 32
    x = torch.randn(batch_size, 3, img_size, img_size)
    t = torch.randint(1000, (batch_size, ))
    y = model(x, t)
    y.shape
