# References:
    # https://nn.labml.ai/diffusion/stable_diffusion/sampler/ddim.html


from typing import Optional, List
import numpy as np
import torch
from labml import monit

from utils import (
    sample_noise,
    index,
    image_to_grid,
    get_linear_beta_schdule,
)
from model import UNet


class DiffusionSampler:
    """
    ## Base class for sampling algorithms
    """
    def __init__(self, model):
        """
        :param model: is the model to predict noise $\epsilon_\text{cond}(x_t, c)$
        """
        super().__init__()
        # Set the model $\epsilon_\text{cond}(x_t, c)$
        self.model = model
        # Get number of steps the model was trained with $T$
        self.n_steps = model.n_steps

    def get_eps(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor, *,
                uncond_scale: float, uncond_cond: Optional[torch.Tensor]):
        """
        ## Get $\epsilon(x_t, c)$

        :param x: is $x_t$ of shape `[batch_size, channels, height, width]`
        :param t: is $t$ of shape `[batch_size]`
        :param c: is the conditional embeddings $c$ of shape `[batch_size, emb_size]`
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param uncond_cond: is the conditional embedding for empty prompt $c_u$
        """
        # When the scale $s = 1$
        # $$\epsilon_\theta(x_t, c) = \epsilon_\text{cond}(x_t, c)$$
        if uncond_cond is None or uncond_scale == 1.:
            # return self.model(x, t, c)
            return self.model.predict_noise(x, t, c)

        # Duplicate $x_t$ and $t$
        x_in = torch.cat([x] * 2)
        t_in = torch.cat([t] * 2)
        # Concatenated $c$ and $c_u$
        c_in = torch.cat([uncond_cond, c])
        # Get $\epsilon_\text{cond}(x_t, c)$ and $\epsilon_\text{cond}(x_t, c_u)$
        # e_t_uncond, e_t_cond = self.model(x_in, t_in, c_in).chunk(2)
        e_t_uncond, e_t_cond = self.model.predict_noise(x_in, t_in, c_in).chunk(2)
        # Calculate
        # $$\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$$
        e_t = e_t_uncond + uncond_scale * (e_t_cond - e_t_uncond)

        #
        return e_t

    def sample(self,
               shape: List[int],
               cond: torch.Tensor,
               repeat_noise: bool = False,
               temperature: float = 1.,
               x_last: Optional[torch.Tensor] = None,
               uncond_scale: float = 1.,
               uncond_cond: Optional[torch.Tensor] = None,
               skip_steps: int = 0,
               ):
        """
        ### Sampling Loop

        :param shape: is the shape of the generated images in the
            form `[batch_size, channels, height, width]`
        :param cond: is the conditional embeddings $c$
        :param temperature: is the noise temperature (random noise gets multiplied by this)
        :param x_last: is $x_T$. If not provided random noise will be used.
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param uncond_cond: is the conditional embedding for empty prompt $c_u$
        :param skip_steps: is the number of time steps to skip.
        """
        raise NotImplementedError()

    def paint(self, x: torch.Tensor, cond: torch.Tensor, t_start: int, *,
              orig: Optional[torch.Tensor] = None,
              mask: Optional[torch.Tensor] = None, orig_noise: Optional[torch.Tensor] = None,
              uncond_scale: float = 1.,
              uncond_cond: Optional[torch.Tensor] = None,
              ):
        """
        ### Painting Loop

        :param x: is $x_{T'}$ of shape `[batch_size, channels, height, width]`
        :param cond: is the conditional embeddings $c$
        :param t_start: is the sampling step to start from, $T'$
        :param orig: is the original image in latent page which we are in paining.
        :param mask: is the mask to keep the original image.
        :param orig_noise: is fixed noise to be added to the original image.
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param uncond_cond: is the conditional embedding for empty prompt $c_u$
        """
        raise NotImplementedError()

    def q_sample(self, x0: torch.Tensor, index: int, noise: Optional[torch.Tensor] = None):
        """
        ### Sample from $q(x_t|x_0)$

        :param x0: is $x_0$ of shape `[batch_size, channels, height, width]`
        :param index: is the time step $t$ index
        :param noise: is the noise, $\epsilon$
        """
        raise NotImplementedError()


class DDIM(DiffusionSampler):
    def __init__(self, model, n_ddim_timesteps, init_beta, fin_beta, ddim_eta=0):
        super().__init__(model)
        # Number of steps, $T$
        self.n_ddim_timesteps = n_ddim_timesteps
        self.init_beta = init_beta
        self.fin_beta = fin_beta

        self.n_timesteps = model.n_steps

        self.timesteps = np.asarray(
            list(range(0, self.n_timesteps, self.n_timesteps // n_ddim_timesteps))
        ) + 1

        self.beta = get_linear_beta_schdule(
            init_beta=init_beta, fin_beta=fin_beta, n_timesteps=n_ddim_timesteps,
        )
        # self.alpha = 1 - self.beta
        alpha_bar = self._get_alpha_bar(self.alpha)
        self.alpha_bar = self._get_alpha_bar(self.alpha)

        with torch.no_grad():
            self.alpha_bar = alpha_bar[self.timesteps].clone().to(torch.float32)
            self.alpha_bar_prev = torch.cat([alpha_bar[: 1], alpha_bar[self.timesteps[:-1]]])

            self.ddim_sigma = ddim_eta
            self.ddim_sigma *= ((1 - self.alpha_bar_prev) / (1 - self.alpha_bar)) ** 0.5
            self.ddim_sigma *= (1 - self.alpha_bar / self.alpha_bar_prev) ** 0.5

            # self.sqrt_1m_alpha_bar = (1 - self.alpha_bar) ** 0.5

    def _get_alpha_bar(self, alpha):
        # "$\bar{\alpha_{t}} = \prod^{t}_{s=1}{\alpha_{s}}$"
        return torch.cumprod(alpha, dim=0)

    def _q(self, t):
        alpha_bar_t = index(self.alpha_bar.to(t.device), t=t)
        mean = (alpha_bar_t ** 0.5)
        var = 1 - alpha_bar_t
        return mean, var

    def _sample_from_q(self, x0, t, eps):
        b, c, h, _ = x0.shape
        if eps is None:
            eps = sample_noise(batch_size=b, n_channels=c, img_size=h, device=x0.device)
        mean, var = self._q(t)
        return mean * x0 + (var ** 0.5) * eps

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor, step: int, index: int, *,
                 repeat_noise: bool = False,
                 temperature: float = 1.,
                 uncond_scale: float = 1.,
                 uncond_cond: Optional[torch.Tensor] = None):
        """
        ### Sample $x_{\tau_{i-1}}$

        :param x: is $x_{\tau_i}$ of shape `[batch_size, channels, height, width]`
        :param c: is the conditional embeddings $c$ of shape `[batch_size, emb_size]`
        :param t: is $\tau_i$ of shape `[batch_size]`
        :param step: is the step $\tau_i$ as an integer
        :param index: is index $i$ in the list $[\tau_1, \tau_2, \dots, \tau_S]$
        :param repeat_noise: specified whether the noise should be same for all samples in the batch
        :param temperature: is the noise temperature (random noise gets multiplied by this)
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param uncond_cond: is the conditional embedding for empty prompt $c_u$
        """

        # Get $\epsilon_\theta(x_{\tau_i})$
        e_t = self.get_eps(x, t, c, uncond_scale=uncond_scale, uncond_cond=uncond_cond)

        # Calculate $x_{\tau_{i - 1}}$ and predicted $x_0$
        x_prev, pred_x0 = self.get_x_prev_and_pred_x0(
            e_t,
            index,
            x,
            temperature=temperature,
            repeat_noise=repeat_noise,
        )
        return x_prev, pred_x0, e_t

    @torch.no_grad()
    def sample(
        self,
        shape: List[int],
        cond: torch.Tensor,
        repeat_noise: bool = False,
        temperature: float = 1.,
        x_last: Optional[torch.Tensor] = None,
        uncond_scale: float = 1.,
        uncond_cond: Optional[torch.Tensor] = None,
        skip_steps: int = 0,
    ):
        """
        :param cond: is the conditional embeddings $c$
        :param temperature: is the noise temperature (random noise gets multiplied by this)
        :param x_last: is $x_{\tau_S}$. If not provided random noise will be used.
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param uncond_cond: is the conditional embedding for empty prompt $c_u$
        :param skip_steps: is the number of time steps to skip $i'$. We start sampling from $S - i'$.
            And `x_last` is then $x_{\tau_{S - i'}}$.
        """

        device = self.model.device
        bs = shape[0]

        # Get $x_{\tau_S}$
        # x = x_last if x_last is not None else torch.randn(shape, device=device)
        x = x_last if x_last is not None else sample_noise(*shape, device=device)

        # Time steps to sample at $\tau_{S - i'}, \tau_{S - i' - 1}, \dots, \tau_1$
        timesteps = np.flip(self.timesteps)[skip_steps:]

        for i, step in enumerate(timesteps):
        # for i, step in monit.enum('Sample', timesteps):
            # Index $i$ in the list $[\tau_1, \tau_2, \dots, \tau_S]$
            index = len(timesteps) - i - 1
            # Time step $\tau_i$
            ts = x.new_full((bs,), step, dtype=torch.long)

            # Sample $x_{\tau_{i-1}}$
            x, pred_x0, e_t = self.p_sample(
                x,
                cond,
                ts,
                step,
                index=index,
                repeat_noise=repeat_noise,
                temperature=temperature,
                uncond_scale=uncond_scale,
                uncond_cond=uncond_cond,
            )

        # Return $x_0$
        return x

    def get_x_prev_and_pred_x0(self, e_t: torch.Tensor, index: int, x: torch.Tensor, *,
                               temperature: float,
                               repeat_noise: bool):
        """
        ### Sample $x_{\tau_{i-1}}$ given $\epsilon_\theta(x_{\tau_i})$
        """

        # $\alpha_{\tau_i}$
        alpha = self.alpha[index]
        # $\alpha_{\tau_{i-1}}$
        alpha_prev = self.alpha_bar_prev[index]
        # $\sigma_{\tau_i}$
        sigma = self.ddim_sigma[index]
        # $\sqrt{1 - \alpha_{\tau_i}}$
        sqrt_one_minus_alpha = self.sqrt_1m_alpha_bar[index]

        # Current prediction for $x_0$,
        # $$\frac{x_{\tau_i} - \sqrt{1 - \alpha_{\tau_i}}\epsilon_\theta(x_{\tau_i})}{\sqrt{\alpha_{\tau_i}}}$$
        pred_x0 = (x - sqrt_one_minus_alpha * e_t) / (alpha ** 0.5)
        # Direction pointing to $x_t$
        # $$\sqrt{1 - \alpha_{\tau_{i- 1}} - \sigma_{\tau_i}^2} \cdot \epsilon_\theta(x_{\tau_i})$$
        dir_xt = (1. - alpha_prev - sigma ** 2).sqrt() * e_t

        # No noise is added, when $\eta = 0$
        if sigma == 0.:
            noise = 0.
        # If same noise is used for all samples in the batch
        elif repeat_noise:
            noise = torch.randn((1, *x.shape[1:]), device=x.device)
            # Different noise for each sample
        else:
            noise = torch.randn(x.shape, device=x.device)

        # Multiply noise by the temperature
        noise = noise * temperature

        #  \begin{align}
        #     x_{\tau_{i-1}} &= \sqrt{\alpha_{\tau_{i-1}}}\Bigg(
        #             \frac{x_{\tau_i} - \sqrt{1 - \alpha_{\tau_i}}\epsilon_\theta(x_{\tau_i})}{\sqrt{\alpha_{\tau_i}}}
        #             \Bigg) \\
        #             &+ \sqrt{1 - \alpha_{\tau_{i- 1}} - \sigma_{\tau_i}^2} \cdot \epsilon_\theta(x_{\tau_i}) \\
        #             &+ \sigma_{\tau_i} \epsilon_{\tau_i}
        #  \end{align}
        x_prev = (alpha_prev ** 0.5) * pred_x0 + dir_xt + sigma * noise

        #
        return x_prev, pred_x0

    @torch.no_grad()
    def q_sample(self, x0: torch.Tensor, index: int, noise: Optional[torch.Tensor] = None):
        """
        ### Sample from $q_{\sigma,\tau}(x_{\tau_i}|x_0)$

        $$q_{\sigma,\tau}(x_t|x_0) =
         \mathcal{N} \Big(x_t; \sqrt{\alpha_{\tau_i}} x_0, (1-\alpha_{\tau_i}) \mathbf{I} \Big)$$

        :param x0: is $x_0$ of shape `[batch_size, channels, height, width]`
        :param index: is the time step $\tau_i$ index $i$
        :param noise: is the noise, $\epsilon$
        """

        # Random noise, if noise is not specified
        if noise is None:
            noise = torch.randn_like(x0)

        # Sample from
        #  $$q_{\sigma,\tau}(x_t|x_0) =
        #          \mathcal{N} \Big(x_t; \sqrt{\alpha_{\tau_i}} x_0, (1-\alpha_{\tau_i}) \mathbf{I} \Big)$$
        return self.sqrt_alpha_bar[index] * x0 + self.sqrt_1m_alpha_bar[index] * noise

    @torch.no_grad()
    def paint(self, x: torch.Tensor, cond: torch.Tensor, t_start: int, *,
              orig: Optional[torch.Tensor] = None,
              mask: Optional[torch.Tensor] = None, orig_noise: Optional[torch.Tensor] = None,
              uncond_scale: float = 1.,
              uncond_cond: Optional[torch.Tensor] = None,
              ):
        """
        ### Painting Loop

        :param x: is $x_{S'}$ of shape `[batch_size, channels, height, width]`
        :param cond: is the conditional embeddings $c$
        :param t_start: is the sampling step to start from, $S'$
        :param orig: is the original image in latent page which we are in paining.
            If this is not provided, it'll be an image to image transformation.
        :param mask: is the mask to keep the original image.
        :param orig_noise: is fixed noise to be added to the original image.
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param uncond_cond: is the conditional embedding for empty prompt $c_u$
        """
        # Get  batch size
        bs = x.shape[0]

        # Time steps to sample at $\tau_{S`}, \tau_{S' - 1}, \dots, \tau_1$
        timesteps = np.flip(self.timesteps[:t_start])

        # for i, step in monit.enum('Paint', timesteps):
        for i, step in enumerate(timesteps):
            # Index $i$ in the list $[\tau_1, \tau_2, \dots, \tau_S]$
            index = len(timesteps) - i - 1
            # Time step $\tau_i$
            ts = x.new_full((bs,), step, dtype=torch.long)

            # Sample $x_{\tau_{i-1}}$
            x, _, _ = self.p_sample(x, cond, ts, step, index=index,
                                    uncond_scale=uncond_scale,
                                    uncond_cond=uncond_cond)

            # Replace the masked area with original image
            if orig is not None:
                # Get the $q_{\sigma,\tau}(x_{\tau_i}|x_0)$ for original image in latent space
                orig_t = self.q_sample(orig, index, noise=orig_noise)
                # Replace the masked area
                x = orig_t * mask + x * (1 - mask)

        #
        return x
