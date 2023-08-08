# Paper Reading
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)
## Methodology
- ***Transitions of this chain are learned to reverse a diffusion process, which is a Markov chain that gradually adds noise to the data in the opposite direction of sampling until signal is destroyed.*** When the diffusion consists of small amounts of Gaussian noise, it is sufficient to set the sampling chain transitions to conditional Gaussians too, allowing for a particularly simple neural network parameterization.
- The joint distribution $p_{\theta}(x_{0:T})$ is called the reverse process, and it is defined as a Markov chain with learned Gaussian transitions starting at $p(x_{T}) = \mathcal{N}(x_{T};0,I)$ (Comment: The variable $x_{T}$ follows normal distribution with mean $0$ and variance $I$.):
$$p_{\theta}(x_{t - 1} \vert x_{t}) := \mathcal{N}(x_{t - 1}; \mu_{\theta}(x_{t}, t), \Sigma_{\theta}(x_{t}, t))$$
$$p_{\theta}(x_{0:T}) := p_{\theta}(x_{T})\prod^{T}_{t = 1}p_{\theta}(x_{t - 1} \vert x_{t})$$
- (Comment: On reverse process, $x_{t - 1}$ follows normal distribution with mean $\mu_{\theta}$ and variance $\Sigma_{\theta}$ given $x_{t}$.)
- What distinguishes diffusion models from other types of latent variable models is that the approximate posterior $q(x_{1:T} | x_{0})$, called the forward process or diffusion process, is fixed to a Markov chain that gradually adds Gaussian noise to the data according to a variance schedule $β_{1}, \ldots, β_{T}$:
$$q(x_{t} \vert x_{t - 1}) := \mathcal{N}(x_{t}; \sqrt{1 - \beta_{t}}x_{t - 1}, \beta_{t}I)$$
$$q(x_{1:T} | x_{0}) := \prod^{T}_{t = 1}q(x_{t} \vert x_{t - 1})$$
- (Comment: On diffusion process, $x_{t}$ follows normal distribution with mean $\sqrt{1 - \beta_{t}}x_{t - 1}$ and variance $\beta_{t}I$ given $x_{t - 1}$.)
## References
- [53] [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/pdf/1503.03585.pdf)

# Formulas
## Forward Process
$$\epsilon_{\theta}(x_{t}, t)$$
$$\alpha_{t} = 1 - \beta_{t}$$
$$\bar{\alpha_{t}} = \prod^{t}_{s=1}{\alpha_{s}}$$
$$\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$
$$q(x_{t} \vert x_{t - 1}) := \mathcal{N}(x_{t}; \sqrt{1 - \beta_{t}}x_{t - 1}, \beta_{t}I)$$
$$q(x_{1:T} | x_{0}) := \prod^{T}_{t = 1}q(x_{t} \vert x_{t - 1})$$
## Reverse Process
$$p_{\theta}(x_{t - 1} \vert x_{t}) := \mathcal{N}(x_{t - 1}; \mu_{\theta}(x_{t}, t), \Sigma_{\theta}(x_{t}, t))$$
$$p_{\theta}(x_{0:T}) := p_{\theta}(x_{T})\prod^{T}_{t = 1}p_{\theta}(x_{t - 1} \vert x_{t})$$
$$\frac{1}{\sqrt{\alpha_{t}}}\Big(x_{t} - \frac{\beta_{t}}{\sqrt{1 - \bar{\alpha_{t}}}}\epsilon_{\theta}(x_{t}, t)\Big)$$

# Kullback–Leibler Divergence (KL Divergence)
- Also called 'relative entropy' and 'I-divergence'.
$$D_{KL}(P || Q)$$
- A measure of how one probability distribution $P$ is different from a second, reference probability distribution $Q$.
- For discrete probability distributions $P$ and $Q$ defined on the same sample space, $\mathcal{X}$, the relative entropy from $Q$ to $P$ is defined to be
$$D_{KL}(P || Q) = - \sum_{x \in \mathcal{X}}P(x)\log\bigg(\frac{Q(x)}{P(x)}\bigg)$$
- For distributions $P$ and $Q$ of a continuous random variable, relative entropy is defined to be the integral:
$$D_{KL}(P || Q) = - \int_{-\infty}^{\infty} p(x)\log\bigg(\frac{q(x)}{p(x)}\bigg)dx$$
- where $p$ and $q$ denote the probability densities of $P$ and $Q$.
## References
- [1] https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
