# 1. Introduction
- Diffusion probabilistic model은 Parameterized Markov chain입니다.
- Variational inference를 사용합니다.
- Markov chain: 각각의 상태에서 다른 상태로의 변이 확률이 정해져 있습니다.
- Transition: State의 변화
- Markov property: 현재의 State가 과거의 State에만 영향을 받습니다.
- Diffusion process는 신호가 붕괴될 때까지 신호에 점진적으로 노이즈를 더하는 Markov chain입니다.

# 2. Background
- 원본 데이터 $\textbf{x}_{0} \sim q(\textbf{x}_{0})$.
- $\textbf{x}_{1}, \ldots, \textbf{x}_{T}$: $\textbf{x}_{0}$와 동일한 차원을 갖는 Latents.
- Diffusion process
    - $\textbf{x}_{t - 1}$에서 $\textbf{x}_{t}$로 변이가 일어날 확률은 $t$ timestep에서 정해진 $\beta_{t}$에 따라 정해지는 정규분포를 따릅니다.
    <!-- - Reparameterization trick: $\epsilon \sim \mathcal{N}(\textbf{0}, \textbf{I})$에 대해  -->
    - $\textbf{x}_{0}$에서 $\textbf{x}_{t}$로 변이가 일어날 확률은 마찬가지로 $t$ timestep에서 정해진 $\beta_{t}$에 따라 정해지는 정규분포를 따릅니다. 따라서 한 번 노이즈를 추가함으로써 $\textbf{x}_{0}$로부터 $\textbf{x}_{t}$를 만들 수 있습니다.
<!-- - Reverse process: Joint distribution $p_{\theta}(\textbf{x}_{0:T})$이며 Markov chain입니다. $p(\textbf{x}_{T}) = \mathcal{N}(\textbf{x}_{T}; \textbf{0}, \textbf{I})$, 즉 $T$ timestep에서의 Latent는 표준정규분포를 따릅니다.
- $t$ timestep에서 $t - 1$ timestep으로의 변이 확률 또한 표준정규분포를 따릅니다.
- Diffusion process
    - 각 Timestep의 Variance는 $\beta_{t}\textbf{I}$로 정해지며 이때 $\beta_{t}$는 정해져 있습니다. (Variance schedule) 이 말은 각 Timestep에서 Data에 더해줄 Gaussian noise가 정해져 있다는 뜻입니다.
    - $\textbf{x}_{0}$를 가지고 $\textbf{x}_{t}$를 만들 때 여러 번에 걸쳐서 노이즈를 추가하지 않고 새롭게 계산된 노이즈를 한 번만 더함으로써 바로 가능합니다. -->
