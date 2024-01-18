# 1. Introduction
# 2. Method
## 2.1) Problem Scenario
- $N$개의 i.i.d. samples로 구성된 Dataset $\textbf{X} = \{\textbf{x}^{(i)}\}^{N}_{i = 1}$를 고려합니다.
- 데이터가 다음의 두 가지 프로세스를 거쳐서 생성된다고 가정합니다.
    1. Continous random variablze $z^{(i)}$가 Prior distribution $p_{\theta}(\textbf{z})$로부터 생성됩니다.
    2. $\textbf{x}^{(i)}$가 conditional distribution $p_{\theta}(\textbf{x} \vert \textbf{z})$로부터 생성됩니다.
- 미분 가능 가정
## 2.2) The Variational Bound
- Variational parameters (Encoder parameters) $\phi$
- Generative parameters (Decoder parameters) $\theta$
## 2.3) The SGVB Estimator and AEVB Algorithm
## 2.4) The Reparameterization Trick
# 1. Visualizations
# 2. Gaussian Case

- Intractable?