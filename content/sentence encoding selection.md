## Token Scoring

Each token embedding $e_i \in \mathbb{R}^d$ is passed through a small MLP: $s_i = \text{MLP}(e_i)$ consisting of

$$
\text{LayerNorm} \rightarrow \text{Linear} \rightarrow \text{GELU} \rightarrow \text{Dropout} \rightarrow \text{Linear}
$$

This produces one scalar score per token.

Padding tokens are masked: $s_i = 0$ if $\text{attn}_i = 0$
## Differentiable Ranking

We need a differentiable approximation of the discrete rank:

$$
\text{rank}(i) = 1 + \sum_j \mathbf{1}[s_j > s_i]
$$

We replace the indicator with a sigmoid:

$$
r_i = 1 + \sum_j \sigma\left(\frac{s_j - s_i}{\tau}\right)^\gamma
$$

where:

- $\tau$ controls smoothness
- $\gamma$ sharpens comparisons

As $\tau \to 0$ we have $r_i \to$ true discrete rank

To stabilize gradients, before ranking, scores are normalized per sequence:

$$
\tilde{s}_i = \frac{s_i - \mu}{\sigma}
$$

Padding tokens receive $r_i = +\infty$ so they are never selected.

## Soft Top-k Gate

Given a target sparsity $\rho$, we compute: $k = \text{round}(\rho \cdot T_{\text{eff}})$, where $T_{\text{eff}}$ is the number of valid tokens.

We define a soft gate:

$$
z_i^{\text{raw}} = \sigma\left(\frac{k - r_i}{\tau}\right)
$$

This is normalized to enforce exact mass:

$$
z_i = \frac{z_i^{\text{raw}}}{\sum_j z_j^{\text{raw}}} \cdot k
$$

Thus:

$$
\sum_i z_i = k
$$

This $z$ is used for training in the backwards pass.
## Hard Top-k for Evaluation

For evaluation, a discrete mask is constructed:

$$
g =
\begin{cases}
1 & r_i \le k \\
0 & \text{otherwise}
\end{cases}
$$

This avoids evaluating metrics on soft fractional weights.

## Reconstruction Objective

Let:

- $f(\cdot)$ be the frozen sentence encoder
- $h = f(x)$ be the full sentence representation
- $\tilde{h} = f(x \odot g)$ be the gated representation (of a given $\rho$)

The loss is:

$$
\mathcal{L}_{\text{recon}} = 1 - \text{Cosine}(\tilde{h}, h)
$$

This forces the selected tokens to preserve semantic information.

The final loss averages over multiple $\rho$ values in a sweep:

$$
\mathcal{L} = \frac{1}{|\mathcal{R}|} \sum_{\rho \in \mathcal{R}} \mathcal{L}_{\text{recon}}(\rho)
$$