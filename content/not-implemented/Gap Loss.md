We assume two independent latent variables $e,f$ are used to generate a particular sentence $s$:
$$
p(e,f,s) = p(e)\, p(f)\, p(s \mid e,f)
$$
- Prior on $e$: sparse Bernoulli (entities are rare)  
- Prior on $f$: standard Gaussian (or other high-capacity prior)  

>**Note:** assuming $f,e$ independents is clearly wrong, however (on top of being easier to model) its an architectural bias that enforces disentanglement

Conditional likelihood:
$$
p(s \mid e,f) = D(e+f) \quad \text{addition may be subsituted with other aggregation}
$$
where $D$ is the shared decoder

We approximate the posterior as factorized:
$$
q(e,f \mid s) = q(e \mid s)\, q(f \mid s)
$$
Implemented by two encoders:
- $E_{sem}(s)$ for $e$ (entity latent)
- $E_{syn}(s)$ for $f$ (form latent)

For a VAE with two latents, the standard evidence lower bound is:
$$
\mathcal{L}_{\text{ELBO}}
=
\mathbb{E}_{q(e,f\mid s)} [ \log p(s \mid e,f) ]
- D_{KL}\!\left( q(e \mid s) \| p(e) \right)
- D_{KL}\!\left( q(f \mid s) \| p(f) \right)
$$

We want $f$ alone to **fail** at reconstructing $s$, so we define form-only likelihood:
$$
p(s \mid f) = D(f)
$$

We introduce an auxiliary term that **maximizes error** (minimizes likelihood):
$$
\mathcal{L}_{gap}
=
-\, \mathbb{E}_{q(f\mid s)} [ \log p(s \mid f) ]
$$
Equivalently, we **add** this term to the ELBO objective with positive weight

The modified ELBO becomes:
$$
\begin{aligned}
\mathcal{L}
&=
\mathbb{E}_{q(e,f\mid s)} [ \log p(s \mid e,f) ] \\
&\quad - D_{KL}\!\left( q(e \mid s) \| p(e) \right) \\
&\quad - D_{KL}\!\left( q(f \mid s) \| p(f) \right) \\ 
&\quad + \alpha\, \mathbb{E}_{q(f\mid s)} [ \log p(s \mid f) ] \quad (\text{maximize error on }f) \\
&\quad + \beta\, \mathbb{E}_{q(e\mid s)} [ \log p(s \mid e) ] \quad (\text{maximize error on }e)
\end{aligned}

$$

where $\alpha,\beta > 0$ controls how strongly we penalize $f$-only and $e$-only reconstructions

Using priors we can control how much information each latent variable carries
- $p(f) = \mathcal{N}(0,I)$ (very expressive)
- $p(e) = \prod_i \text{Bernoulli}(\pi)$ with $\pi \ll 0.5$ (not very expressive)

>**Note:** Decoder $D$ is shared: used both for $f$ alone and for $e+f$

**Reparameterization for Gaussian $f$**: while we model the posterior of the form latent $f$ as a Gaussian $q_\phi(f \mid s) = \mathcal{N}\big(f;\, \mu_\phi(s),\, \sigma_\phi^2(s)\big)$, direct sampling $f \sim \mathcal{N}(\mu,\sigma^2)$ is non-differentiable, so we use the **reparameterization trick**:
$$
f = \mu_\phi(s) + \sigma_\phi(s) \odot \epsilon,
\qquad
\epsilon \sim \mathcal{N}(0,I)
$$
This separates the noise ($\epsilon$) from the parameters, enabling gradient backpropagation through $\mu_\phi$ and $\sigma_\phi$

**Gumbel-Sigmoid Relaxation for Binary $e$**: instead we model the posterior of the entity latent $e$ as Bernoulli with logits $l_\phi(s)$: $q_\phi(e_i \mid s) = \text{Bernoulli}(\sigma(l_{\phi,i}(s)))$.  Again binary sampling is non-differentiable, so we use the **Gumbel-Sigmoid (Concrete Bernoulli) relaxation**:
1. Sample Gumbel noise:
$$
g_i = -\log(-\log U_i), \qquad U_i \sim \text{Uniform}(0,1)
$$
2. Compute relaxed sample:
$$
\tilde{e}_i =
\sigma\!\left(\frac{l_{\phi,i}(s) + g_i}{\tau}\right)
$$
- $\tau$ is a temperature parameter (lower $\tau$ is closer to binary)
- During inference, threshold $\tilde{e}_i$ at 0.5 to obtain discrete mask $e_i$
This allows approximate binary sampling with gradients for backpropagation




