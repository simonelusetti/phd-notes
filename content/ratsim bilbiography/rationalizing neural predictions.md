[https://arxiv.org/pdf/1606.04155] (ACL)

Question: *Given a sequence of tokens labelled with a single vector, what is the minimal essential subset that sequence?*
# Problem setup and notion of a “rationale”

Our samples are an input sequence of word embeddings $x = (x_1,\dots,x_L), x_t \in \mathbb{R}^d$ with target vector $y \in \mathbb{R}^m$

The explored contexts are:
- Multi-aspect sentiment: $y$ contains normalized ratings for aspects (appearance, smell, palate, taste, overall)
- QA retrieval: $y$ is not given explicitly; instead, there are similarity constraints (positive vs negative pairs)

A **rationale** is defined by a binary selection vector $z = (z_1,\dots,z_L),\ z_t \in {0,1}$. Where $z_t = 1$ means “keep token $t$” and $z_t = 0$ means “erase token $t$”. The rationale is the kept subsequence


The two desiderata for a rationale are:
1. **Short / coherent**: few tokens (preferably contiguous)
2. **Sufficient**: the model can predict $y$ correctly _using only_ the selected tokens.

These desiderata are enforced in a unsupervised setting with no gold labels, however the model needs a target to be trained
# Model

There are two parts to the model: the encoder and the generator

The encoder is a neural network $\text{enc}_\theta(\cdot) : \text{sequence} \to \mathbb{R}^m$, with parameters $\theta_e$. For an input $x$, the prediction is: $\hat{y} = \text{enc}_\theta(x)$

The generator is a probabilistic model $\text{gen}_\phi(x) = p_\phi(z\mid x)$ parameterized by $\theta_g$ (denoted as $\phi$ when convenient). It outputs a distribution over all binary vectors $z$ of length $L$, i.e. over all possible subsets of tokens

They consider two variants: independent and dependent selection

The first assume each $z_t$ is conditionally independent given the whole input: $p(z\mid x) = \prod_{t=1}^L p(z_t \mid x)$ and goes as follows:
- To get $p(z_t\mid x)$, they use a **bidirectional RNN** over the input: $\overrightarrow{h}_t = \overrightarrow{f}(x_t,\overrightarrow{h}_{t-1}), \overleftarrow{h}_t = \overleftarrow{f}(x_t,\overleftarrow{h}_{t+1})$
- Then a logistic layer: $p(z_t = 1\mid x) = \sigma_z\left(W_z [\overrightarrow{h}_t;\overleftarrow{h}_t] + b_z \right)$
- So each token’s selection probability depends on its context via BiRNN, but **selections are independent**

In the dependent (recurrent) case, to encourage phrase-level selections and avoid weird patterns, they also define a **dependent** model: $p(z\mid x) = \prod_{t=1}^L p(z_t \mid x, z_1,\dots,z_{t-1})$
- They introduce an additional recurrent state $s_t$ that summarizes past selections: $p(z_t = 1\mid x,z_{1:t-1}) = \sigma_z\left(W_z [\overrightarrow{h}_t;\overleftarrow{h}_t; s_{t-1}] + b_z\right)$ 
  $s_t = f_z([\overrightarrow{h}_t;\overleftarrow{h}_t; z_t], s_{t-1})$
- This defines a **stochastic policy** over sequences of selections, essentially a small RNN “controller” deciding which tokens to pick
# Objective: short and sufficient rationales

For a given sampled selection $z$, they define the **rationale cost** $\text{cost}(z,x,y) = L(z,x,y) + \Omega(z)$

Where:
- $L(z,x,y) = |\text{enc}(z,x) - y|_2^2$ (sufficiency: rationale alone should predict (y)),
- $\Omega(z)$ is a **regularizer** enforcing brevity and contiguity: $\Omega(z) = \lambda_1 |z|_1 + \lambda_2 \sum_{t} |z_t - z_{t-1}|$

We can interpret each member as:
- Sufficiency cost: $|z|_1 = \sum_t z_t$ = number of selected tokens → penalized by $\lambda_1$
- Sparsity cost: $\sum_t |z_t - z_{t-1}|$ penalizes **boundaries** (transitions 0→1 or 1→0)
    - If you select a contiguous block like `00011111000`, you have 2 transitions
    - Many disjoint selections lead to many transitions → higher penalty → encourages **spans** and not scattered tokens

# Training objective

>**Not in the paper:** the following is basically a description of the REINFORCE style reinforcement learning schema, it's not fully needed to understand this paper, especially since we do not rely on it for our model

Since rationales are not observed, we optimize the **expected cost** under the generator: $\mathcal{J}(\theta_e,\theta_g) = \sum_{(x,y)\in D} \mathbb{E}_{z \sim \text{gen}(x)} [\text{cost}(z,x,y)].$

Equivalently: 
$$
\min_{\theta_e,\theta_g} \sum_{(x,y)\in D} \mathbb{E}_{z\sim p_\phi(z\mid x)} \left[  |\text{enc}(z,x) - y|_2^2 + \lambda_1|z|_1 + \lambda_2\sum_t |z_t - z_{t-1}|  \right]
$$

This is a **joint training** of encoder and generator; the rationale is a latent variable whose distribution is shaped by this loss

## Optimization: REINFORCE-style gradient

The expectation over all $z$ is intractable (exponential by length), so they **sample** from the generator and use a REINFORCE-style estimator:
- Consider a single example $(x,y)$ for generator parameters $\theta_g$: $\frac{\partial}{\partial \theta_g} \mathbb{E}_{z\sim p(z\mid x)}[\text{cost}(z,x,y)] = \sum_z \text{cost}(z,x,y) \frac{\partial p(z\mid x)}{\partial \theta_g}$
- Rewrite using the log-derivative trick: $\frac{\partial p(z\mid x)}{\partial \theta_g} = p(z\mid x)\frac{\partial \log p(z\mid x)}{\partial \theta_g}.$
- So:
$$
\frac{\partial}{\partial \theta_g} \mathbb{E}_{z}[\text{cost}] = \sum_z \text{cost}(z,x,y) p(z\mid x)\frac{\partial \log p(z\mid x)}{\partial \theta_g} = \mathbb{E}_{z\sim p(z\mid x)}\left[ \text{cost}(z,x,y)\frac{\partial \log p(z\mid x)}{\partial \theta_g} \right]
$$

This is precisely the REINFORCE estimator: treat $-\text{cost}$ as a reward, and $\log p(z\mid x)$ as the log policy. In practice, you approximate the expectation by Monte Carlo:
- Sample $K$ rationales $z^{(1)},\dots,z^{(K)}$ from $p(z\mid x)$,
- Use $\frac{1}{K}\sum_{k=1}^K \text{cost}(z^{(k)},x,y)\frac{\partial \log p(z^{(k)}\mid x)}{\partial \theta_g}$

For encoder parameters $\theta_e$: $\frac{\partial}{\partial \theta_e}\mathbb{E}_{z}[\text{cost}(z,x,y)] = \mathbb{E}_z\left[\frac{\partial\text{cost}(z,x,y)}{\partial \theta_e}\right]$. This is easier since you backprop through the encoder given each sampled rationale. They refer to this as a **doubly stochastic gradient** (stochastic over examples and over sampled rationales).

# Experiments

**Multi-aspect sentiment (BeerAdvocate)**
- Input: review text
- Output: normalized ratings $y \in [0,1]^m$
- Base encoder is trained with MSE loss: $L(x,y) = |\text{enc}(x) - y|_2^2$  
- The rationale training uses the same loss, but on $\text{enc}(z,x)$, plus regularizer $\Omega(z)$
Evaluation:
- **MSE** on ratings to judge prediction quality
- **Precision of rationales**: are selected tokens inside sentences manually annotated as belonging to the target aspect?
- They compare:
    - Bigram SVM (feature weights used for rationale)
    - Attention model (soft attention used for selection)
    - Generator (independent vs recurrent)

Result: rationale generators achieve **80–96% precision** vs ~30% for SVM and somewhat lower for attention.

**Question similarity (AskUbuntu)**
- Here encoder is trained with a **hinge loss** on cosine similarity:
	- Positive pair $(q, q^+)$,
	- Negative candidate $q^-$.
	- Define: $s(q,q') = \cos(\text{enc}(q), \text{enc}(q'))$
	- “One-vs-all” hinge loss: $\max(0, 1 - s(q,q^+) + s(q,q^-))$
	- This trains encoder representations
- For rationales:
	- The generator produces rationales $z$ for each question (query and candidates),
	- Encoder sees only rationals
	- They evaluate retrieval quality via **mean average precision (MAP)**.
- They compare:
	- Full title (short, clean text) → upper bound
	- Full body (long, noisy)
	- Rationale-selected subsets of the body (5–30% of tokens)

Rationales reach MAP $\approx$ title level and are much better than using entire body, proving **short subsequences suffice** and are meaningful
# Conclusions

This paper demonstrated that the encoder-generator framework, trained in an end-to-end manner, gives rise to quality rationales in the absence of any explicit rationale annotations

>**Not in the paper:** of note is that while rationales aren't tagged, the task it uses as proxy (in this case similarity and regression) do need a target label, which our model does not