 We do not have labels for entities, but we reason that if we remove an entity word that should disproportionally change the sentence's embedding. We use **rationale selection**, a developed theory about selecting a subset of tokens to perform a task. Since we do not have such a task (unsupervised problem) we consider the problem as data augmentation: *which tokens, if removed, change the sentence's embedding the most?*
# Model

**Step 1 - Embedding**

Each sentence is tokenized and encoded with a pretrained encoder (chosen from a sbert, e5, bge, ...):
$$  
H = [h_1, \dots, h_L] \in \mathbb{R}^{L \times d}
$$
where $h_i$ are contextual embeddings

**Step 2 - Raw tokens scores**

A lightweight selector network $f$ assigns a **scalar score** to each token: $s_i = f(h_i) = w^\top h_i + b$

Collecting all scores: $s = [s_1, \dots, s_L]$

**Step 3 - Softmaxing the scores**

Token scores are normalized using a temperature-scaled softmax: 

$$p_i = \frac{\exp(s_i / \tau)}{\sum_{j=1}^L \exp(s_j / \tau)}$$

this softmax has the property of $\sum_i p_i = 1$

This induces **global competition** among tokens: increasing one token’s probability necessarily decreases others.

**Step 4 - Budget mass assignment**

Instead of selecting tokens independently, we impose a **per-sentence selection budget**.

Let: $T_{\text{eff}} = \sum_{i=1}^L m_i$ be the number of valid (non-padding) tokens. The selection budget is defined as:

$$
K = \max\left(1, \; \text{round}(\rho \cdot T_{\text{eff}})\right)
$$

where $\rho \in (0,1]$ is a hyperparameter controlling the **fraction of tokens to keep**, a forced selection rate.

We then get **soft gates** for each tokens: $z_i = K \cdot p_i$, satisfying: $\sum_i z_i = K$

Essentially scaling the softmax selection mass from 1, to an arbitrary $K$

**Step 5 – Hard top-$K$ selection**

To obtain a discrete rationale, we convert $z$ into a **hard Top-$K$ mask**. Define:

$$
h_i =
\begin{cases}
1 & \text{if } z_i \text{ is among the top } K \text{ values} \\
0 & \text{otherwise}
\end{cases}
$$

At inference time, this hard mask is used directly. During training, we apply a **straight-through (ST) estimator**:

$$
g = h + \big(z - \text{stopgrad}(z)\big)
$$

This ensures:
- the **forward pass** uses an exact discrete Top-$K$ selection
- the **backward pass** propagates gradients through the continuous variable $z$

The resulting gate vector $g \in \mathbb{R}^L$ is binary but differentiable.

**Step 6 – Masked Sentence Encoding**

The sentence encoder supports **fractional attention masks**, allowing continuous gating. We compute:
- **Target embedding** (full sentence): $e = e(H, m)$
- **Rationale embedding** (selected tokens only): $\hat{e} = e(H, m \odot g)$
# Losses

The selector is trained to preserve the original sentence representation using only the selected tokens:

$$
\mathcal{L}_{\text{rec}} = 1 - \cos(\hat{e}, e)
$$

This encourages the selected subset to retain maximal semantic information.

>Note: lack of regularization losses, such as certainty or sparsity is **by design**. Scaled mass softmax imposes both in hard-coded, structured way:
>	- The softmax pushes high scored tokens to gain mass, and low scored tokens to lose mass, effectively implementing a certainty loss
>	- The top-k hard selection, along with softmax scaling with $\rho$ effectively imposes a sparsity loss, modulated by this hyperparam