[https://arxiv.org/pdf/1902.10186] (ACL)

Question: *Attention is commonly (even if implicitly) used as a proxy for importance of tokens and as an explanation method. Is this correct?*
# Formal definition of Attention

1. Input: you observe a sequence of tokens $x = (x_1, \ldots, x_T), x_t \in \mathbb{R}^{|V|}$ (one-hot).
2. Embedding produces $x^e = (x^e_1,\ldots,x^e_T) \in \mathbb{R}^{T\times d}$
3. Encoder: a function $h = \text{Enc}(x^e)$ returns contextual hidden states $hh_t \in \mathbb{R}^{m}, \quad h = (h_1,\ldots,h_T)$
	1. In most experiments Enc is a **BiLSTM**, meaning $h_t = \big[\overrightarrow{h_t};\overleftarrow{h_t}\big]$
4. Attention: given a query vector $Q\in\mathbb{R}^m$, attention scores are calculated in two different manners:
	1. Additive attention (Bahdanau): $e_t = v^\top \tanh(W_1 h_t + W_2 Q)$
	2. Scaled dot-product (Transformer-like): $e_t = \frac{1}{\sqrt{m}}\, h_t^\top Q$
5. Attention distribution: $\hat{\alpha} = \text{softmax}(e_1,\ldots,e_T)$
6. Context vector and prediction: $h_{\hat{\alpha}} = \sum_{t=1}^T \hat{\alpha}\ h_t$
7. Output: $\hat{y} = \sigma(\theta^\top h_{\hat{\alpha}})$, with $\sigma =$ sigmoid or softmax depending on task

# What Would Be Required for Attention to Be an “Explanation”?

For attention weights to be _faithful explanations_, the following should be true:
1. **Correlation requirement:**  
    Attention weights should correlate with some (classical) notion of **feature importance**, the studied ones are two
    - gradient importance: $g_t = \left| \frac{\partial \hat{y}}{\partial x_t} \right|$
    - leave-one-out (erasure) importance: $\Delta \hat{y}_t = \text{TVD}\left( \hat{y}(x), \hat{y}(x_{-t}) \right)$
2. **Counterfactual uniqueness requirement:**  
    If you change the attention distribution to some $\tilde{\alpha}$, then the output should change in a corresponding way: $\tilde{\alpha} \ne \hat{\alpha} \Rightarrow \hat{y}(x,\tilde{\alpha}) \ne \hat{y}(x,\hat{\alpha})$
The paper shows **both fail** for standard attention
 
>**Not in the paper:** the kendall $\tau$ score: a metric to compare ranking lists that considers only the relative ranking and not the absolute scores.
>Given two rankings $\{a_i\}$ and $\{b_i\}$, for each pair of $(i,j)$ such that $i<j$ we consider them concordant if $(a_i​−a_j​)(g_i​−g_j​)>0$ or discordant if $(a_i​−a_j​)(g_i​−g_j​)<0$. Ties do not contribute
>Let $C$ = number of concordant pairs, $D =$ number of discordant pairs, $N =$ number of total pairs. The kendall $\tau$ score for the two rankings is $\tau = (C-D)/N$, going from full agreement $1$ to full disagreement $-1$
# Gradient-Based Feature Importance

Define for each token $t$
$$
g_t = \sum_{w=1}^{|V|} \mathbf{1}[x_{tw} = 1]\; \left| \frac{\partial \hat{y}}{\partial x_{tw}} \right|
$$

This is the magnitude of the gradient of the output wrt. the one-hot vector of token $t$. The Kendall $\tau$ correlation between attention and gradient importance is empirically small (Table 2) (∼0.1–0.4 for BiLSTM on many datasets), showing **weak monotonic relationship**
## Leave-One-Out Importance (Feature Erasure)

Define:
- Remove word $t$: $x_{−t}$
- Compute the change in prediction using **total variation distance**:
$$
\Delta \hat{y}_t = \text{TVD}\big(\hat{y}(x), \hat{y}(x_{−t})\big) = \frac{1}{2} \sum_{i=1}^{|Y|} \Big| \hat{y}_i(x) - \hat{y}_i(x_{−t}) \Big|
$$
Again $\tau$ between LOO and attention is small for BiLSTMs 
## Key result
They show that:
$$
\tau(\text{LOO}, \text{grad}) \gg \tau(\text{attention}, \text{grad})\quad\text{and}\quad \tau(\text{LOO}, \text{grad}) \gg \tau(\text{attention}, \text{LOO})
$$
So **gradient and erasure agree with each other much more than either agrees with attention** (Figures 3–5).
# Counterfactual Attention via Permutation

If attention weights were a reliable explanation metric, then the prediction should be very sensible to meddling with them. Two study this they modify attention in two ways:
- randomly shuffling the weights
- creating an ad-hoc new attention heatmap that is as far as possible from the original one while keeping the same prediction, which they call **adversarial attention**
## Counterfactual Attention via Permutation 

The simplest test:
1. Keep encoder representation $h$ fixed.
2. Shuffle the attention weights: $\alpha^{(p)} = \text{Permute}(\hat{\alpha})$
3. Compute output difference: $\Delta^{(p)} = \text{TVD}(\hat{y}(x,\hat{\alpha}), \hat{y}(x,\alpha^{(p)}))$

Result (Fig. 6): even when the **maximum attention weight is large**, permuting attention barely changes the prediction.
## Adversarial Attention (Main Mathematical Contribution)

This is the strongest result (Section 4.2.2).

Goal: find **new attention distributions** $\alpha^{(1)},\ldots,\alpha^{(k)}$ that are:
- As **far as possible** from original attention (in JSD)
- Mutually different
- Yet produce **almost the same prediction** (here defined as being up to $\varepsilon$ away from the original prediction distribution)

The resulting optimization Problem (Equation 1–2) is the following:
- Maximize:
$$
f(\{\alpha^{(i)}\}) = \sum_{i=1}^k \text{JSD}(\alpha^{(i)}, \hat{\alpha}) \;+\; \frac{1}{k(k-1)} \sum_{i<j} \text{JSD}(\alpha^{(i)}, \alpha^{(j)})
$$
- subject to:
$$
\forall i\ \text{TVD}\big(\hat{y}(x,\alpha^{(i)}), \hat{y}(x,\hat{\alpha})\big) \le \varepsilon
$$

>**Not in the paper:** they use the JSD (**Jensen–Shannon Divergence**) to score the distance in prediction distribution. This metric is a symmetric and bounded version of the KLD
>- Given two discrete probability distributions $P = (p_1,\ldots,p_T), Q = (q_1,\ldots,q_T)$ (that both sum up to $1$)
>- Define the midpoint distribution: $M = \frac{1}{2}(P + Q)$
>- Then the Jensen–Shannon Divergence is: $\text{JSD}(P, Q) = \frac{1}{2} \, \mathrm{KL}(P \,\|\, M) + \frac{1}{2} \, \mathrm{KL}(Q \,\|\, M)$
>Resulting in it being symmetric, a true metric, and most importantly bounded $0\leq \text{JSD}(P,Q)\leq \log 2 \approx 0.6931$. Even when $p_t$ or $q_t = 0$ (which is usually the case for attention weights)

They relax this with a Lagrangian penalty:
$$
\max_{\alpha^{(1)},\ldots,\alpha^{(k)}}\; f(\{\alpha^{(i)}\}) + \lambda \sum_{i=1}^k \max\left( 0,\, \text{TVD}(\hat{y}(x,\alpha^{(i)}), \hat{y}(x,\hat{\alpha})) - \varepsilon \right)
$$
This is optimized using **Adam** with $\lambda=500$

Result (Figure 7): we see that the **maximum JSD** achievable while keeping prediction within $\varepsilon$ is often near the theoretical maximum ($\approx$ 0.69). You can radically change attention weights and the prediction barely moves. This is a mathematical proof-of-concept that **attention is not the causal pathway** responsible for predictions
### Relation to "picky" attention

Attention heatmaps with concentrated masses around few features are intuitively understood as strong explanation proxies as they show a clear "preference" shown by the model for just a few tokens. One might hope that such attention weights would be more sensible to both random permutation and adversarial attention swap than more spread out heatmaps

This is shown in Figure 6 and 8 to not be the case, or very weakly so
# Conclusion and Limitations

The main conclusion is that using attention heatmaps as a direct, "human readable", proxy for the model's decision is not idea

The main limitations of the study are:
- They use gradient and feature erasure as the baseline for how explainable a result is. While commonly used, there's no guarantee that they should align with "human" explainability themselves (although they mention how the correlation between these two does suggest so)
- The kendall $\tau$ score itself could be mudded by noisy low signal features (but again, it would so when measuring the score between grad and LOO)
- The results are mostly concerning RNNs (BiLSTM) and attention, simple feedforwards seem to do better
- Particularly for adversarial attention: its existence doesn't necessarily show the original attention is useless, they might both be "sufficient" (interchangeable) explanations