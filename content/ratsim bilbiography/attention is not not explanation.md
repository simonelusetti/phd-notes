[https://arxiv.org/pdf/1908.04626] (ACL)

Question: *essentially a rebuttal to Attention is not Explanation. They correct some of the claims of the previous paper and show that it can be used as explanation in some contexts*
# Formal definition of attention

They use the same definition used by the previous paper: [[attention is not explanation#Formal definition of Attention]]
# Main Claims and Rebuttals

**Swapping attention without retraining:** J&W threat attention as a modular function to be plucked and plugged, this is not logically sound as attention is trained alongside the rest of the model and just swapping it is not a meaningful operation

**Existence Vs Exclusivity of Explanation:** J&W claim that the existence of an adversarial explanation shows irrelevance of the attention as explanation, but just shows that it's not *the* explanation, not *an* explanation. This is especially relevant to binary classification, where a dense large feature space is reduced to a single scalar
# Experiment 1 — Uniform Attention as a Mathematical Baseline

Define uniform attention:
$$
\alpha_{uni} = \left[\frac{1}{T},\dots,\frac{1}{T}\right]
$$
Run the model with this fixed uniform attention, removing all capacity of the attention mechanism. If: $\hat{y}(x,\alpha_{uni}) \approx \hat{y}(x,\alpha_\text{trained})$ then attention is irrelevant, and cannot be used for explanations (faithful or plausible)

Table on page 4 (Table 2) shows:
- IMDB, SST, MIMIC: learned attention $>$ uniform
- But on AG News, 20News: learned attention $\approx$ uniform

Thus some datasets cannot meaningfully test attention explainability at all. This is already a mathematical correction of J&W: attention is not explanation if the model simply doesn't use it

# Experiment 2 — Expected Variance from Random Seeds

J&W adversarial construction is based on a chosen tolerance for the distance in output distribution, but to make sense we need a baseline divergence to calibrate this tolerance against

- Let: $\alpha^{(s)} =$ attention distribution learned by seed $s$
- Define seed variance baseline: $\mathrm{JSD}(\alpha^{(s)},\alpha^{(s')})$

Across multiple seeds (page 5–6, Fig. 3), they show:
- SST and IMDB have **low seed-to-seed attention variance**
- Diabetes has **high negative-class variance even without adversarial manipulation**

Thus, adversarial JSD values from J&W cannot be interpreted unless compared to:

baseline variance due to initialization.\text{baseline variance due to initialization}.baseline variance due to initialization.

This is the mathematically correct control
# Experiment 3 - Diagnostic Attention via Non-contextual MLP

The authors build a new, much simpler model that uses no recurrence, contextual info or learned attention. This new model is just MLP over tokens which is allowed to use only the attention weights given to it (frozen)

- For each token $x_t$ predict $z_t = \tanh(W x_t + b), z_t \in \mathbb{R}^d$ 
- Then apply the provided attention distribution $\alpha$: $h_\alpha = \sum_{t=1}^T \alpha_t z_t$ 
- And then predict $\hat{y} = \sigma(\theta^\top h_\alpha)$  

They test 4 kinds of attention weights:
- Uniform: $\alpha_t = \frac{1}{T}$
- LSTM: $\alpha_t^\text{LSTM}(x)$, AKA the original attention weights
- MLP learned: do not freeze $\alpha$ allowing the simple model to learn attention for itself

For all datasets except one outlier: $\text{LSTM-attention} > \text{MLP-learned} > \text{uniform}$. This supports the claim that attention is important, even for a non contextual model, the weights do carry some structure inherent in the data
# Experiment 4 — Coherent Adversarial Attention Training**

Given a model $M_a$, they define an adversarial model $M_b$​ trained with the loss:
$$\mathcal{L} = \mathrm{TVD}(\hat{y}_a, \hat{y}_b) - \lambda \, \mathrm{KL}(\alpha_a \,\|\, \alpha_b)$$
​(with $\hat{y}$ and $\alpha$ being the prediction and attention patterns of each model)

Compared to J&W:
- J&W solve for $\tilde{\alpha}$ only, per-instance without changing model parameters
- W&P solve for global parameters of the entire model over all data

This transforms the question into: *Does a full model exist that produces alternative coherent attention while retaining predictive behavior?*

Results
- For SST and MIMIC datasets, models can achieve JSD $> 0.4$ while keeping TVD $< 0.05$ — _but only moderately so_
- For IMDB, adversarial models cannot push JSD as far as J&W’s per-instance adversaries

**Crucial finding:** Adversarial attention trained coherently fails in the diagnostic MLP (Table 3 bottom row)
# Defining Explainability

The section argues that “explainability” in AI is not a single concept, but actually involves three distinct notions:
- Transparency: Understanding how specific components of a model map to human-interpretable concepts.  Under Lipton’s definition, attention weights _can_ count as partial transparency because they expose which hidden states the model emphasizes.
- Explainability (Plausible Rationales): Rudin and Riedl describe explanations as plausible stories that help humans understand decisions, even if they are not exact reconstructions of internal computations. Extractive rationales (like attention weights or selected input tokens) are often acceptable as explanations, especially when validated with human studies.
- Interpretability (Faithful Mechanistic Understanding): This is stronger than explainability: it requires a true, global understanding of how inputs map to outputs (e.g., linear model coefficients).  Achieving interpretability is hard, and attention may not satisfy this strict notion.

J&W require that: *For attention to be an explanation, there must be only one (or few) correct sets of attention weights*

But this requirement only makes sense if one seeks faithful mechanistic interpretability, not plausible explanation.Using the broader definitions discussed above:
- The existence of multiple possible explanations does not invalidate a given explanation
- A human evaluation would be required to judge _plausibility_, which J&W do not perform
Thus their critique does _not_ undermine attention as a source of plausible explanations

Given the strict definition of transparency: *Do high attention weights reliably indicate the tokens that cause the model’s prediction?*

This stricter question is still unresolved, and motivates the adversarial experiments the paper develops.