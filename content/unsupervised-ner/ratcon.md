 We do not have labels for entities, but we reason that if we remove an entity word that should disproportionally change the sentence's embedding. We use **rationale selection**, a developed theory about selecting a subset of tokens to perform a task. Since we do not have such a task (unsupervised problem) we consider the problem as data augmentation: *which tokens, if removed, change the sentence's embedding the most?*
# Model
**Step 1 - Embedding**
Each sentence is tokenized and encoded with a pretrained **Sentence-BERT** encoder:
$$  
H = [h_1, \dots, h_L] \in \mathbb{R}^{L \times d}
$$
where $h_i$ are contextual embeddings

**Step 2 - HardKuma Selector**
The selector projects token embeddings to parameters $\alpha_i, \beta_i$:
$$
\alpha_i, \beta_i = \text{Softplus}(W_2 ,\text{GELU}(W_1 h_i)) + 1, \quad \alpha_i,\beta_i \in [1,10]  
$$
The **HardKuma sampler** produces a continuous gate variable:
$$  
g_i \sim \text{Kuma}(\alpha_i, \beta_i), \quad g_i \in (0,1)  
$$
- At training time this is reparameterized stochastically
- At inference it approximates a Bernoulli mask

**Step 3 - SBERT Encoding**
Three sentence-level vectors are derived via the SBERT pooling module:
- **Anchor** (all tokens):  $e(s) = \text{SBERT}(H)$
- **Rationale** (selected tokens): $\hat{e} = \text{SBERT}(H\odot g)$  
- **Complement** (unselected tokens): $\hat{e}_{\text{comp}} = \text{SBERT}(H \odot (1{-}g))$  
# Losses
Contrastive, base loss to reconstruct the sentence correctly
$$
\mathcal{L}_{\text{sent}} = \text{CE}\big(\hat{e}^\top e(s), I\big)
$$

The following are regularization terms

Complement repulsion, forces the model to actually drop unimportant tokens as it's penalized if it can correctly reconstruct sentences from the complement of the selection
$$\mathcal{L}_{\text{comp}} = -\text{CE}\big(\hat{e}_{\text{comp}}^\top e(s), I\big)$$
Sparsity, forces the model to select only a few tokens
$$
\mathcal{L}_{\text{s}} = \frac{\sum_t g_t m_t}{\sum_t m_t + \varepsilon}
$$
Total variation, forces the model to select contiguous tokens 
$$
\mathcal{L}_{\text{tv}} = \frac{\sum_t |g_t - g_{t-1}| m_t m_{t-1}}{\sum_t m_t m_{t-1} + \varepsilon}
$$
Final loss
$$
\mathcal{L} = \mathcal{L}_{\text{rat}} + \lambda_{\text{comp}}\mathcal{L}_{\text{comp}} + \lambda_{\text{s}}\mathcal{L}_{\text{s}} + \lambda_{\text{tv}}\mathcal{L}_{\text{tv}}
$$
