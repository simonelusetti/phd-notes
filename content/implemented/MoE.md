This model assumes that each **token contributes differently** to a set of **latent experts**, each expert being responsible for a specific “semantic factor.”  
The model learns to **route** token embeddings into experts via soft assignments, forming an interpretable mixture structure.

---
### Mathematical Formulation
Each sentence is represented as token embeddings  
$$  
E = [e_1, e_2, \dots, e_T] \in \mathbb{R}^{T \times d}  
$$

**Step 1 — Routing (Gating)**
A learned gate maps each token to expert logits:  
$$  
G_t = W_2 , \sigma(W_1 e_t) \in \mathbb{R}^{K}  
$$
where $K$ is the number of experts.

Then, **soft routing weights** are obtained via **Softmax routing:**  
$$  
\pi_{t,k} = \frac{\exp(G_{t,k})}{\sum_j \exp(G_{t,j})}  
$$
**Gumbel routing (Optional):** adds Gumbel noise to approximate discrete selection.

These weights form the tensor $\Pi \in \mathbb{R}^{B \times T \times K}$

**Step 2 — Token-to-Expert Aggregation**
Each expert receives a weighted combination of tokens:  
$$  
f_k = \frac{\sum_{t=1}^{T} \pi_{t,k} e_t}{\sum_t \pi_{t,k} + \varepsilon}  
\quad \text{for } k=1,\dots,K  
$$
producing the **factor embeddings** $F = [f_1, \dots, f_K] \in \mathbb{R}^{K \times d}$

**Step 3 — Expert Transformations**
Each expert has a (possibly shared) transformation:  
$$  
\tilde{f}_k = T_k(f_k)  
$$
with $T_k$ = MLP (Linear–GELU–Linear), mapping to a **factor dimension** $d_f$, so we get $\tilde{F} = [\tilde{f}_1, \dots, \tilde{f}_K] \in \mathbb{R}^{K \times d_f}$

**Step 4 — Reconstruction**
Each expert has a reconstruction head $R_k$ that projects $\tilde{f}_k$ back to the **sentence embedding space** $d_s$:  
$$  
\hat{s}_k = R_k(\tilde{f}_k)  
$$

The sentence reconstruction is the **sum over experts:**  
$$  
\hat{s} = \sum_{k=1}^K \hat{s}_k  
$$ 
Optionally, tokens can also be reconstructed via:  
$$  
\hat{E} = \sum_k \pi_{t,k} T_k(f_k)  
$$
**Step 5 — Regularization Terms**
The model includes **structured penalties** to control routing and factor orthogonality:

| Regularizer | Expression                                         | Purpose                               |
| ----------- | -------------------------------------------------- | ------------------------------------- |
| Entropy     | $H(\Pi) = -\sum_t \sum_k \pi_{t,k} \log \pi_{t,k}$ | Encourage confident or smooth routing |
| Overlap     | $O = \frac{1}{2}(1 - \sum_k \pi_{t,k}^2)$          | Encourage diversity per token         |
| Balance     | $\sum_k (\bar{\pi}_k - \tfrac{1}{K})^2$            | Encourage uniform expert usage        |
| Diversity   | $\|\text{offdiag}(F^\top F)\|_F^2$                 | Encourage orthogonal experts          |

>**Core Idea:** It learns a **mixture decomposition** of the sentence through differentiable routing. Each expert captures a latent “semantic axis,” and the sum of reconstructions aims to rebuild the anchor sentence embedding from the distributed factors.