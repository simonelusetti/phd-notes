This model assumes that each **token contributes differently** to a set of **latent experts**, each expert being responsible for a specific *semantic factor*.
The model learns to **route** token embeddings into experts via soft assignments, forming an interpretable mixture structure.
# Model
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
\hat{e}_k = R_k(\tilde{f}_k)  
$$
The sentence reconstruction is the **sum over experts:**  
$$  
\hat{e} = \sum_{k=1}^K \hat{s}_k  
$$
## Prototypes
Define prototypes $p_k \in \mathbb{R}^d$, one per expert

Define usage per expert:  
$$ 
u_k = \frac{\sum_t m_t\pi_{t,k}}{\sum_t m_t + \varepsilon}  
$$
Consistency loss
$$
\mathcal{L}_{\text{cons}} = \mathbb{E}_{\text{batch}}\Big[ \sum_k u_k | f_k - p_k |_2^2 \Big]
$$
Prototype update, with decay $\alpha$:
$$
\text{update}_k = \frac{\sum_t m_t,\pi_{t,k} f_k}{\sum_t m_t,\pi_{t,k} + \varepsilon}, \quad
	p_k \leftarrow \alpha p_k + (1-\alpha) \cdot \text{update}_k  
$$
Separation loss: normalize $\hat p_k = p_k / (|p_k|_2+\varepsilon)$, compute cosines $c_{ij} = \hat p_i^\top \hat p_j$, and apply a hinge margin $m$:
$$
\mathcal{L}_{\text{sep}} = \frac{1}{K(K-1)} \sum_{i\ne j} \max\big(0, c_{ij} - (1-m)\big). 
$$
The expert loss adds $\lambda_{\text{cons}}\mathcal{L}_{\text{cons}} + \lambda_{\text{sep}}\mathcal{L}_{\text{sep}}$. Prototypes thus anchor each expert's factors across sentences and push different experts' prototypes apart
# Losses
Starting from the SBERT embedding of the sentence $e(s)$ and that of the reconstructed $\hat{e}$ 

Contrastive, base loss to reconstruct the sentence, the others are regularization terms
$$
\mathcal{L}_{\text{sent}} = \text{CE}(e(s)^\top\hat{e}, I)
$$
Overlap, forces the model to push each token to one single factor
$$
\mathcal{L}_{\text{overlap}} = \mathbb{E}_t[1 - \sum_k \pi_{t,k}^2]
$$
Balance, forces the model to spread out its decision to avoid collapsing on a single factor
$$b_k = \mathbb{E}_t[\pi_{t,k}] \quad \mathcal{L}_{\text{balance}} = | b - \tfrac{1}{K}\mathbf{1}|_2^2
$$
Continuity, forces the model to assign contiguous tokens to the same factor
$$
\mathcal{L}_{\text{cont}} = \mathbb{E}_t |\pi_{t} - \pi_{t-1}|_2^2
$$
Final Loss
$$
\mathcal{L} = \lambda_{sent}\mathcal{L}_{\text{sent}} + \lambda_{overlap}\mathcal{L}_{\text{overlap}} + \lambda_{balance}\mathcal{L}_{\text{balance}} + \lambda_{cont}\mathcal{L}_{\text{cont}} + \lambda_{cont}\mathcal{L}_{\text{cont}}
$$
