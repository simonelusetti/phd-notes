Instead of routing tokens to experts, this model encodes all tokens into a **shared latent space** and then projects them onto several **submanifolds** (product structure):  
$$  
\mathcal{M} = \mathcal{M}_1 \times \mathcal{M}_2 \times \dots \times \mathcal{M}_K  
$$
Each submanifold corresponds to a distinct semantic component, and the **sum of projections** reconstructs the sentence.

**Step 1 — Token Encoding**
Each token is encoded as:  
$$  
z_t = f_{\text{enc}}(e_t) = W_2 \sigma(W_1 e_t) \in \mathbb{R}^{d_z}  
$$
Stacked:  
$$  
Z = [z_1, \dots, z_T] \in \mathbb{R}^{T \times d_z}  
$$

**Step 2 — Product Projection**
Each projector $P_k$ is an MLP (possibly identity-like) mapping latent tokens:  
$$  
z_t^{(k)} = P_k(z_t)  
$$
Resulting in:  
$$  
\text{Subspaces: } Z^{(k)} \in \mathbb{R}^{T \times d_z}, \quad k=1,\dots,K  
$$
and  
$$  
Z^{\text{subspaces}} = \text{stack}(Z^{(1)}, \dots, Z^{(K)}) \in \mathbb{R}^{T \times K \times d_z}  
$$

**Step 3 — Aggregation**
Tokens are aggregated (masked average):  
$$  
\tilde{z}_t = \sum_k z_t^{(k)}, \quad \tilde{z}_t \in \mathbb{R}^{d_z}  
$$
and averaged across tokens to obtain per-subspace factors:  
$$  
f_k = \frac{1}{T}\sum_t M_t z_t^{(k)}  
$$
**Step 4 — Reconstruction**
- **Token-level reconstruction:**  
$$  
\hat{e}_t = f_{\text{dec}}(\tilde{z}_t)  
$$
- **Sentence-level reconstruction:**  
$$  
\hat{s} = \frac{\sum_t M_t \hat{e}_t}{\sum_t M_t}  
$$
>**Core Idea:** Each projector defines a **latent subspace** (manifold component) within the overall embedding space. The model learns to distribute token information across these subspaces **without explicit routing** — all tokens contribute to all subspaces, but differently via the learned projections.
