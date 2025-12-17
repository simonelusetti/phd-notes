# Shared notation
Let a sentence be tokenized into $T$ tokens.

### Token-level representations

Let  $\mathbf{H} = [\mathbf{h}_1, \dots, \mathbf{h}_T]^\top \in \mathbb{R}^{T \times d}$ be the final-layer token embeddings produced by a frozen Transformer encoder

Let  $\mathbf{m} \in {0,1}^T$ be the attention mask, where $m_i = 1$ if token $i$ is valid and $0$ if padding

Define the set of valid tokens: $\mathcal{V} = { i \mid m_i = 1 }, \quad |\mathcal{V}| = N$
# Vanilla mean encoder 

The sentence embedding is the **un-normalized mean** of token embeddings:
$$\mathbf{s}_{\text{vanilla}}
\frac{1}{N}  
\sum_{i \in \mathcal{V}}  
\mathbf{h}_i  
 $$
**Properties**
- Linearity: $\mathbf{s}_{\text{vanilla}}  \text{is linear in } \mathbf{H}$-
- Token contribution: $\frac{\partial \mathbf{s}}{\partial \mathbf{h}_i} \frac{1}{N} \mathbf{I}$

**Geometry**
- No constraint on norm: $|\mathbf{s}|_2$ varies with sentence length and content
- Embedding space is **anisotropic**

**Effect of token removal** Removing token $j$:
 - $\Delta\mathbf{s} = \mathbf{s} - \mathbf{s}_{\setminus j}\frac{1}{N}\mathbf{h}_j\frac{1}{NN-1}\sum_{i \neq j}\mathbf{h}_i$

Large-norm or directionally unique tokens $often named entities$ produce **large drift**

This encoder preserves **maximal token-level information**, but sentence similarity is poorly calibrated
# SBERT-style normalized mean encoder

The sentence embedding is the **L2-normalized mean** of token embeddings:
$$
\mathbf{u}
=
\frac{1}{N}
\sum_{i \in \mathcal{V}}
\mathbf{h}_i \qquad
\mathbf{s}_{\text{SBERT}}
=
\frac{\mathbf{u}}{\|\mathbf{u}\|_2}
$$
**Properties**
- Non-linearity: normalization introduces a global non-linearity
- Token contribution:$$
  \frac{\partial \mathbf{s}}{\partial \mathbf{h}_i}
  \propto
  \left(
  \mathbf{I}
  -
  \mathbf{s}\mathbf{s}^\top
  \right)
  $$
**Geometry**
- Fixed norm: $\|\mathbf{s}\|_2 = 1$
- Cosine similarity is well-defined
- Embedding space is **approximately isotropic**

**Effect of token removal:** Removing token $j$:
$$
\Delta \mathbf{s}
\approx
\frac{1}{\|\mathbf{u}\|}
\left(
\mathbf{I}
-
\mathbf{s}\mathbf{s}^\top
\right)
\frac{\mathbf{h}_j}{N}
$$
Uniform tokens are suppressed, while directionally informative tokens induce **moderate drift**

This encoder preserves entity information **directionally**, while stabilizing sentence similarity
# Retrieval-first encoder (E5 / GTE)

The sentence embedding is the **normalized sum** of token embeddings:
$$
\mathbf{s}_{\text{retrieval}}
=
\frac{
\sum_{i \in \mathcal{V}}
\mathbf{h}_i
}{
\left\|
\sum_{i \in \mathcal{V}}
\mathbf{h}_i
\right\|_2
}
$$
**Properties**
- Non-linearity: induced by normalization and contrastive training
- Token contribution:$$
  I_i
  =
  \left\|
  \frac{\partial \mathbf{s}}{\partial \mathbf{h}_i}
  \right\|_2
  \approx \text{low variance across } i
  $$
**Geometry**
- Fixed norm: $\|\mathbf{s}\|_2 = 1$
- Optimized cosine space
- Embedding space is **globally semantic**

**Effect of token removal:** Removing token $j$:
$$
\|\Delta \mathbf{s}\|_2 \text{ is small and smooth}
$$
Token influence is **diffuse**; semantic factors should be kept only if retrieval-relevant

This encoder intentionally **flattens token-level salience** in favor of global semantic alignment
# Late-compression encoder (BGE / LLM-style)

The sentence embedding is taken from the **last non-padding token**:
$$
\mathbf{s}_{\text{late}}
=
\mathbf{h}_k = \text{last non padding token of attn transformed sequence}
$$
(Optionally normalized) $\mathbf{s}\leftarrow\frac{\mathbf{s}}{\|\mathbf{s}\|_2}$$

**Properties**
- Extreme non-linearity: all information is routed via attention
- Token contribution:
  $$
  \frac{\partial \mathbf{s}}{\partial \mathbf{h}_i}
  =
  \frac{\partial \mathbf{h}_k}{\partial \mathbf{h}_i}
  $$
**Geometry**
- Strong positional bias
- Highly compressed representation
- Embedding space is **highly abstract**

**Effect of token removal**
Two regimes:
- Irrelevant token → negligible effect
- Attention-dominant token → catastrophic shift

Semantic factors should survive **only if promoted by attention routing**

This encoder defines a **hard upper bound** on token-level interpretability
