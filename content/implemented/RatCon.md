**Main idea:** we do not have labels for entities, but we reason that if we remove an entity word that should disproportionally change the sentence's embedding. We use **rationale selection**, a developed theory about selecting a subset of tokens to perform a task. Since we do not have such a task (unsupervised problem) we consider the problem as data augmentation: *which tokens, if removed, change the sentence's embedding the most?*
# Model

#### Embedding
Each sentence is tokenized and encoded with a pretrained **Sentence-BERT** encoder:
$$  
H = [h_1, \dots, h_L] \in \mathbb{R}^{L \times d}, \quad m \in {0,1}^L  
$$
where $h_i$ are contextual embeddings and $m$ is the attention mask.
#### HardKuma Selector
The selector projects token embeddings to parameters $\alpha_i, \beta_i$:
$$
\alpha_i, \beta_i = \text{Softplus}(W_2 ,\text{GELU}(W_1 h_i)) + 1, \quad \alpha_i,\beta_i \in [1,10]  
$$
The **HardKuma sampler** produces a continuous gate variable:
$$  
g_i \sim \text{Kuma}(\alpha_i, \beta_i), \quad g_i \in (0,1)  
$$
- At training time this is reparameterized stochastically
- At inference it approximates a Bernoulli mask.
#### SBERT Encoding
Three sentence-level vectors are derived via the SBERT pooling module:
- **Anchor** (all tokens):  $h^{\text{anc}} = \text{Pool}(H, m)$
- **Rationale** (selected tokens): $h^{\text{rat}} = \text{Pool}(H, m \odot g)$  
- **Complement** (unselected tokens): $h^{\text{cmp}} = \text{Pool}(H, m \odot (1{-}g))$  
#### Losses
**InfoNCE (NT-Xent)**
Given a batch of size (B), anchors vs. rationals are contrasted:
$$  
\mathcal{L}_{\text{rat}} = -\frac{1}{B} \sum_{i=1}^B \log \frac{\exp\big( \tfrac{h^{\text{anc}}_i \cdot h^{\text{rat}}_i}{\tau}\big)}{\sum_{j=1}^B \exp\big( \tfrac{h^{\text{anc}}_i \cdot h^{\text{rat}}_j}{\tau}\big)}  
$$
Similarly, anchors vs. complements:
$$
\mathcal{L}_{\text{cmp}} = -\frac{1}{B} \sum_{i=1}^B \log \frac{\exp\big( \tfrac{h^{\text{anc}}_i \cdot h^{\text{cmp}}_i}{\tau}\big)}{\sum_{j=1}^B \exp\big( \tfrac{h^{\text{anc}}_i \cdot h^{\text{cmp}}_j}{\tau}\big)}  
$$
**Regularization**
- **Sparsity loss** encourages few tokens to be selected:  
$$  
\mathcal{L}_{\text{sparsity}} = \frac{\sum_i m_i g_i}{\sum_i m_i}  
$$
- **Total variation** encourages contiguous selections:  
$$  
\mathcal{L}_{\text{TV}} = \frac{\sum_{i=1}^{L-1} |(g_{i+1}-g_i) m_i m_{i+1}|}{\sum_i m_i}  
$$

**Total Loss:**
$$  
\mathcal{L} = \mathcal{L}_{\text{rat}} - \lambda_{\text{cmp}} \mathcal{L}_{\text{cmp}} + \lambda_s \mathcal{L}_{\text{sparsity}} + \lambda_{\text{tv}} \mathcal{L}_{\text{TV}}  
$$

## Improvements

#### Attention Augment
**💡Idea:** the model could have an hard time distinguishing between important nouns, verbs, adjective, and so on

**⚙ Application:** each token receives two extra scalar features $\text{in}_i, \text{out}_i$ which represents how much incoming and outgoing attention they have, possibly giving the model enough info to separate those words

#### Fourier Filters
**💡Idea:** apply Fourier transform's filters to remove superficial "syntactic" info about the sentence embedding

**⚙ Application:** each pooled vector is passed through a **frequency-domain filter**: 
$$  
\tilde h = \mathcal{F}^{-1}( M \odot \mathcal{F}(h)),  
$$
Where $\mathcal{F}$ is the FFT and $M$ is a frequency mask (lowpass, highpass, or bandpass).
#### Dual (+) Models
**💡Idea:** use multiple models at the same time, but force them to develop different rationales, each should learn only one of nouns, verbs, adjectives, etc etc

**⚙ Application:** their gate distributions are aligned with a symmetric KL:
$$  
\mathcal{L}_{\text{KL}} = \frac{1}{2}D_{\text{KL}}(g^{(1)} | g^{(2)}) + \frac{1}{2}D_{\text{KL}}(g^{(2)} | g^{(1)})  
$$
The total loss becomes:
$$  
\mathcal{L} = \sum_{k=1}^2 \Big(\mathcal{L}^{(k)}_{\text{rat}} - \lambda_{\text{cmp}} \mathcal{L}^{(k)}_{\text{cmp}} + \lambda^{(k)}_s \mathcal{L}^{(k)}_{\text{sparsity}} + \lambda_{\text{tv}} \mathcal{L}^{(k)}_{\text{TV}} \Big) + \lambda_{\text{KL}} \mathcal{L}_{\text{KL}}  
$$
