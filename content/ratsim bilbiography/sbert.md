[[https://arxiv.org/pdf/1908.10084]] (EMNLP)

Question: *Can we create a model to encode an entire sentence in a single vector?*
# What problem the paper solves

Vanilla BERT is great at _pair_ tasks (e.g., STS) when used as a **cross-encoder**: it takes _both_ sentences together and predicts a score/label. But that makes large-scale retrieval/clustering expensive because you must run BERT for every pair: for $n=10000$, that’s $n(n-1)/2 \approx 50$ million forward passes, reported as ~65 hours on a V100 GPU.

The paper proposes **SBERT**: a **siamese / triplet** architecture that produces **independent sentence embeddings** so that similarity can be computed cheaply with cosine similarity. Compute embeddings once for all sentences, then similarity search is fast.
# Model: Sentence-BERT (SBERT)

**Encoder + pooling:** Let a sentence be tokenized and passed through BERT/RoBERTa, producing contextual token embeddings: $H = [h_1, h_2, \dots, h_T], h_i \in \mathbb{R}^{d}$

SBERT adds a **pooling** step to turn token embeddings into a single sentence vector $s \in \mathbb{R}^{d}$. They test:
- **CLS pooling:** $s = h_{\text{[CLS]}}$
- **MEAN pooling (default):** $s = \frac{1}{T}\sum_{i=1}^{T} h_i$
- **MAX pooling:** $s_j = \max_{i\in{1..T}} (h_i)_j$

The siamese part means: given sentence $A$ and $B$, both are encoded with **tied weights** (same BERT parameters) to get vectors $u$ and $v$

**Training objectives:** SBERT is trained so that cosine similarity between embeddings reflects semantic similarity/task labels

1. Classification objective (NLI fine-tuning)
	1. They build a feature vector from two embeddings: $x = [u; v; |u-v|] \in \mathbb{R}^{3d}$
	2. Then a linear classifier with softmax: $o = \text{softmax}(W_t x), W_t \in \mathbb{R}^{3d \times k}$, where $k$ is number of labels (e.g., 3 for NLI). Optimize cross-entropy loss. This architecture is shown in Fig. 1 of the paper
	3. **Key ablation result:** the element-wise difference $|u-v|$ is the most important part of the concatenation for NLI training
2. Regression objective (STS fine-tuning)
	1. Compute cosine similarity: $\cos(u,v)=\frac{u^\top v}{|u||v|}$
	2. Use MSE loss against gold similarity score $y$: $\mathcal{L} = (\cos(u,v) - y)^2$
	3. At inference they still use cosine similarity (no expensive cross-encoder)
3. Triplet objective (Wikipedia section triplets)
	1. Given anchor $a$, positive $p$, negative $n$, with embeddings $s_a,s_p,s_n$, minimize: $\mathcal{L}=\max(|s_a-s_p|-|s_a-s_n|+\epsilon,0)$ using Euclidean distance, with margin $\epsilon=1$ in their experiments

**Training details:** For their main “SBERT-NLI” models they train on **SNLI + MultiNLI** for **1 epoch** using: batch size 16, Adam, learning rate $2\times10^{-5}$, linear warmup over 10% of training data, default MEAN pooling
# Datasets

**Natural Language Inference (for embedding supervision)**
- **SNLI:** ~570k sentence pairs labeled contradiction, entailment or neutral
- **MultiNLI:** ~430k sentence pairs, multiple genres, same label set
- Used to train SBERT with the **classification objective** to obtain generally useful semantic embeddings

**Semantic Textual Similarity (evaluation + supervised fine-tuning)** They evaluate on standard STS datasets where sentence pairs have similarity labels **0–5**:
- **SemEval STS 2012–2016** (STS12–STS16)
- **STS Benchmark (STSb)**: 8,628 pairs total across captions/news/forums; split into train 5,749 / dev 1,500 / test 1,379.
- **SICK-Relatedness (SICK-R)**

**Metric:** They use **Spearman rank correlation $\rho$** between cosine similarities and gold labels (they argue Pearson is less suitable)

>**Not in the paper: the spearman coefficient**
>Spearman’s $\rho$ measures how well two variables preserve the **same ordering**, regardless of scale or linearity
>Given two vectors $X$ and $Y$:
>1. Convert values to ranks $R(X)$ and $R(Y)$  
>2. Compute Pearson correlation on the ranks: $\rho = \mathrm{corr}(R(X), R(Y))$
>If there are no ties: $\rho = 1 - \frac{6 \sum_i (r_{x_i} - r_{y_i})^2}{n(n^2 - 1)}$
>Interpretation
>$\rho = 1$ → identical ordering  
>$\rho = 0$ → no monotonic relation  
>$\rho = -1$ → reversed ordering  
>Why it’s used for embeddings: 
>- Works on **ranks**, not raw cosine values
>- Robust to scaling and non-linearity
>- Ideal for **retrieval / similarity** evaluation  

**Argument Facet Similarity (AFS)**
- **AFS corpus:** ~6,000 argument pairs from social media dialogs on **gun control, gay marriage, death penalty**, scored 0–5. Similarity requires matching both _claims_ and _reasoning_, not just topical overlap.  
    Evaluated in:
1. **10-fold cross-validation** (as in prior work)
2. **Cross-topic** generalization: train on 2 topics, test on the held-out topic.

**Wikipedia section triplets (weakly supervised triplet data)**
- From Dor et al. (2018): sentences from the same Wikipedia article section are treated as thematically closer. Create triplets:
- anchor and positive from same section
- negative from different section of the same article. They train on ~1.8M triplets (1 epoch) and test on 222,957 triplets; metric is **accuracy**: is positive closer than negative?

**SentEval transfer tasks (downstream probing)**
- SentEval trains a logistic regression classifier on top of frozen embeddings (10-fold CV) on tasks: MR, CR, SUBJ, MPQA, SST, TREC, MRPC

# Main experimental findings

1. “Vanilla BERT sentence embeddings” are weak for cosine similarity: They compare averaging BERT token outputs or using CLS directly, and show these perform poorly on STS. Often worse than average GloVe—when evaluated with cosine similarity. (See Table 1.)
2. The fact that MEAN pooling outperforms CLS suggests that **entity-relevant information is distributed across tokens rather than localized in a single control token**, which aligns well with your hypothesis that importance emerges from token-level interactions (attention aggregation) rather than from a single summary vector
3. SBERT (trained on NLI) greatly improves STS (unsupervised). On STS12–STS16, STSb, SICK-R: SBERT-NLI and SRoBERTa-NLI get much higher Spearman $\rho$ than InferSent and Universal Sentence Encoder on average (Table 1)
4. Supervised STSb: SBERT is competitive, cross-encoder BERT still strongest. When fine-tuned on STSb, SBERT reaches strong performance, but the cross-encoder BERT variants can score higher because they compare both sentences jointly with attention (Table 2)
5. AFS: SBERT close to BERT in CV, but drops more in cross-topic. In-topic evaluation is strong; cross-topic generalization is harder for SBERT because it must map single sentences from unseen topics into a consistent embedding space (Table 3)
6. Wikipedia triplets: SBERT + triplet loss works well. SBERT trained with triplet loss outperforms the referenced BiLSTM triplet approach on accuracy (Table 4)
7. Efficiency: SBERT enables scalable semantic search. They report high embedding throughput and emphasize the workflow shift: encode once, then use vector similarity search; they also introduce “smart batching” (group by length to reduce padding) for speedups (Table 7)