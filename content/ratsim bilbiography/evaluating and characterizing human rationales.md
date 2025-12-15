[[https://aclanthology.org/2020.emnlp-main.747.pdf?utm_source=chatgpt.com]] (EMNLP)

Question: *Human annotated rationales are often used as gold labels for rationale models. However, are these even good at the automated metrics we use to use to score model extracted rationales?*
# Framework: rationales, masking, and fidelity

A rationale is represented as a binary mask over tokens: $\alpha \in {0,1}^{|x|}$, where:
- $\alpha_j = 1$: token $j$ is included in the rationale
- $\alpha_j = 0$: token $j$ is excluded

Masking is implemented by **removing tokens** (not replacing them with learned embeddings), while preserving special tokens like `[CLS]` and `[SEP]`

Let:
- $p(y \mid x)$: model probability for class $y$ given full input
- $p(y \mid x, \alpha)$: probability when only rationale tokens are kept
- $\hat{y} = \arg\max_y p(y \mid x)$

**Sufficiency:** Sufficiency measures whether the rationale alone is enough to support the prediction. Formally: $\text{Suff}(x, \hat{y}, \alpha) = 1 - \max\bigl(0, p(\hat{y}\mid x) - p(\hat{y}\mid x,\alpha)\bigr)$

- If removing non-rationale tokens barely changes confidence → high sufficiency
- The metric is **clipped** to stay in ([0,1])

**Comprehensiveness:** Comprehensiveness measures whether the rationale contains _all_ critical information: $\text{Comp}(x, \hat{y}, \alpha) = \max\bigl(0, p(\hat{y}\mid x) - p(\hat{y}\mid x,1-\alpha)\bigr)$

- If removing the rationale causes confidence to drop sharply → high comprehensiveness
- Empty rationales for majority classes can trivially appear “sufficient” but non-comprehensive

A faithful rationale is **expected** to score high on both.
# Experimental setup

The authors analyze **six datasets** with human rationales:
- **Classification tasks**
    - WikiAttack (token-level, class-asymmetric)
    - SST (token-level, derived rationales)
    - Movie Reviews (token-level, non-comprehensive by design)
- **Document / query-style tasks**
    - MultiRC (sentence-level, comprehensive)
    - FEVER (sentence-level, non-comprehensive)
    - E-SNLI (token-level, class-asymmetric)

These datasets vary along critical axes:
- rationale length (11%–35% of tokens)
- granularity (token vs sentence)
- expected completeness
- class asymmetry

This diversity is **essential** to the paper’s conclusions.

Four model families are evaluated:
1. Logistic regression (bag-of-words)
2. Random forests
3. BiLSTM (1 layer)
4. RoBERTa-base (fine-tuned)

- Fidelity is evaluated **post hoc**, with no rationale-aware training
- RoBERTa is used as the main reference model due to accuracy dominance

This choice is deliberate: the paper aims to study **evaluation artifacts**, not to optimize explanations.

# Core findings

**Human rationales are often NOT sufficient:** Across datasets, sufficiency decreases as model accuracy increases

In particular:
- RoBERTa — the most accurate model — often shows lower sufficiency
- Logistic regression and random forests sometimes show higher sufficiency simply due to low confidence calibration

This reveals a paradox:
> Better models make human rationales look worse under sufficiency metrics

This makes sense: if we measure comprehensiveness as the difference between prediction score with full input vs using only the rationale, a model that is very good with full information will have lower comprehensiveness even if their performance is still good using only the rationale, compared to a model which is bad at both

**Comprehensiveness is highly class-dependent:** The study shows dramatic variation by class:
- WikiAttack “no-attack” class has near-zero comprehensiveness (empty rationales)
- FEVER and Movie show asymmetric effects between positive/negative classes

These effects often reflect **model bias**, not rationale quality.

## Normalization: fixing a broken metric:

The authors introduce the **null difference** :  
$$
\text{NullDiff}(x,\hat{y}) = \max\bigl(0, p(\hat{y}\mid x) - p(\hat{y}\mid x,0)\bigr)  
$$
This captures how much confidence the model has _without any input_.

Interpretation:
- Strong class priors → large null difference
- Sufficiency without normalization is meaningless without this baseline

They define:  
$$
\text{NormSuff}(x,\hat{y},\alpha)  
= \frac{\text{Suff}(x,\hat{y},\alpha) - \text{Suff}(x,\hat{y},0)}{1 - \text{Suff}(x,\hat{y},0)}  
$$
$$
\text{NormComp}(x,\hat{y},\alpha)  
= \frac{\text{Comp}(x,\hat{y},\alpha)}{\text{Comp}(x,\hat{y},1)}  
$$

After normalization:
- Apparent paradoxes disappear
- Empty rationales are exposed as uninformative
- Fidelity becomes **interpretable across classes**

This is probably the most important innovation from this paper, and we should introduce this too somehow 
## Accuracy-based sufficiency

The authors compare three regimes:
1. Train on full text, test on full text
2. Train on full text, test on rationales
3. Train on rationales, test on rationales

Surprising result:
- Rationale-only training **sometimes outperforms full-text training**
- “Insufficient” rationales can improve accuracy

This shows that:
> Probability-based sufficiency does not predict _practical usefulness_.

## Fidelity curves: diagnosing redundancy and dependency

They introduce **fidelity curves**:
- Randomly remove increasing fractions of rationale tokens
- Track normalized sufficiency and comprehensiveness

Interpretation:
- Slow sufficiency drop → redundancy or irrelevance
- Fast sufficiency drop → brevity or dependency
- Curve shapes disentangle these cases

Empirical finding:
- Classification datasets → redundant rationales
- Document/query tasks → dependent rationales

This is a **diagnostic lens**, not a scalar score 
# Conclusions

This paper establishes several principles:
1. **Human explanations are not gold**  
    → explains why attention or semantic importance may diverge from annotations
2. **Evaluation metrics are model-relative**  
    → directly supports your skepticism toward static NER evaluation
3. **Faithfulness is multidimensional**  
    → motivates structural, semantic, or perturbation-based approaches
4. **Redundancy vs dependency matters**  
    → highly relevant for span-based and discontinuous entity reasoning
5. **Normalization is essential**  
    → any future “semantic impact” metric must be baseline-aware
