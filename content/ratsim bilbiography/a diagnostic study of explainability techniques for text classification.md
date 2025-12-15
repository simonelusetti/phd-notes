[[https://aclanthology.org/2020.emnlp-main.263.pdf]] (EMNLP)

Question: *How do we compare different explainability techniques? Do we have sound metrics to do so?*
# Notation and objects

They have a labeled test set with _token-level human rationales_:

 Dataset $X={(x_i, y_i, w_i)}_{i=1}^N$ where:
    - $x_i = (x_{i,1},\dots,x_{i,|x_i|})$ is the token sequence
    - $y_i \in {1,\dots,C}$ is the gold class label
    - $w_i = (w_{i,1},\dots,w_{i,|x_i|})$, with $w_{i,j}\in{0,1}$, is a binary mask of “human-salient” tokens

A trained classifier $M$ outputs class probabilities  $p_i = M(x_i)\in[0,1]^C,p_{i,c}=P_M(c\mid x_i)$, let the predicted class be $k_i=\arg\max_c p_{i,c}$ with confidence $p_{i,k_i}$

An explainability method $\omega$ produces a **saliency score per token and per class**:
$$
\omega^M_{x_i,c} = \left(\omega^M_{(i,1),c},\dots,\omega^M_{(i,|x_i|),c}\right)\in\mathbb{R}^{|x_i|}
$$
Intuition: $\omega^M_{(i,j),c}$ measures how “responsible” token $x_{i,j}$ is for class $c$


They compare:
- A set of **same-architecture models** trained with different random seeds $\mathcal{M}{M_1,\dots,M_K}$
- A set of **randomly initialized** (untrained) models: $\mathcal{M}^{RI}={M^{RI}_1,\dots,M^{RI}_K}$
	- These are used for the consistency diagnostics and for sanity baselines.

# Proposed new Metrics
## 1. Agreement with Human Rationales (HA)

This metric measures the agreement between the explanation given by the technique and the human generated labels. Notably, it's not a measure of faithfulness since it's guided by human agreement, and so is more akin to what is commonly called Plausibility 

We start with instance-level score: Average Precision (AP). For a given instance $i$, you compare:
- binary ground-truth relevance labels $w_{i,j}\in{0,1}$
- continuous saliency scores $s_{i,j} := \omega^M_{(i,j),,y_i}$ (they use the **gold class** $c=y_i)$

By averaging across all instead we get the Mean Average Precision:
$$
AP(\omega, M, X) = \frac{1}{N}\sum_{i∈[1,N]} AP(\omega_i, \omega^M_{x_i,y_i})
$$

## 2. Confidence Indication (CI)

Goal: Can the explanation itself predict model confidence $p_{i,k}$?

For each token $j$, compare the saliency for the predicted class $k$ to saliency for other classes, indicated by $K\setminus k$, defining a “Saliency Distance”. Aggregating over tokens we get:  
$$
SD_i=\sum_{j=1}^{|x_i|} D\Bigl(\omega^M_{(i,j),k_i},\ \omega^M_{(i,j),K\setminus{k_i}}\Bigr)  
$$

$D$ is defined as
- If $K=2$ classes: $D(a,b)=a-b$
- If $K>2$: they build a small feature vector from differences to the other classes. Let
	- $\Delta_{i,j,c}=\omega^M_{(i,j),k_i}-\omega^M_{(i,j),c}\quad \text{for } c\neq k_i$ 
	- Then $D$ is the concatenation of:  $D = \bigl[\max_c \Delta_{i,j,c},\ \min_c \Delta_{i,j,c},\ \mathrm{mean}_c \Delta_{i,j,c}\bigr]$
- So $SD$ is either a scalar (binary case) or a low-dim vector (multi-class case), after summing over tokens.

They fit a logistic regression $LR(\cdot)$ that maps $SD\mapsto \hat{p}_{i,k_i}$

Finally they compute Mean Absolute Error: $MAE(\omega,M,X)=\frac{1}{N}\sum_{i=1}^N \left|p_{i,k_i}-LR(SD_i)\right|$

Lower MAE means **the explanation structure encodes confidence** well

**Why perturbation methods do well here:** SHAP/LIME/Occlusion explicitly measure output changes, so saliency magnitude naturally correlates with confidence

## 3. Faithfulness (F)

This is the perturbation test: “If the saliency says token (j) matters, then removing it should hurt the model.”

### 3.1 Build thresholded perturbation datasets (X^\omega_t)

For thresholds $t\in{0,10,20,\dots,100}$, they create a perturbed dataset $X^\omega_t$ by masking the top $t\%$ tokens (by saliency) **in each instance**.

Let $P(\cdot)$ be a task metric (they use macro F1 for classification). Define baseline performance $P(M(X^\omega_0))$ and performance after masking $t\%$: $P(M(X^\omega_t))$.

They compute a curve of drops: $\Delta(t)=P(M(X^\omega_0))-P(M(X^\omega_t))$
Then compute area under the threshold–performance curve
$$
AUC\text{-}TP(\omega,M,X)=AUC\Bigl({(t,\Delta(t))\mid t\in{0,10,\dots,100}}\Bigr)  
$$

**Direction:** A “good” method causes performance to degrade quickly when you remove top-salient tokens. In their tables they treat _smaller AUC_ as better (because the curve definition is “drop”; depending on plotting convention, smaller can mean steeper early drop—this is exactly why one must read their definition carefully). The important thing is: **they compare to a random saliency baseline** to ensure the explanation is non-trivial.
## 4. Rationale Consistency (RC)

This is the most “structural” diagnostic: explanations should vary _with internal reasoning_, not arbitrarily.

For a model $M$, define an activation representation $A_M(x_i)$, e.g. concatenated/averaged activations across layers and units (they say averaged across layers and neural nodes). Then for two models $M_s, M_p$ (same architecture), define activation distance:
$$
d_A(M_s,M_p;x_i)=D\bigl(A_{M_s}(x_i),A_{M_p}(x_i)\bigr)  
$$
Here $D(\cdot,\cdot)$ is a vector distance (implementation-wise: they use absolute difference and then a norm / aggregation)

For the same instance, compute distance between saliency vectors:$d_\omega(M_s,M_p;x_i)=D\bigl(\omega^{M_s}_{x_i,y_i},\ \omega^{M_p}_{x_i,y_i}\bigr)$

They mention taking absolute value so the distance is invariant to order.

Now collect pairs $\bigl(d_A(M_s,M_p;x_i), d_\omega(M_s,M_p;x_i)\bigr)$ over instances $i$. Compute Spearman’s rank correlation:
$$
\rho(M_s,M_p,X,\omega)=\rho_{Spearman}\Bigl({d_A(M_s,M_p;x_i)}_{i=1}^N,\ {d_\omega(M_s,M_p;x_i)}_{i=1}^N\Bigr)  
$$

High positive $\rho$ means:
- when two models internally behave similarly on an instance, their explanations are similar
- when they behave differently, explanations differ

They include both trained-seed models and random-initialized ones to ensure the comparison spans “near” and “far” rationales 
## 5. Dataset Consistency (DC)

Same principle as RC, but swap “two models on one instance” with “one model on two instances.”

For a fixed model $M$, define: $d_A(M;x_i,x_j)=D\bigl(A_M(x_i),A_M(x_j)\bigr)$
$$
d_\omega(M;x_i,x_j)=D\bigl(\omega^{M}_{x_i,y_i},\ \omega^{M}_{x_j,y_j}\bigr)  
$$
(Implementation detail: labels in saliency extraction matter; the paper’s notation shows $y_i$ but the intent is “compare explanations of the instances under their relevant label”.)
$$
\rho(M,X,\omega)=\rho_{Spearman}\Bigl({d_A(M;x_i,x_j)}_{(i,j)},\ {d_\omega(M;x_i,x_j)}_{(i,j)}\Bigr)  
$$
They do **sampling of pairs** (because all pairs are too many): 2000 high-overlap pairs + 2000 random pairs
# Models, Experimental Setup, and Results

### Models

The experimental analysis is conducted over three widely used neural architectures for text classification: CNNs, LSTMs and Transformers

For each architecture, multiple models are trained using different random seeds to enable the analysis of explanation consistency across models with identical structure but different parameter initializations. In addition, randomly initialized (untrained) versions of each architecture are included as control models, allowing the authors to assess whether explanation methods capture meaningful model behavior rather than architectural artifacts.
### Datasets

The evaluation is performed on three text classification datasets that provide human-annotated token-level rationales:
- The e-SNLI dataset extends the natural language inference task by including free-text human explanations, which are converted into token-level rationale annotations. The task involves sentence-pair classification into entailment, contradiction, or neutrality.
- The Movie Reviews dataset contains long-form reviews annotated with rationales identifying spans relevant to sentiment classification. Compared to the other datasets, this corpus contains substantially longer input sequences, posing additional challenges for explanation methods.
- The Tweet Sentiment Extraction dataset consists of short, informal texts annotated with spans that justify sentiment labels. The brevity of the inputs allows for fine-grained evaluation of token-level saliency alignment.
### Experimental Setup

All models are trained using supervised learning on their respective training splits, with hyperparameters selected via grid search on development sets. Performance is measured using macro-averaged F1 score to account for class imbalance.

The techniques evaluated for explainability are: 
- Random (control group)
- Shapely sample
- LIME
- Occlusion
- Saliency ($\mu$)
- Saliency ($\mathcal{l}_2$) 
- InputXGrad ($\mu$)
- InputXGrad ($\mathcal{l}_2$) 
- GuidedBP ($\mu$)
- GuidedBP ($\mathcal{l}_2$) 

---

### Results

- Across all datasets and architectures, gradient-based explanation methods consistently achieve the strongest overall performance. In particular, gradient saliency and Input×Gradient with $\mathcal{l}_2$ aggregation outperform perturbation-based and simplification-based approaches 
- HA is highest for gradient-based methods applied to Transformer models, suggesting a correlation between model accuracy and alignment with human explanations. However, this agreement does not uniformly extend to simpler architectures, indicating that human plausibility alone is insufficient to characterize explanation quality
- Faithfulness analysis reveals that gradient-based explanations induce the steepest degradation in model performance when salient tokens are masked, especially for CNN models. This suggests that, despite their simplicity, convolutional architectures produce explanations that are tightly coupled to decision-critical features. Importantly, high faithfulness does not always coincide with high human agreement
- Confidence indication results favor perturbation-based methods such as Shapley sampling and LIME, which more accurately predict model confidence from saliency patterns. This outcome is expected, as these methods explicitly measure changes in output probabilities under input perturbations
- Rationale consistency analysis demonstrates that explanations generated by gradient-based and LIME-based methods correlate positively with similarities in internal activation patterns across models trained with different random seeds. This indicates that these methods are sensitive to genuine differences in model reasoning rather than surface-level artifacts. In contrast, some explanation techniques exhibit low or even negative correlation with activation similarity, suggesting weak coupling to internal model behavior
- Dataset consistency scores are generally moderate across all methods, reflecting the inherent variability of explanations across instances. Nevertheless, gradient-based methods again show the strongest monotonic relationship between activation similarity and explanation similarity, particularly for CNN and Transformer models

### Summary

The experimental results support the central claim of the study: explanation methods must be evaluated along multiple, independent dimensions, as no single diagnostic property captures explanation quality in isolation. Gradient-based approaches emerge as the most robust and faithful explanation techniques across tasks and architectures, while perturbation-based methods excel primarily in conveying confidence-related information at higher computational cost

From a broader perspective, the findings highlight the importance of grounding explainability evaluation in internal model behavior rather than relying solely on human plausibility or visual appeal. In particular this ties in with the Plausibility Vs Faithfulness discussion
