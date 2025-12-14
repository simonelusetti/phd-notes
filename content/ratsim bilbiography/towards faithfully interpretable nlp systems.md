[https://arxiv.org/pdf/2004.03685] (ACL)

Question: *before asking if a model is interpretable or an explanation is interpretable, we need to more rigorously define these terms, avoiding confusion with other, similar but not directly related, definitions*
# Plausibility and Faithfulness

They mention two important properties of explanations: 
- **Plausibility:** how convincing an explanation is to humans, AKA how well this maps human concepts to the model
- **Faithfulness:** how "accurate" the explanation is, how accurately it describes the actual model behaviors in how it used the data
These are two independent properties

**Inherently interpretable?:** they mention that explanation of models is done mainly in two ways: *post-hoc analysis of a model's behavior* (the black box techniques) and *inherently interpretable models*. They mention that a model being "inherently interpretable" is a big claim that must be confirmed carefully

## Utility Based Evaluation

They mention that the literature uses Intelligent User Interfaces (IUI), via Human-Computer Interaction (HCI) to evaluate an explanation. In these contexts a model aids a human to solve a task, giving an explanation for its prediction, with the assumption that a better "explained" model would increase the model-user trust and increase the performances of the latter

However the authors state this is more reflective of Plausibility than Faithfulness and could be misleading: 

*Consider a scenario where the explanation is an attention heatmap of the input tokens. When the model is correct the heatmap is lit around random character tokens, while when incorrect it shows attention going to random punctuation marks*

While the user, convinced by the nicer looking (plausible) explanation would increase their performances, the faithfulness of these explanations is quite low
# Guidelines for Explainability

They outline some general ideas that should guide testing for explainability
- Be mindful and explicit if you are measuring Faithfulness or Plausibility
- Human judgment shouldn't be included in any measure of Faithfulness, human judgement always measures Plausibility 
- Faithfulness should never involve human generated gold labels, for the same reason above
- No trust of "inherently explainable", each model should be held to the same standards as post-hoc methods
- Faithfulness of HCI shouldn't be measured by the user's performances, while useful it's more indicative of Plausibility
# Rigorously defining Faithfulness

They mention how basically each paper has its own definition of Faithfulness or explainability and that makes comparisons hard, so they outline what they believe to be three common assumptions to all of them:

- **Assumption 1 (the model assumption)**: two models will make the same prediction $\iff$ they use the same reasoning process
	- **Corollary 1.1:** An interpretation system is un-faithful if it results in different interpretations of models that make the same decisions

The corollary is particularly useful for counter-proof: if two models make the same prediction (so by assumption are using the same reasoning), and an interpretation method shows two different interpretation, it's automatically not faithful

- **Corollary 1.2 (Fidelity):** an explanation method is unfaithful if it makes different predictions than the actual model

Some explanation methods are models themselves, like decisions tree. Even when a model can't really make a prediction we can simulate that by giving a bunch of humans the input $x$, the explanation $e(x)$ and is if given these the human prediction $y_h$ is the same as the model's $y_m$. A model that (in either way) makes different predictions is automatically unfaithful

- **Assumption 2 (interpretation robustness):** on similar inputs the model gives similar outputs $\iff$ the model's reasoning is the same
	- **Corollary 2.1:** an interpretation system is unfaithful if it gives different interpretations for similar outputs

Again, this is useful for counter-proofs: by showing that a system gives very different interpretations of similar inputs and outputs we can rule it as unfaithful. This is particularly hard in NLP given the discrete inputs

- **Assumption 3 (linearity):** Certain parts of the input are more important to the model reasoning than others. Moreover, the contributions of different parts of the input are in-dependent from each other
	- **Corollary 3.1:** under certain conditions, heatmaps can be faithful

This is an important idea because we can start with an heatmap and "stress test" to see if it fails. For example one employed method is **erasure**, where a supposedly important part of the input is removed to see if the model's decision changes. Viceversa, the erasing a non important part shouldn't change the decision
# A better definition of Faithfulness

The authors claim that by imagining faithfulness as a binary we are restricting ourselves. They mention the vast amount of work that proves a system is NOT faithful. However, since an explanation is always a proxy of the model's decisions, and these decisions may not fully map to human concepts, this approximation might result in **every** system being unfaithful

They ask the community to solve this challenge: *can we find a definition of faithfulness that tells us when an explanation is faithful enough?*
- **Across tasks and models:** one idea is to stop looking for universality. A system may be unfaithful in Question-Answering tasks, but be very faithful in sentiment analysis
- **Across input space:** similarly, trying to find a system that explains all inputs may be intractable, and a system's faithfulness on a subset of the input shouldn't be discarded because it fails on other






