*Given only observations $x$, can we recover latent factors $z$ such that each dimension of $z$ corresponds to a distinct, interpretable generative factor?*

>For $d > 1$, let $z \sim P$ denote any distribution which admits a density $p(z) = \Pi_i^d p(z_i)$. Then, there exists an infinite family of bijective functions $f : \text{supp}(z) \to \text{supp}(z)$ such that $\partial f_i(u)/\partial u_j \neq 0$ almost everywhere for all $i$ and $j$ (i.e., $z$ and $f(z)$ are completely entangled) and $P (z \leq u) = P (f (z) \leq u)$ for all $u \in \text{supp}(z)$ (i.e., they have the same marginal distribution)

What is this saying? Let's imagine the real world has perfectly factorizable factors $p(s) = \Pi_i p(s_i)$ associated with a generative process (which we don't know) $g(s) = x$, with $x$ being the observed data. We would like to reconstruct these factors, aka managing to get $z = s$

*Sketch of the Proof:* we can easily build an invertible function $T$ such that $z' = T(z)$, where $z'$ is the entangled version of $z$. However since $T$ is invertible we can easily show that a generative model $g' = g \circ T^{-1}$ exists and it's the one we would learn. Since we can't know "which" model we have don't know if we found $z$ or $z'$
## Breaking the Symmetry

Locatello's shows us that we must induce some kind of bias to correctly disentangle our latent factors, below are some ways to do so

**Note:** No method *grantee* us that there will be no entanglement, they just render it unlikely by reducing the number of valid transformations the latent representation can take
#### Markovian Structure
Let's model our sequence of tokens as a Markov Chain, implying a dependence of the current factor on the one before it. Let's impose 
$$
p(e_{1:T})=p(e_1)\prod_{t}p(e_t\mid e_{t-1})
$$
then the only reparameterizations $T$ of the latent space that preserve this structure for _all_ sequences are those for which $e_t' = T(e_t)$ induces the **same transition matrix** $p(e_{t+1}'\mid e_t')=p(e_{t+1}\mid e_t)$ and therefore $T$ must preserve the equivalence classes of the Markov states




