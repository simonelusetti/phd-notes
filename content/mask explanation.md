Basics of attentions: $X$ = tokens, $W_k, W_q, W_v$ keys, queries and values matrices

$$K = W_k X\quad Q = W_q X\quad V = W_v X$$

$$S = (Q K^T) / \sqrt{d}\quad  S \in R^{T x T}$$

$$A = \text{softmax}_\text{rows}(S) \implies A_m = A · \text{diag}(m)$$

$$D = \text{diag}(A_m · 1) \implies A_r = D^{-1} · A_m \quad \text{where 1 is a T long, column vector of 1s}$$

Note: the last bit is equivalent to row normalization

Let's check token $i$ in the output: 
- $O_i = ( \sum_k A[i,k] * m[k] * V_k ) / ( \text{norm denom})$
- So the influence of token $j$ on $i$ is scaled by $m[j]$, meaning if it's 0 token $j$ will be "suppressed"
- Note that token $j$ will still receive attention update, but if we don't use it we don't care


