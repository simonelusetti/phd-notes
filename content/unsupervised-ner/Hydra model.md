We progressively build a tree of ratcon+moe models, level by level

Starting from sentence $s$, the selector model (ratcon) filters it into $\bar{s}$, then the factor model (moe) divides it into various experts $s_1, s_2, ..., s_k$. This is the first level

Then freeze the current level, and increase the depth by one, so each $s_i$ is filtered into $\bar{s}_i$, then divided into $s_{i1}, s_{i2}, ..., s_{ik}$

The maximum level is defined as a hyperparam $L$, calculating the f1, precision and recall of each leaf at each level

**Planned improvements**:
- Combine various leaves into an ensamble model (a sort of giant OR gate)
- Better division of factors (theory needed)