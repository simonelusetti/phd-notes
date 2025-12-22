[[https://aclanthology.org/W18-6010.pdf]] (UDW)

**Note:** This paper isn't directly useful for us, but their dataset is. One problem we are finding is that there's only one good chunking dataset, conll2000, which is constructed from TreeBank. However, we can do something similar to how they generated conll2000 from Unviversal Dependencies. There's already some literature on how to do that which is exactly this paper but with a limitation: *As a first target, we decide to restrict the task to the most common chunks: noun-phrases (NP)*
# Rules

*We deduce minimal NP-chunks, which means that embedded prepositional (PP) chunks are not included in our NP-chunks*
This means that they produce something like this `[NP screenshots][NP two beheading videos]` instead of `[NP screenshots of two beheading videos]` (notice the lack of "of" in the first)


Step 1: *We first identify the core tokens of NPs: the nouns (NOUN), proper nouns (PROPN) and some pronouns (PRON)*

Step 2: Once a core token is found, they attach children of that token $\iff$ the incoming dependency relation follows the following rules:

| Dependency relation       | Included? | Conditions / Notes                                             |
| ------------------------- | --------- | -------------------------------------------------------------- |
| `compound`                | ✅         | Always                                                         |
| `compound:prt`            | ✅         | Always                                                         |
| `flat`                    | ✅         | Always                                                         |
| `flat:name`               | ✅         | Always                                                         |
| `goeswith`                | ✅         | Always                                                         |
| `fixed`                   | ✅         | Always                                                         |
| `nummod`                  | ✅         | Always                                                         |
| `det`                     | ✅         | Only if the determiner **precedes** its head                   |
| `amod`                    | ✅         | Only if the child is **not an adverb**                         |
| `conj`                    | ✅         | **Only if both child and head are adjectives**                 |
| `appos`                   | ✅         | Only if child is **immediately before or after** the head      |
| `advmod`                  | ✅         | Only if: child ≠ `PART` or `VERB` **and** head is an adjective |
| `nmod:poss`               | ✅         | Only if child is **not** `NOUN` or `PROPN`                     |
| `obl:npmod`               | ✅         | Preceding or following                                         |
| `obl:tmod`                | ✅         | Preceding or following                                         |
| `obl`                     | ✅         | Only if the **head has an incoming `amod`**                    |
| `case`                    | ❌         | Would create PP (PPs explicitly excluded)                      |
| `nmod` (general)          | ❌         | Would embed PP                                                 |
| `acl`, `acl:relcl`        | ❌         | Clause boundary                                                |
| `ccomp`, `xcomp`, `advcl` | ❌         | Clause boundary                                                |
| `cc`                      | ❌         | Coordination marker                                            |
| `conj` (non-adjective)    | ❌         | Avoids merging coordinated NPs                                 |
| `mark`                    | ❌         | Subordinate clause                                             |
| `punct`                   | ❌         | Not structural                                                 |

Step 3: Fixxing chunks
- *When grouping a core token with one of its `det, compound, nummod or nmod:poss` children, we automatically attach tokens which are in between*
	- Example: "my very best friend", with "my $\to$ friend". They attach "very best" to the same chunk as the other words since they are "in between"
- *if split chunks remain, we attach the non-attached tokens which are in between two part of a chunk* $\to$ AKA, if all else fails and a chunk is still discontinuous, fuse it with the "in between" words






