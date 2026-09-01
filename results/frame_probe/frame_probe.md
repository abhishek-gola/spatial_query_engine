# Frame instructability

Minimal pairs: two sentences differing only in an explicit marker of whose left is meant, on scenes where the two readings pick different objects. **frame_blind** means the system gave the same answer to both -- it has no frame to instruct.

| system | pairs | switched correctly | frame blind | switched wrongly | partial | no answer | control stable |
|---|---|---|---|---|---|---|---|
| resolver (cue-following) | 35 | 100.0% | **  0.0%** |   0.0% |   0.0% |   0.0% | 100.0% |
| pinned:egocentric | 35 |   0.0% | **100.0%** |   0.0% |   0.0% |   0.0% | 100.0% |
| pinned:intrinsic | 35 |   0.0% | **100.0%** |   0.0% |   0.0% |   0.0% | 100.0% |
| claude-opus-5 (self-administered) | 35 |  42.9% | ** 25.7%** |  11.4% |  20.0% |   0.0% |  80.0% |
| claude-haiku-4.5 (self-administered) | 35 |   5.7% | ** 48.6%** |  14.3% |  25.7% |   5.7% |  37.1% |

**`control stable` is load-bearing.** It is the fraction of *non-contrastive* paraphrase pairs -- equally awkward, matched for shape, differing in nothing that should change the answer -- that the system answers identically. A high `frame_blind` rate only means "has no frame to instruct" if `control stable` is also high. A system with low `control stable` is unstable to surface form, and its frame-blindness is unattributable.

## Conditioned on agreement about the *unmarked* sentence

`frame_blind` means "same answer to both cued arms". If a system already disagrees with this resolver about the plain, uncued sentence, its two identical answers can be a consistent reading of a sentence I resolve differently -- the disagreement is about the baseline, not about whether the cue landed. Pooling the two populations charges baseline disagreement to frame-blindness. Counts, not percentages: the subset is small.

| system | pairs kept | dropped | switched correctly | frame blind | switched wrongly | partial |
|---|---|---|---|---|---|---|
| resolver (cue-following) | 35 | 0 | 35 | **0** | 0 | 0 |
| pinned:egocentric | 18 | 17 | 0 | **18** | 0 | 0 |
| pinned:intrinsic | 17 | 18 | 0 | **17** | 0 | 0 |
| claude-opus-5 (self-administered) | 16 | 19 | 12 | **2** | 1 | 1 |
| claude-haiku-4.5 (self-administered) | 13 | 22 | 1 | **7** | 1 | 3 |

**This is the row to read.** A pooled `frame_blind` rate over all 35 pairs mixes in every item where the system and I simply read the plain sentence differently, and there is no ground truth yet saying which of us is right. On the conditioned subset the cue mostly does land, and the honest summary is that the pooled rate was not measuring what its name says.

## The same table, restricted to control-matched pairs

Each control is built from one pair, so every pair has an item-level check on whether that system's answers move for reasons unrelated to the frame. Restricting to the pairs that passed their own check is the reading of `frame_blind` that survives a noisy system: on these items the system answers the same sentence the same way twice.

| system | pairs kept | dropped | switched correctly | frame blind | switched wrongly | partial |
|---|---|---|---|---|---|---|
| resolver (cue-following) | 35 | 0 | 100.0% | **  0.0%** |   0.0% |   0.0% |
| pinned:egocentric | 35 | 0 |   0.0% | **100.0%** |   0.0% |   0.0% |
| pinned:intrinsic | 35 | 0 |   0.0% | **100.0%** |   0.0% |   0.0% |
| claude-opus-5 (self-administered) | 28 | 7 |  42.9% | ** 28.6%** |  10.7% |  17.9% |
| claude-haiku-4.5 (self-administered) | 13 | 22 |   7.7% | ** 61.5%** |   0.0% |  23.1% |

## Default convention on the uncued sentence

With no marker, which arm's answer does the system give?

| system | egocentric | intrinsic | neither |
|---|---|---|---|
| resolver (cue-following) | 18 | 17 | 0 |
| pinned:egocentric | 35 | 0 | 0 |
| pinned:intrinsic | 0 | 35 | 0 |
| claude-opus-5 (self-administered) | 22 | 4 | 9 |
| claude-haiku-4.5 (self-administered) | 11 | 10 | 14 |

## Reading the controls

* **`resolver (cue-following)`** is a circularity check, not a result. The pairs were built by forcing this resolver's own frames, so it must score near 100% switched-correctly; that only confirms the stimulus is well-formed and the scoring works. It is not evidence that the resolver is right about anything.
* **`pinned:<frame>`** is the positive control for the `frame_blind` label: a system that provably cannot switch must score 100% frame-blind. If it does not, the metric is broken and no other row means anything.