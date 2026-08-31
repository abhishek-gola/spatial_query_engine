# Frame instructability — controls only

Minimal pairs: two sentences differing only in an explicit marker of whose left is meant, on scenes where the two readings pick different objects. **frame_blind** means the system gave the same answer to both -- it has no frame to instruct.

| system | pairs | switched correctly | frame blind | switched wrongly | partial | no answer | control stable |
|---|---|---|---|---|---|---|---|
| resolver (cue-following) | 35 | 100.0% | **  0.0%** |   0.0% |   0.0% |   0.0% | 100.0% |
| pinned:egocentric | 35 |   0.0% | **100.0%** |   0.0% |   0.0% |   0.0% | 100.0% |
| pinned:intrinsic | 35 |   0.0% | **100.0%** |   0.0% |   0.0% |   0.0% | 100.0% |

**`control stable` is load-bearing.** It is the fraction of *non-contrastive* paraphrase pairs -- equally awkward, matched for shape, differing in nothing that should change the answer -- that the system answers identically. A high `frame_blind` rate only means "has no frame to instruct" if `control stable` is also high. A system with low `control stable` is unstable to surface form, and its frame-blindness is unattributable.

## Default convention on the uncued sentence

With no marker, which arm's answer does the system give?

| system | egocentric | intrinsic |
|---|---|---|
| resolver (cue-following) | 18 | 17 |
| pinned:egocentric | 35 | 0 |
| pinned:intrinsic | 0 | 35 |

## Reading the controls

* **`resolver (cue-following)`** is a circularity check, not a result. The pairs were built by forcing this resolver's own frames, so it must score near 100% switched-correctly; that only confirms the stimulus is well-formed and the scoring works. It is not evidence that the resolver is right about anything.
* **`pinned:<frame>`** is the positive control for the `frame_blind` label: a system that provably cannot switch must score 100% frame-blind. If it does not, the metric is broken and no other row means anything.