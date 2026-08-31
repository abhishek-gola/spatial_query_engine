# Frame sensitivity: 882 proposed queries, 5 ScanNet++ scenes, ground-truth perception

Measured without annotation. **This is not accuracy.** It is how much the answer depends on which reference frame is used, which is the precondition for any accuracy claim: if the frame never changed the answer, the frame would not matter.

882 queries, 512 of them frame-dependent (370 frame-free).

## Headline: frames disagree with each other

### 18.8% of frame-dependent queries (96 of 512)

Two plausible reference frames, both constructible and both returning a confident answer, select **different objects**. This is the statistic to quote: it says nothing about which frame is right, so it does not depend on my policy being correct. It is a lower bound -- a frame that could not be built at all, because an anchor's front was not estimable, counts as agreement here rather than as disagreement.

| relation type | n | frames disagree |
|---|---|---|
| ordinal | 54 |   7.4% |
| projective_frontal | 228 |  15.4% |
| projective_lateral | 230 |  24.8% |

## Secondary, and weaker: flip rate against a fixed convention

How often forcing a single fixed frame -- what a pipeline that never names its convention effectively has -- picks a different object from the policy.

**This is not an error rate.** It measures divergence from *my* policy's answer, and the policy has not been validated against human labels yet. If the policy is wrong, a high flip rate means only "disagrees with my choice". Treat it as an upper bound on how much the frame choice could matter, not as a measurement of how much anyone gets wrong.

| forced frame | queries | answer changed | flip rate | no answer under this frame |
|---|---|---|---|---|
| egocentric | 512 | 198 |  38.7% | 158 |
| intrinsic | 512 | 301 |  58.8% | 185 |
| addressee | 512 | 273 |  53.3% | 181 |
| world | 512 | 300 |  58.6% | 118 |

The last column matters as much as the flip rate: those queries have no answer at all under that frame, usually because the anchor's front could not be estimated. They are counted as unchanged above, which is another reason the flip rate is not an error rate.

## Frames the policy chose

* egocentric: 354
* intrinsic: 153
* world: 1

## Ambiguity flags raised, by kind

653 of 882 queries were flagged. These are the system's own flags, not annotator judgements. Note the composition: `anchor` and `score_tie` dominate because a real room contains several instances of most classes, and they are not what this project claims. The benchmark scores each kind separately for exactly this reason.

| kind | n flagged |
|---|---|
| anchor | 401 |
| score_tie | 297 |
| weak_match | 114 |
| frame | 96 |
| ordinal_tie | 28 |
| ordinal_degenerate | 10 |
| level_even | 5 |