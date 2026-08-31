# Frame sensitivity: 5 ScanNet++ scenes, ground-truth perception (unenriched sample - population rate)

> Item set sampled **without** frame-sensitivity enrichment, so rates here are population rates over the generator's candidate space.

Measured without annotation. **This is not accuracy.** It is how much the answer depends on which reference frame is used, which is the precondition for any accuracy claim: if the frame never changed the answer, the frame would not matter.

882 queries, 512 of them frame-dependent (370 frame-free).

## Headline: frames disagree with each other

### 4.1% of frame-dependent queries (21 of 512)

Two plausible reference frames, both constructible and both returning a confident answer, select **different objects**. This is the statistic to quote: it says nothing about which frame is right, so it does not depend on my policy being correct. It is a lower bound -- a frame that could not be built at all, because an anchor's front was not estimable, counts as agreement here rather than as disagreement.

| relation type | n | frames disagree |
|---|---|---|
| ordinal | 54 |   7.4% |
| projective_frontal | 228 |   2.6% |
| projective_lateral | 230 |   4.8% |

## Secondary, and weaker: flip rate against a fixed convention

How often forcing a single fixed frame -- what a pipeline that never names its convention effectively has -- picks a different object from the policy.

**This is not an error rate.** It measures divergence from *my* policy's answer, and the policy has not been validated against human labels yet. If the policy is wrong, a high flip rate means only "disagrees with my choice". Treat it as an upper bound on how much the frame choice could matter, not as a measurement of how much anyone gets wrong.

| forced frame | queries | no answer under it | flip rate (counting no-answer as a change) | flip rate (answered only) |
|---|---|---|---|---|
| egocentric | 512 | 233 |  47.9% |   4.3% |
| intrinsic | 512 | 324 |  75.8% |  34.0% |
| addressee | 512 | 324 |  74.4% |  30.3% |
| world | 512 | 196 |  59.4% |  34.2% |

Read the last column. The middle one counts a query with **no answer at all** under the forced frame as a change, so a frame that simply cannot be constructed -- usually because the anchor's front is not estimable -- inflates it. That is a third reason the flip rate is the weaker statistic.

## Frames the policy chose

* egocentric: 387
* intrinsic: 120
* world: 1

## Ambiguity flags raised, by kind

640 of 882 queries were flagged. These are the system's own flags, not annotator judgements. Note the composition: `anchor` and `score_tie` dominate because a real room contains several instances of most classes, and they are not what this project claims. The benchmark scores each kind separately for exactly this reason.

| kind | n flagged |
|---|---|
| anchor | 354 |
| score_tie | 272 |
| weak_match | 236 |
| ordinal_tie | 28 |
| frame | 21 |
| ordinal_degenerate | 10 |
| level_even | 5 |