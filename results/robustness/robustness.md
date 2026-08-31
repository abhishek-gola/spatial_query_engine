# Threshold robustness: unenriched sample, 5 ScanNet++ scenes

The resolver has 47 hand-set numeric constants and, until the benchmark is annotated, no labelled examples they could have been fitted on. So the question is whether the headline sensitivity number is a property of the scenes or of the thresholds.

20 trials. Every one of 34 `RelationConfig` fields and 9 query-time module constants jittered by a log-uniform factor of up to ±30%, with ordering constraints repaired afterwards.

| | frame disagreement rate |
|---|---|
| **as configured** | **4.1%** |
| perturbed, median | 3.7% |
| perturbed, 10th–90th pct | 2.9% – 5.1% |
| perturbed, full range | 2.0% – 6.8% |

| relation type | median | min | max |
|---|---|---|---|
| ordinal | 7.4% | 7.4% | 7.4% |
| projective_frontal | 3.3% | 1.8% | 6.6% |
| projective_lateral | 3.9% | 0.4% | 7.0% |

Held fixed, because they act at scene-build time rather than query time and varying them means rebuilding every scene per trial:

* `sqe.perception.orientation.CLEARANCE_SCALE`
* `sqe.perception.orientation.AGAINST_TOL`
* `sqe.geom.room.FORWARD_MARGIN_AMBIGUOUS`
* `sqe.query.resolver.MAX_ANCHOR_CANDIDATES`

**What this does and does not establish.** It shows whether the sensitivity number is stable against the thresholds. It does not validate the thresholds: only annotated data can say whether the resolver's answers are the ones a person meant. A stable number here plus an unvalidated policy is still an unvalidated policy -- it just means the *size of the frame problem* does not depend on my particular choices.