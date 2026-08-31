# Spatial query benchmark

25 queries over 2 scenes.  8 (32.0%) marked ambiguous by the annotator.  14 are frame-dependent.

## Composition

| relation type | n |
|---|---|
| vertical | 7 |
| ordinal | 6 |
| projective_lateral | 6 |
| projective_frontal | 2 |
| proximity | 2 |
| between | 1 |
| containment | 1 |

| annotated frame | n |
|---|---|
| any | 11 |
| egocentric | 8 |
| intrinsic | 5 |
| world | 1 |

## Accuracy by condition

| condition | n | overall | frame-dependent | frame-free | frame stated | frame unstated |
|---|---|---|---|---|---|---|
| ours (policy frame) | 25 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| fixed frame: egocentric | 25 |  96.0% |  92.9% | 100.0% |  66.7% | 100.0% |
| fixed frame: intrinsic | 25 |  84.0% |  71.4% | 100.0% |  66.7% |  72.7% |
| fixed frame: world | 25 |  96.0% |  92.9% | 100.0% |  66.7% | 100.0% |
| oracle frame | 25 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| gold parse | 25 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

## Accuracy by relation type

| relation type | n | ours (policy frame) | fixed frame: egocentric | fixed frame: intrinsic | fixed frame: world | oracle frame | gold parse |
|---|---|---|---|---|---|---|---|
| between | 1 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| containment | 1 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| ordinal | 6 | 100.0% |  83.3% |  33.3% |  83.3% | 100.0% | 100.0% |
| projective_frontal | 2 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| projective_lateral | 6 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| proximity | 2 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| vertical | 7 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

## Accuracy by annotated frame

| annotated frame | n | ours (policy frame) | fixed frame: egocentric | fixed frame: intrinsic | fixed frame: world | oracle frame | gold parse |
|---|---|---|---|---|---|---|---|
| any | 11 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| egocentric | 8 | 100.0% | 100.0% |  50.0% | 100.0% | 100.0% | 100.0% |
| intrinsic | 5 | 100.0% |  80.0% | 100.0% |  80.0% | 100.0% | 100.0% |
| world | 1 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

## Ambiguity detection

Scored as a binary classifier against the annotator's `ambiguous` flag, on the primary condition.

| precision | recall | F1 | tp | fp | fn | tn |
|---|---|---|---|---|---|---|
|  77.8% |  87.5% |  82.4% | 7 | 2 | 1 | 15 |

On 14 frame-dependent queries, the plausible reference frames picked different objects in 3 cases (21.4%).

## Failure attribution

Each failure of the primary condition is attributed to the first cause that repairs it, in this fixed order: unresolvable -> parse -> perception -> frame_unavailable -> frame_convention -> geometry -> ambiguous_item.

| cause | n | share of failures |
|---|---|---|
| unresolvable | 0 |   n/a |
| parse | 0 |   n/a |
| perception | 0 |   n/a |
| frame_unavailable | 0 |   n/a |
| frame_convention | 0 |   n/a |
| geometry | 0 |   n/a |
| ambiguous_item | 0 |   n/a |
