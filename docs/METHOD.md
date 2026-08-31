# Method

## The problem

"The mug to the left of the laptop" has no answer until you say whose left.

There are at least five defensible readings of that sentence, and in a cluttered
room they routinely pick different mugs:

| frame | "left of X" means | "in front of X" means |
|---|---|---|
| **egocentric** (relative, viewer-centric) | to *my* left, seen from where I stand | between me and X |
| **egocentric_bearing** | further left *in my visual field* | nearer to me in range |
| **egocentric_image** | further left *in the picture*, camera roll included | nearer the camera |
| **intrinsic** (object-centric) | to *X's own* left, as if X were an agent | on the side X faces |
| **addressee** | to the left as seen by someone facing X | on the side X faces |
| **world** (allocentric, room-canonical) | towards the room's left | towards the room's front |

Open-vocabulary 3-D segmentation is a crowded field and this repo treats it as a
component. The gap is downstream of it: pipelines that ground spatial language
pick one of these readings, usually by accident, usually never mentioning which,
and are then wrong whenever the speaker meant a different one. Because the
failure looks like a wrong object, it gets counted as a perception error and
gets attacked with a bigger detector.

This project makes the frame an explicit, first-class, *measured* part of the
pipeline.

## Three claims, and how each is tested

1. **The frames genuinely disagree, often.** Measured directly: for every
   frame-dependent query, resolve it under every plausible frame and count how
   often the answers differ. Reported as `frame_disagreement` in
   `results.json`.
2. **A single fixed convention is materially worse than a policy.** Measured by
   running the same benchmark with the frame forced to each fixed choice, which
   is what an implicit convention amounts to.
3. **Frame-convention errors are a large share of projective failures, and are
   not perception errors.** Measured by counterfactual attribution (below), and
   by contrasting frame-dependent against frame-free relations on the same
   scenes with the same instances.

## Conventions, stated

Full details in [CONVENTIONS.md](CONVENTIONS.md). The two that matter most:

**A frame's `front` axis points in the direction "in front of" means, and its
`right` axis in the direction "to the right of" means.** With that fixed:

* `intrinsic` and `world` satisfy `right = front × up` and are right-handed;
* `egocentric` has `front` pointing from the anchor back towards the viewer
  while `right` stays the viewer's own right, and comes out **left-handed**.

That handedness flip is not a bug. It *is* the mirror error — it is why the mug
on your left is on the chair's right, and why a pipeline that builds one basis
and reuses it for both axes is wrong for one of them whichever basis it picks.
`ReferenceFrame.handedness` reports it and `tests/test_frames.py` asserts it.

**The allocentric frame has a 4-fold ambiguity that most work inherits
silently.** Manhattan axes fix the room's grid but not which way is "forward",
so "the left side of the room" is undefined until something breaks the tie. Four
named conventions are implemented (`composite`, `principal_wall`, `trajectory`,
`longest_axis`) plus `dataset_axes`, the "just use +Y" baseline. Each returns a
**margin**; below `FORWARD_MARGIN_AMBIGUOUS = 0.12` the room has no canonical
front and queries leaning on it are flagged rather than answered.

On the five ScanNet++ scenes here the margin is small in every case — the office
`0b031f3119` scores **0.024** — which is itself a finding: the room-canonical
frame is usually underdetermined in real rooms, and any system that reports
confident room-relative directions is reporting a coin flip.

## The frame-selection policy

The default is a small table in `sqe/frames/policy.py`. Its most important
property is an asymmetry between the two projective axes:

| relation | preferred frame | why |
|---|---|---|
| `front` / `behind` | **intrinsic**, when the anchor has an estimable front | "in front of the sofa" means the side the sofa faces; nobody means "between me and the sofa" |
| `left` / `right` | **egocentric**, even when the anchor has a front | "the mug to the left of the laptop" is overwhelmingly read as the speaker's left |
| any, anchor is room shell | **world** | "the left wall" is room-relative |

Anchoring both axes to the same frame — which is what one reused basis gives you
— is wrong for one of them. Explicit linguistic markers override the table
entirely, and `FrameDecision.explicit` records whether the sentence stated a
frame or the policy supplied one. The benchmark reports those two populations
separately; the second is the interesting number.

The policy also *down-weights* rather than trusts: when an anchor's front
confidence is below 0.25 the intrinsic reading's prior is cut, and when a frame
cannot be built at all it is reported unavailable with a reason instead of
falling back silently.

## Intrinsic front estimation

"In front of the sofa" needs the sofa's facing, which no dataset ships for most
classes. `sqe/perception/orientation.py` estimates it in two explicitly separate
stages:

1. **which horizontal axis** the front lies on — mostly shape priors;
2. **which of the two directions** along it — placement and structure.

Cues are gated per category by `sqe/categories.py`: `backrest` (an upright mass
offset to one side — chairs, sofas, beds, and open laptops, which are the same
geometry at a different scale), `away_from_wall`, `thin_face`, `long_side`,
`open_face`.

Two decisions worth defending:

* **Camera visibility is deliberately excluded.** Which side the camera looked at
  is the strongest single signal for televisions and wall units. Using it would
  make the "intrinsic" frame partly egocentric and quietly contaminate the
  egocentric-versus-intrinsic comparison this repo exists to measure. It is
  implemented, off by default, and only ever reported as a diagnostic.
* **Abstention is a valid output.** The estimator returns no front when the
  category has none (a mug), when the axis is undetermined, or when the axis is
  known but its two ends are indistinguishable (a keyboard — reported as
  `axis_only`). Those are different situations and the resolver treats them
  differently. On the ScanNet++ office it produces a front for 18 of 43
  front-bearing objects and declines on the rest; of those it does produce,
  83–86% agree on axis with the annotations' own `dominantNormal`, which is an
  independent check rather than the thing being fitted.

The two-stage split was not the first design. Scoring all four directions on one
scale picked the wrong *axis* for real ScanNet++ monitors, because a monitor with
its stand is 0.56 × 0.20 m at the panel but 0.32 m deep as an instance, so the
"thin slab" test never fired and clearance decided the axis instead.

## Ordinals

"The second mug from the left on the middle shelf" needs three things, and
pipelines usually take two of them from one place:

1. an **ordering axis** — taken from the geometry the objects sit on (the
   shelf's long horizontal axis), which keeps the ordering stable when the
   viewer stands off to one side;
2. a **sign** — which cannot come from the shelf, since its long axis has no
   preferred end, and so comes from the reference frame;
3. an **index**.

Because the axis and the sign come from different places, the ordering is
identical across frames while the *answer* differs. The resolver also refuses in
two situations that pipelines answer anyway:

* **degenerate spread** — the candidates are not separated along the counting
  axis at all (four mugs along a shelf have no order along the room's
  front-back axis);
* **fragile ordering** — two candidates are closer along the axis than the
  noise, reported with the tie flagged.

"The middle shelf" of a unit with an even number of shelves has two equally good
referents, and `middle_level` returns both rather than rounding.

## Ambiguity as an output

A resolver that always returns one object is easier to score and less useful.
`sqe/query/ambiguity.py` distinguishes: `frame` (the plausible frames disagree),
`frame_unavailable`, `world_undetermined`, `ordinal_degenerate`, `ordinal_tie`,
`level_even`, `score_tie`, `anchor`, `weak_match`, `no_candidate`.

The benchmark labels every item for whether it *should* be ambiguous, and
ambiguity detection is scored as a binary classifier alongside accuracy. Roughly
a fifth to a third of naturally-phrased spatial queries land here.

One scoring rule follows from this and is worth stating, because getting it
wrong penalises the behaviour we want: for an item marked ambiguous **with no
acceptable answer** ("the plant in the corner nearest the window", where there
is no corner object), the correct behaviour is to say the query cannot be
answered, so the item is scored on the ambiguity flag alone rather than on the
object returned.

## Benchmark design

300–500 hand-annotated queries over ScanNet++ scenes. Each item records the gold
target(s), the **frame the annotator judged the sentence to mean**, whether the
frame was stated in the text, whether the query is ambiguous and of what kind, a
gold structured parse, the viewpoint, and optionally the answer under each
frame.

**Proposal, not pre-filling.** `sqe/bench/generate.py` proposes candidate
queries from scene geometry, weighted towards frame-disagreement cases and
balanced across relation families, with frame-free controls on the same scenes.
It writes `target_ids: []` and `frame: "unspecified"`. The annotation tool is
**blind by default** and does not show what the resolver would answer, because a
human confirming a system's prediction produces a benchmark that measures
agreement rather than correctness. `--show-prediction` exists for debugging and
stamps every item it touches so the report can separate them.

**Anchors are chosen for salience**, not alphabetically. A benchmark full of
"the box to the left of the blind rail" measures nothing anyone cares about, and
alphabetical iteration over a ScanNet++ label set produces exactly that.

**Thresholds are not tuned on the benchmark.** Every physical threshold lives in
`configs/relations.yaml` with a comment saying what it means, set from the
meaning of the words and the geometry of ordinary rooms. Fitting them to the
benchmark would make the reported accuracy meaningless.

## Counterfactual failure attribution

Each failure of the primary condition is attributed to the **first** cause below
that repairs it. The order is fixed and printed in every report so the numbers
cannot be reshuffled afterwards:

1. `unresolvable` — no candidate at all, usually a missing anchor
2. `parse` — repaired by the gold parse
3. `perception` — repaired by ground-truth instances
4. `frame_unavailable` — the annotated frame could not be constructed
5. `frame_convention` — repaired by forcing the annotated frame
6. `geometry` — repaired by nothing; the relation scoring ranked wrong
7. `ambiguous_item` — the item has no single gold answer

Putting `parse` and `perception` **before** `frame_convention` is the
conservative choice: a failure is blamed on the frame only when nothing upstream
explains it, so the headline frame number is a lower bound rather than a
flattering one.

The report also carries the comparison that makes the inference safe: accuracy on
frame-dependent relations against accuracy on frame-free relations
(`on`/`above`/`between`/`near`/comparatives) over the same scenes with the same
instances. If frame-free relations score well and projective ones do not,
perception is not the explanation.

## Verification, and what it caught

Two checks run outside the benchmark, because a benchmark cannot catch a
systematic geometry error -- it just reports lower accuracy.

**Box overlays on real frames** (`sqe render`). The 3-D boxes are projected into
the capture's own RGB frames using the dataset's poses and intrinsics. If the
mesh alignment, pose convention, intrinsic scaling or box fit is wrong, the boxes
land in the wrong place and it is obvious. It found three real bugs:

* `best_view` clipped angular size at 1.0, so it preferred frames pressed
  against an object. This is the default egocentric viewpoint for queries, and a
  viewpoint inside the anchor makes its left and right meaningless.
* With joint anchor scoring, each candidate anchor resolved its own viewpoint,
  and the *reported* frame was not the frame that scored the winner. The
  viewpoint is a property of the query, so it is now resolved once per constraint
  and shared across anchor candidates.
* Projective relations had no locality presupposition, so "the monitor to the
  left of the keyboard" was satisfied by a monitor 3.2 m away on another desk.
  A low-weight term inside a geometric mean cannot veto; locality is now a
  multiplicative gate scaled by the anchor's size.

**Ground-truth audit** (`sqe audit`). ScanNet++ instance 114 of scene
`0b031f3119` is labelled `office chair` and is the desk partition. Two
independent signals catch it: the box fitted to the instance's own points
disagreeing with the annotated box (median disagreement across the scene is
0.3 micrometres, so 23 cm is conspicuous), and a size implausible for the label.
Across the five scenes 15 of 495 instances (3.0%) are flagged; they stay in the
scene but are excluded from proposal generation, so a mislabelled instance cannot
produce an unanswerable query whose failure is then attributed to geometry.

The size heuristic is applied only to categories whose height is genuinely
constrained (`height_stable` in `sqe/categories.py`). Applied to everything it
flagged good annotations: a floor lamp really is 1.4 m, a closed laptop really is
4 cm, and "shelf" spans a shoe rack and a full bookcase.

## Known limitations

* **Front estimation abstains often on real data** — 18 of 43 front-bearing
  objects in the ScanNet++ office. Those queries are reported as
  `frame_unavailable` rather than guessed. It is honest, and it is also a
  ceiling on intrinsic-frame accuracy.
* **The world frame is nearly undetermined in all five scenes.** Conclusions
  about allocentric readings rest on very little.
* **The rule parser is not a semantic parser.** It handles the phrasings in the
  benchmark and a good deal beyond, but the LLM parser exists as a separate
  condition for a reason, and the gold-parse condition is what isolates spatial
  reasoning from language.
* **Colour attributes use mean vertex colour**, which is blunt. It narrows a
  field of four mugs; it does not decide a query.
* **Five scenes is not a lot.** The per-scene accuracy table is in the report
  precisely so that scene-level variance is visible rather than averaged away.
* **The open-vocabulary backend is a component, not a contribution**, and its
  quality is modest: geometric proposals plus multi-view CLIP reach recall@0.25
  of 0.63 and mean best IoU 0.34 against the ground-truth instances, with heavy
  over-segmentation of large objects. Those numbers are measured and reported
  rather than assumed, which is what makes the `perception` attribution row
  interpretable. A learned proposal network is the obvious replacement and plugs
  in at `scene_from_masks`.
