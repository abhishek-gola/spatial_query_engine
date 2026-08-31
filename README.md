# Spatial query engine — reference-frame-aware relation resolution

You walk through a room filming with a phone. The system builds a 3-D map that
knows what each object is and where it sits relative to everything else, so you
can type *"the second mug from the left on the middle shelf"* and it points at
exactly that one.

The part that is actually hard is the phrase **"from the left"**.

> *"Left of the laptop"* — left from whose view? The camera's, the laptop's own
> front, or the room's canonical axes? Pipelines that ground spatial language
> compute this from object coordinates without committing to, or reporting, a
> frame. Because the resulting failure looks like a wrong object, it gets counted
> as a perception error and attacked with a bigger detector.

That is a narrow claim and it is checkable, so here is the check. **SpatialRGPT**
defines its relation set as "left, right, above, below, behind, front …" and
computes it by traversing object nodes and using "the point cloud centroids and
bounding boxes" — no frame named. **SpatialVLM** asks "which is more towards the
left?" and, in one template, "how far is A positioned behind B *relative to the
camera*" — so the frame is the camera's, stated inside a template string rather
than as a modelling commitment. Literal quotes, page numbers and the rest of the
positioning are in **[docs/RELATED_WORK.md](docs/RELATED_WORK.md)**.

The frame trichotomy itself is not new — it is standard in linguistics, and
Levinson (2003) devotes a chapter to it. **The contribution here is making it
operational in a 3-D pipeline and measuring it, not discovering it.**

## How often does the frame actually change the answer?

On **882 queries over 5 ScanNet++ scenes with ground-truth perception**:

### Two plausible frames pick different objects on 12–20% of frame-dependent queries

18.8% as configured; median 16.1% and never below 11.3% across 20 trials that
jitter all 43 query-time constants by ±30%
([results/robustness/](results/robustness/robustness.md)).
Highest for lateral relations (24.8%), lower for frontal (15.4%).

This is the number to quote, for two reasons. It makes no reference to *which*
frame is correct, so it does not depend on my policy being right. And it is a
lower bound: a frame that could not be constructed at all — because an anchor's
front was not estimable — counts as agreement here, not as disagreement.

A second, **weaker** statistic: forcing one fixed convention changes the answer
on 38.7–58.8% of frame-dependent queries depending on the convention. That is
*not* an error rate — it measures divergence from my policy's answer, and the
policy has not yet been validated against human labels. Full table and caveats
in [results/sensitivity/](results/sensitivity/sensitivity.md).

**No accuracy number is claimed yet.** Accuracy needs the hand-annotated frame
labels; 882 proposals are generated and waiting. See
[the benchmark section](#the-benchmark).

This repo makes the reference frame an explicit, first-class, **measured** part
of the pipeline: a frame-aware relation resolver, a hand-annotated benchmark
labelled by relation type *and* by whether each query is ambiguous at all, and a
failure attribution that separates frame-convention errors from perception
errors.

Open-vocabulary 3-D segmentation is treated as a replaceable component. It is
not the contribution.

---

## The one thing to look at

Same sentence, same scene, same geometry. Only the frame changes.

```
$ sqe query --scene 0b031f3119 "the second mug from the left on the middle shelf"

query: "the second mug from the left on the middle shelf"
  parsed as: second mug [on shelf <level: middle>]
  anchor 'shelf' -> #0 bookshelf (4.78, 2.00, 0.90) 0.40x1.60x1.80, level 1 of 3 at z=0.58
  frame: egocentric (policy default) | eye (2.00, 2.00, 1.55) | right (+0.00, -1.00) | conf 1.00
  ordering: support_long_axis along (-0.00, -1.00) spread 1.20 m over 4 candidates
  ANSWER: #3 mug (4.72, 2.00, 0.67)
  ambiguous (frame): the reference frames disagree:
      mug #2 under intrinsic; mug #3 under addressee/egocentric;
      answering with the egocentric reading
```

The ordering axis comes from the *shelf's* long side. The **sign** comes from the
frame — and the shelf's own right is the exact opposite of the viewer's right, so
"second from the left" is a different mug depending on who is speaking. The
resolver answers with the default reading and *says* what the alternative was.

In the viewer, clicking a row of the frame table re-resolves under that frame and
the highlighted object jumps.

## Why this is a contribution and not implementation work

Three things, none of which existing pipelines do:

1. **Five frames, built explicitly**, with the sign conventions worked out and
   tested — including the fact that the egocentric frame is *left-handed* under
   the natural reading of "in front of". That handedness flip is the mirror
   error.
2. **A documented selection policy** with an asymmetry that matters:
   `front`/`behind` default to the object's own frame, `left`/`right` to the
   viewer's. One reused basis is wrong for one of them whichever you pick.
3. **Ambiguity as an output.** When the plausible frames disagree, the honest
   answer is "here is the default reading, and here is what the other reading
   gives" — not a confident single object.

Full reasoning, conventions, and experimental design: **[docs/METHOD.md](docs/METHOD.md)**.

## Install

The core — geometry, frames, relations, resolver, benchmark, viewer — needs only
`numpy`, `scipy`, `plyfile` and `pyyaml`. That is deliberate: the part worth
reading should not be gated behind a CUDA install.

```bash
conda create -y -n sqe python=3.11 && conda activate sqe
pip install -e ".[iphone]"      # + lz4, for ScanNet++ iPhone depth
# pip install -e ".[all]"       # + torch/transformers for open-vocab perception
```

Check it works without any data at all:

```bash
sqe selftest        # 46 hand-derived frame checks on synthetic rooms
pytest              # 168 tests
```

## Quickstart

```bash
# build and cache the scene graphs (~7 s per ScanNet++ scene)
sqe build --root /path/to/scannetpp --all

# ask things
sqe query --scene 0b031f3119 "the monitor to the left of the keyboard"
sqe query --scene 0b031f3119 "the trash can nearest to the door"
sqe query --scene 0b031f3119 "the office chair in front of the whiteboard"

# inspect every frame around one object, and what the policy picks for each relation
sqe frames --scene 0b031f3119 --anchor 20

# the web viewer: point cloud, boxes, frame axes, live query, relation links
sqe viewer --scene 0b031f3119
```

## Verifying the geometry by eye

The fastest way to catch a pose, intrinsic or box-fitting mistake is to project
the 3-D boxes into the real camera frames and look.

```bash
# scene overview + best view of specific classes + a plan view
sqe render --root /path/to/scannetpp --scene 0b031f3119            --per-object chair table monitor whiteboard

# one query, drawn on the frame that best sees the answer
sqe render --root /path/to/scannetpp --scene 0b031f3119            "the monitor to the left of the keyboard"
```

The query render shows the answer in amber, anchors in red, the runner-up in
blue, and the reference frame's `right`/`front` axes drawn at the anchor — so a
left/right answer can be checked against a photograph instead of a number. The
plan view is even quicker for lateral relations, since "left of" is a plan-view
property: if the amber box is on the wrong side of the red arrow, it is wrong.

`run_benchmark.sh` writes these automatically into `renders/verify/`.

This is not decoration. It found three real bugs that the numbers hid:

* **`best_view` preferred frames pressed against an object**, because it clipped
  angular size at 1.0. The "best view" of an office chair was a close-up of the
  cables under the desk. That mattered well beyond rendering: `best_view` is the
  default egocentric viewpoint for every query, and a viewpoint *inside* the
  anchor makes its left and right meaningless.
* **The reported reference frame belonged to the wrong anchor.** With several
  candidate anchors scored jointly, each was resolving its own viewpoint, and the
  frame that got reported was not the frame that scored the winner — so the
  explanation named a keyboard 1.7 m from the one actually used. The viewpoint is
  a property of the query, and is now resolved once and shared.
* **Projective relations had no locality presupposition.** "The monitor to the
  left of the keyboard" was satisfied by a monitor on a different desk 3.2 m
  away, because "to the left of" is geometrically true of almost any distant
  pair. A low-weight proximity term inside a geometric mean cannot veto that; it
  is now a multiplicative gate that scales with the anchor's size, so "in front
  of the whiteboard" still reaches across a room while "left of the keyboard"
  does not.

## Auditing the ground truth

Ground truth is not ground truth.

```bash
sqe audit --scene 0b031f3119
```

```
0b031f3119: 5 of 112 instances flagged
  fitted-vs-annotated box centre error: median 2.97e-07 m, p90 2.84e-03 m, max 0.233 m
  #110 office chair
      - the box fitted to this instance's points is 0.233 m from the annotated
        box centre, so the mask and the annotation disagree about what the object is
      - footprint 1.75 x 0.40 m is 4.4:1, too elongated for a office chair
```

Instance 114 of that scene is labelled `office chair` and is a desk partition.
A benchmark query saying "the office chair" there is unanswerable for reasons
that have nothing to do with spatial reasoning, and the failure would have been
attributed to geometry.

Two independent signals catch it, neither needing a human: the box fitted to the
instance's own points disagreeing with the annotated box (median disagreement
across the scene is 0.3 µm, so 23 cm stands out), and a size implausible for the
label. Flagged instances stay in the scene but are excluded from benchmark
proposal generation. Across the five scenes, **15 of 495 instances (3.0%)** are
flagged.

Scene building is the only expensive step. It is cached, and query time is then
pure geometry over numpy arrays — **single-digit milliseconds**, no GPU, no model
loading.

## The benchmark

One script does everything. Run it, annotate some queries, run it again.

```bash
./run_benchmark.sh /path/to/scannetpp
```

It checks the environment, runs the self-test, builds the scene graphs, proposes
candidate queries, and — once anything is annotated — evaluates and writes the
report. It is idempotent and never overwrites annotations.

Annotate in the terminal (blind, with a top-down map):

```bash
sqe annotate --items benchmark/queries/proposals_5scenes.jsonl \
             --out   benchmark/queries/scannetpp_5scenes.jsonl
```

…or in the browser, clicking the target object:

```bash
sqe viewer --scene 0b031f3119 --items benchmark/queries/scannetpp_5scenes.jsonl
```

**The annotation tool is blind by default** — it does not show what the resolver
would answer. A human confirming a system's own prediction produces a benchmark
that measures agreement rather than correctness, and that is the easiest way to
make the headline number meaningless. `--show-prediction` exists for debugging
and stamps the items so the report can separate them.

Aim for 300–500 items, hard ones first. What matters most is that the `frame`
field says which reading the sentence means, and that genuinely ambiguous queries
are marked rather than forced to one answer.

### What the report contains

`results/<timestamp>/report.md`:

* accuracy per condition — the policy, each **fixed-frame baseline** (what an
  implicit convention amounts to), the oracle frame, the gold parse;
* accuracy split by relation type, by annotated frame, by difficulty, by scene,
  and by whether the sentence stated a frame;
* ambiguity detection as precision/recall/F1 against the annotator's flag;
* how often the frames disagreed at all;
* **failure attribution** — each failure assigned to the first of
  `unresolvable → parse → perception → frame_unavailable → frame_convention →
  geometry → ambiguous_item` that repairs it.

That last table is the point. It is what turns "71% on projective relations" into
"of the projective failures, N% are repaired by forcing the frame the annotator
meant, and only M% by switching to ground-truth perception".

A worked example on synthetic rooms ships in
[`results/synthetic_example/report.md`](results/synthetic_example/report.md).
It is 25 items on two toy rooms — enough to show the report's shape and to
exercise every code path, far too small and too clean to support any claim.

## What is already established

Not benchmark results — those need the annotation. These are properties of the
implementation, measured on the five ScanNet++ scenes and on synthetic rooms with
exact ground truth:

| | |
|---|---|
| Own OBB fit vs ScanNet++ annotation boxes | **0.3 µm median**, 2.8 mm p90 centre error (112 instances) |
| iPhone depth unprojected against the mesh | **1.0 cm median** distance (validates poses + intrinsics + depth decode) |
| Front estimation, synthetic ground truth | **12/14 estimated, all correct, zero 180° flips**, 2 honest abstentions |
| Front estimation vs annotation `dominantNormal` | **83–86% same axis** on real scenes |
| Self-test | **46/46** hand-derived frame checks |
| Room-canonical forward margin, office `0b031f3119` | **0.024** — the allocentric frame is undetermined |
| Frame disagreement, 512 frame-dependent queries | **18.8%** (12.7–19.4% under ±30% threshold jitter) |
| Scene build / query latency | ~7 s per scene / **~2–15 ms** per query |
| Box overlays on real frames | boxes land on the objects across all scenes checked (see `renders/`) |
| Dubious GT instances found | **15 of 495 (3.0%)**, excluded from proposals |
| Open-vocab backend (measured, not assumed) | recall@0.25 **0.63**, recall@0.5 0.33, mean IoU 0.34, label acc 0.35 |

The last one is a finding in its own right: in all five real rooms the
room-canonical frame is close to a coin flip, so any system reporting confident
room-relative directions is reporting one.

## What is not established

Stated plainly, because these are the things a reviewer will find:

* **No accuracy number exists.** Every accuracy, attribution and
  ambiguity-detection figure in the report is defined against hand annotation
  that has not been done yet. The sensitivity numbers above are the only
  quantitative results currently available.
* **47 hand-set constants, zero labelled examples.** 34 in
  `configs/relations.yaml` and 13 module-level. They are set from what the words
  mean and from the geometry of ordinary rooms, and documented with rationales —
  but pre-registration only earns credibility once an evaluation has run against
  it. `sqe robustness` is the partial answer: the sensitivity result holds
  (11.3–26.0%) under ±30% jitter of all 43 query-time constants, so the *size of
  the frame problem* does not hinge on my choices. It says nothing about whether
  the resolver's answers are the ones a person would give.
* **The synthetic suite is saturated and partly circular.** 100% on 25
  hand-derived items is a regression test, not evidence of generalisation — the
  rooms were built by me, and the thresholds were adjusted while looking at them.
  Its value is catching sign and handedness regressions, which it has done
  repeatedly.
* **Five scenes.** Enough for a real finding, not enough to claim it
  generalises. Per-scene tables are in the report so the variance is visible.
* **Ambiguity flags fire on 74% of queries**, dominated by `anchor` (401 of 882)
  and `score_tie` (297) — real rooms contain five keyboards and four tables. The
  claimed kind, `frame`, fires on 96. The report scores each kind separately for
  exactly this reason; a pooled precision/recall would be dominated by the two
  kinds that have nothing to do with the contribution.

## Repository map

```
sqe/
  geom/          transforms, yaw-only OBB fitting, point clouds, room structure,
                 support surfaces and shelf-level detection
  frames/        reference_frame.py  the five frames and their sign conventions
                 cues.py             reading the frame out of the sentence
                 policy.py           the selection table and viewpoint resolution
  relations/     projective (left/right/front/behind), vertical, proximity,
                 ordinal, comparative  -- all thresholds in configs/relations.yaml
  query/         schema, rule parser, LLM parser, resolver, ambiguity
  perception/    orientation.py  intrinsic-front estimation
                 backends/       open-vocabulary instance proposals
  data/          scannetpp, arkitscenes, synthetic rooms with exact ground truth
                 quality.py  flags dubious ground-truth annotations
  bench/         schema, proposal generation, blind annotation, evaluation
  viewer/        stdlib HTTP server + three.js front end
  viz/           3-D boxes projected into real frames, plan views, point splats
  selftest.py    hand-derived frame checks

docs/METHOD.md        the reasoning and the experimental design
docs/CONVENTIONS.md   every sign convention, in one place
configs/relations.yaml every physical threshold, with its rationale
run_benchmark.sh      build -> propose -> annotate -> evaluate
```

## Datasets

**ScanNet++** is the primary target and is fully supported. Three facts were
established by probing the data rather than by reading documentation, and two of
them contradict what is commonly assumed:

* `iphone/depth.bin` is a sequence of `[uint32 size][LZ4 block]` per frame,
  decoding to `uint16` millimetres at 192×256. It is *not* a single zlib stream —
  that is the other format the dataset ships, and these scenes use this one.
* `aligned_pose` is camera-to-world in **OpenCV** axes, in the mesh frame.
  Verified by unprojecting depth and measuring distance to the mesh: 1.0 cm
  median. The OpenGL reading is out by 1.4 m.
* `dslr/nerfstudio/transforms.json` is **not** in the mesh frame. Its
  `aabb_range` is the mesh box under a signed axis permutation, which the loader
  recovers per scene and then verifies — returning nothing, with a reason, when
  it cannot.

**ARKitScenes** is supported as a secondary dataset for one specific reason: its
3DOD annotations ship *oriented* boxes, so the intrinsic frame can be built from
annotated orientation instead of an estimate. Note that an annotated box fixes
the orientation **axis** but not which face is the front, so the sign is still
estimated — treating the axis as a front would be a ground-truth leak dressed up
as a measurement.

**Synthetic rooms** ship with the code and need no download. Two rooms, built so
that the egocentric, intrinsic and allocentric frames give *provably different*
answers to specific sentences, with every answer derived by hand from the layout.
They are the regression suite.

## Status

Working end to end on real ScanNet++ data: loaders, scene graphs, front
estimation, all five frames, the relation set, the resolver, ambiguity
detection, ground-truth auditing, box-overlay verification, proposal
generation, annotation, evaluation, the viewer, and 168 tests.

The open-vocabulary backend runs too: geometric proposals plus multi-view CLIP,
58 s per scene on an M-series Mac via MPS, with its instance quality measured
against the ground truth rather than assumed. It is a component, not a
contribution, and the honest numbers say so — recall@0.25 of 0.63 means a third
of the objects are not found at all. A learned proposal network is the obvious
drop-in (`scene_from_masks`).

Outstanding:

* **the annotation itself** — 882 proposals are generated and waiting. Every
  accuracy number in the report depends on them, and none is claimed until then.
* the LLM parser condition needs an API key to run;
* the open-vocabulary backend would benefit from a GPU pass over all five
  scenes, and from a learned proposal network in place of the geometric one.

## Licence

Apache-2.0.
