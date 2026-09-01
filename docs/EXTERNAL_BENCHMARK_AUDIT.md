# Auditing the standard benchmarks

## The decision this document records

**This benchmark scores the field, not the resolver.**

Consequences, and they are binding:

* Items come from **real Nr3D utterances**, not from my proposal generator. A
  generated item set can only ever show that *my* system's frames disagree; an
  Nr3D item set shows that the field's standard labels are frame-dependent.
* My resolver appears as **one system among several**, and its agreement with
  anything is never the headline.
* Anywhere my generator's items are used, the number is a property of my
  generator and is labelled as such. See "Enrichment" below, which is exactly
  this mistake caught after the fact.

---

## Finding 1 — the standard benchmark has no viewer-relative "left", by design

**Verified from the ReferIt3D ECCV 2020 paper (Table 1, p.7) and its
supplementary (§2.5, p.9).** Both PDFs are in `3d-papers/pdfs/` (901-, 900-).

Table 1 gives the exact composition, in contexts and utterances:

| | Horiz. prox. | Vert. prox. | Support | **Allocentric** | Between | All |
|---|---|---|---|---|---|---|
| contexts | 34,001 | 1,589 | 747 | **1,880** | 3,569 | 41,786 |
| utterances | 68,002 | 3,178 | 1,494 | **3,760** | 7,138 | 83,572 |

So the projective class is **3,760 utterances, 4.50% of Sr3D**.

> **Correction.** An earlier version of this document, and of the README, said
> "~8%, ≈6,700 utterances". That was wrong: 8.5% and 3,569 contexts are the
> **`between`** row, and ≈6,700 came from multiplying that share by the 83,572
> total. Two rows of the same table were read across each other. The correct
> figure is 3,760 utterances (4.50%), and it is smaller than the number the
> mistake produced.

The supplementary defines the class:

> "**Allocentric Relations.** These relations indicate where a target object
> might exist **with respect to the anchor orientation**. For example, the
> armchair (target object) that is at the right of the TV (anchor object). For
> generating allocentric relations, we need to know: (a) whether the anchor
> objects have an **intrinsic front view** (e.g., armchair) or not (e.g.,
> stool); and (b) the orientation of the objects in ScanNet. For (a), we used
> the annotations of **PartNet** […] For (b) we utilized the **Scan2CAD**
> annotations that provide **9DOF alignments** between ShapeNet models and
> ScanNet objects. For every anchor object that has an intrinsic front view, we
> create **four oriented sections** (regions) […]"

And from the figure caption:

> "This figure shows how we determine where the target object might exist with
> respect to one of the four oriented sections of the anchor (front, back, left,
> and right)."

### This is a design consequence, not an oversight

The paper says what it is doing and why. View-independence is an explicit,
stated goal of the work, not something it forgot about:

> "…**without a camera view dependency** – can benefit many downstream robotics
> applications…" (p.2)

> "This flexibility enables us also to **bypass camera view dependency**." (p.2)

> "In order to **remove any camera view bias**, we initialize the 'speaker' and
> 'listener' 3D interfaces with different randomized camera parameters." (p.8)

Nr3D even carries `View-Independent` as one of its annotated utterance
properties, and the paper reports that view-independence "does not require the
observer to place themselves into the scene facing certain objects" (p.9).

Read against that goal, Sr3D's `allocentric` class is the **only** thing it could
have been. Remove the camera from the loop and the anchor's own front is the one
frame still available for left/right, which is exactly why the generator reaches
for PartNet fronts and Scan2CAD 9DoF alignments. The design is coherent on its
own terms, and nothing is concealed: the convention is stated in the
supplementary and its motivation is stated in the introduction.

**That makes the finding stronger, not weaker.** The problem is not a mistake
someone made; it is a structural property of how the field's standard resource is
built:

> Sr3D's 3,760 projective utterances are all computed in the anchor's intrinsic
> frame, because view-independence is a design goal. It therefore contains **no
> viewer-relative projective utterances at all** — not few, none. Searching the
> whole 13-page supplementary for `camera`, `viewer`, `observer`, `point of view`,
> `egocentric` and `view-dependent` returns **zero matches**. But each utterance
> still carries a **single gold answer**, so a model that reads "left of the TV"
> the way most English speakers do is marked wrong, and the frame is nowhere a
> parameter of the evaluation. The dataset's declared frame and the model's
> silent default are different, and the scoring cannot see the difference.

**The naming point is dead and should not be made.** An earlier version of this
document argued that "allocentric" denotes the world frame in the standard
taxonomy (Levinson 2003, ch. 2) and so mislabels what Sr3D computes. But the main
paper's own definition of the class says exactly what frame it uses:

> "(iv) **Allocentric:** Allocentric relations encode information about the
> location of the target with respect to the **intrinsic self-orientation** of an
> anchor." (p.6)

The word *intrinsic* is theirs, in the definition, in the main paper. Nothing is
mislabelled, and the argument does not need it — it was a weaker claim standing in
front of a stronger one.

One further detail that cuts in the same direction: Nr3D's own annotation finds
only **63%** of its utterances view-independent, so roughly a third *are*
view-dependent — and there the paper notes speakers "were instructed to guide the
listeners on how to place themselves in the scene" (p.9). The field is aware the
frame matters. What is missing is the frame as an explicit parameter of the
score.

**Why this matters quantitatively.** On my own five ScanNet++ scenes, forcing the
intrinsic frame changes **34.0%** of answerable frame-dependent answers relative
to the natural-default policy (`results/sensitivity_unenriched/`). So the
convention Sr3D picked is not a detail.

**A gift, not just a finding.** Sr3D's orientation source is **Scan2CAD 9DoF
alignments**. That is precisely the ground-truth orientation the audit wants, and
it means the Scan2CAD-covered subset is already the subset where Sr3D's
allocentric items live. On that subset the intrinsic frame needs no estimator at
all.

### What still has to be checked

* Whether Sr3D's four oriented sections are *quadrants* or *cones*, and the
  occupancy threshold — both change which items are frame-ambiguous.
* Whether ScanNet's world axes make an allocentric reading even well-defined
  (my five scenes say the room-canonical frame is near-undetermined, margin
  0.02–0.22).
* **Nr3D** has no such control: it is human free-form, so it contains a
  *mixture* of frames, unlabelled. That is where the ambiguity claim bites, and
  it is why the item set must come from Nr3D.

---

## Finding 2 — my own 18.8% was enrichment, not a population rate

Caught by review, and it was a real error in a number I had already put in the
README.

`propose_projective` sorted frame-sensitive candidates first and *then* capped
per relation. Every rate measured over the resulting 882 items was therefore a
rate over "queries selected for being frame-sensitive".

| | frame-dependent | disagree | rate |
|---|---|---|---|
| enriched (as previously reported) | 512 | 96 | **18.8%** |
| **unenriched, random sample** | 512 | 21 | **4.1%** |

Inflated **4.6×**. By relation type, unenriched: lateral **4.8%** (was 24.8%),
frontal **2.6%** (was 15.4%), ordinal 7.4% (unchanged — ordinals were never
enriched).

Fixes: `--no-enrich` samples at random; `sqe sensitivity --enriched yes|no`
prints a banner on every report saying which it is; both runs are kept, in
`results/sensitivity/` and `results/sensitivity_unenriched/`.

**The enriched number is not worthless — it is just a different claim.** It says
that when you deliberately look for frame-ambiguous queries you find plenty
(96 in 512). It cannot be quoted as "X% of spatial queries are frame-ambiguous".

A second error found while fixing this: the report's prose said queries with no
answer under a forced frame "are counted as unchanged", when `None != id` meant
they counted as *changed*. The flip rate is now reported both ways, and the
answered-only column is the one to read.

---

## Experiment plan

### E1 — audit Nr3D / Sr3D / ScanRefer on ScanNet v2

**Primary, strong version: the Scan2CAD-covered subset.** Anchor orientation
comes from annotated 9DoF CAD alignment, so the intrinsic frame is ground truth
rather than my estimate. Smaller n, far stronger claim, and it is the same subset
Sr3D itself used.

**Secondary, clearly labelled: the full set with estimated fronts.** Reported
separately and never merged with the above.

Two numbers per corpus:

1. Of items whose utterance contains a projective term, what fraction are
   **frame-ambiguous** — two constructible frames select different objects while
   the benchmark assigns one gold answer. That is a noise floor on every accuracy
   ever reported on that benchmark.
2. Do the gold answers **cluster on one frame**? For Sr3D the answer is known a
   priori from the supplementary (intrinsic, by construction) and the audit
   becomes a *verification* that the generator does what it documents. For Nr3D
   it is an open empirical question and the real result.

Status: loader and audit scaffolding to be built; needs ScanNet v2 (licence) plus
the Nr3D/Sr3D CSVs and Scan2CAD annotations.

### E2 — minimal-pair instructability probe

Built and **run**: `sqe/bench/minimal_pairs.py`, `sqe/bench/frame_probe.py`,
`sqe/bench/selfprobe.py`, `sqe/bench/vendors.py`. Protocol written up in
[FRAME_PROBE_PROTOCOL.md](FRAME_PROBE_PROTOCOL.md).

35 validated pairs from 4 of the 5 scenes, plus **35 non-contrastive controls**
— the requirement recorded here previously, now met. Calibration holds:
cue-following resolver 100% `switched_correctly`, all four pinned frames 100%
`frame_blind` at 100% control stability.

Result, self-administered arm (Claude Opus 5 / Claude Haiku 4.5, each block of
trials answered by an isolated agent that never saw the other arm of any pair).
**The reportable finding is the uncued default:** on the plain, unmarked
sentence Opus's answer matches the viewer-relative reading on **22 of 35** pairs,
the anchor's own frame on 4, neither on 9 — and 9-to-3 viewer-first even on
`front`/`behind`. That is the number that meets Finding 1 head-on, since Sr3D's
projective class is entirely intrinsic.

**The instructability rate is only reportable conditioned on baseline
agreement.** `frame_blind` means "same answer to both cued arms", but a system
that already disagrees with this resolver about the unmarked sentence can give
two identical answers as a consistent reading of a sentence I resolve
differently; pooling charges that to frame-blindness. On the 16 pairs where Opus
and this resolver agree about the unmarked sentence: 12 switched correctly, **2
frame-blind**. The pooled figures (42.9% switched, 25.7% frame-blind) are kept in
`results/frame_probe/frame_probe.md` and marked not-quotable.

Two analyses that were tried and should **not** be published: the pooled
frame-blind rate (above), and a lateral-vs-frontal asymmetry in cue-following —
the raw split is 11 of 18 against 4 of 17, but it tracks baseline agreement
almost exactly (72% vs 18%) and the conditioned frontal subset is n=3, so
attributing it to the front/back-is-intrinsic default is circular.

Haiku fails the control-stability precondition (37%) and its row is reported as
uninterpretable rather than as a finding.

Two decisions made while running it, both of which changed the numbers:

* **Pairs now record a concrete observer position**, not the rule that found one.
  A pair storing `viewpoint: best_view` marks its egocentric arm against a
  viewpoint the answerer was never told about, so that arm was unanswerable by
  construction. Fixed in `generate_for_scene`; `build_prompt` now refuses a pair
  without a concrete position rather than silently defaulting.
* **The object listing must contain the anchor and every candidate.** The listing
  filter dropped room-fixed objects and some anchors are room-fixed, so four
  trials asked about a heater or a wall clock that was not in the list. Four
  independent answerers reported it. Fixed and tested.

**Still open:** an independent, multi-vendor API run. The harness dispatches on a
vendor-prefixed model name and has a `--preflight` reachability check; it needs
`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY`. Until that runs, the
result is a pilot from one model family and is labelled as such everywhere.

### E3 — human disagreement, run first

Reshaped: **5–7 raters × 120–150 items**, not 20 × 20. Same budget, but it
supports a *per-utterance* claim — "this utterance has no single answer, and here
is the human distribution" — instead of a demo.

Items drawn from **Nr3D utterances**, so the human distribution is directly
comparable with Nr3D's gold label.

### E4 — annotator-population stability (was "cross-linguistic")

Reframed. The original framing promised a Hindi/English difference; both are
relative-frame-preferring languages, so a null result would be uninterpretable
and a positive one hard to attribute. The defensible question is **convention
stability across annotator populations**: does the frame distribution shift
between recruitment pools at all?

Recruits screened for English fluency, with fluency **recorded as a covariate**
rather than used as a filter only.
