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

## Finding 1 — Sr3D fixes the frame to *intrinsic* and calls it "allocentric"

**Verified from the ReferIt3D ECCV 2020 supplementary, §2.5 (p.9).** The PDF is
in `3d-papers/pdfs/900-referit3d-…-SUPPLEMENTARY.pdf`.

Sr3D has five relation types with roughly this distribution: horizontal
proximity ~81%, **allocentric ~8%**, support ~5%, vertical proximity ~4%,
between ~2%, over 83,572 utterances. So the projective class is **≈6,700
utterances**.

The supplementary defines that class:

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

Three things follow.

**1. The category name denotes the wrong frame.** In the standard taxonomy
(Levinson 2003, ch. 2) *allocentric* / absolute means the world or environment
frame — cardinal directions, room axes. What Sr3D computes is the **intrinsic**
frame: the anchor's own front, back, left and right. A reader who sees
"allocentric" and reasons about the dataset accordingly will mis-predict its
behaviour on every one of those ~6,700 utterances.

**2. The viewer never appears.** Searching the whole 13-page supplementary for
`camera`, `viewer`, `observer`, `point of view`, `egocentric` and
`view-dependent` returns **zero matches**. Sr3D therefore contains **no
viewer-relative projective utterances at all** — not a few, none. The relative
frame, which is the dominant reading of "left of X" in ordinary English, is
absent by construction.

**3. It is documented, and the claim must respect that.** The convention *is*
stated, in supplementary material. So the honest claim is **not** "and never says
so". It is:

> Sr3D applies a single intrinsic convention uniformly across ~6,700 projective
> utterances, files it under a name that means the opposite frame, and therefore
> contains none of the viewer-relative readings that dominate natural English. A
> model evaluated on Sr3D is *rewarded* for the intrinsic reading and *penalised*
> for the relative one, and nothing in the benchmark's headline numbers reveals
> that.

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

Built: `sqe/bench/minimal_pairs.py`, `sqe/bench/frame_probe.py`.
35 validated pairs from 5 scenes. Controls calibrated: cue-following resolver
100% `switched_correctly`, all four pinned frames 100% `frame_blind`.

**Still required before any model result is meaningful:** a control set of
paraphrases that are *equally awkward but not frame-contrastive*, with the same
correct answer. Without it, `frame_blind` cannot be separated from "the model
failed to parse an unusual sentence". This is the difference between a result and
an artefact.

Budget: two weeks, not one. Most of the work is normalising inputs across a 2-D
VLM, a region VLM and a 3-D grounder — not the probe.

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
