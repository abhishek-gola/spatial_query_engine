# Post draft — reference frames in 3D spatial queries

> Draft. Numbers are current as of the last run; regenerate with
> `./run_benchmark.sh <data>` before posting. **The accuracy numbers do not exist
> yet** — nothing in this draft claims one, and the draft should not acquire one
> until the benchmark is annotated.

**Asset:** `renders/gif/frame_switch.gif` — one query, one scene, one camera, and
the highlighted object jumps when the reference frame changes. Lead with it. It
is a rendering of the resolver's own output, not a screen recording of the
viewer; describe it as such if anyone asks.

---

## Version A — short (LinkedIn)

> *"The mug to the left of the laptop."* Left from whose view?
>
> Yours? The laptop's own front? The room's axes? All three are legitimate
> readings, and in a real room they pick different mugs.
>
> I built a spatial query engine for indoor 3D scenes that treats the reference
> frame as an explicit parameter rather than an accident. Then I measured how
> much it matters.
>
> **Sr3D — 83,572 template utterances, the standard benchmark for spatial
> reference in 3-D — has a relation class named `allocentric` covering
> "left/right/front/back", about 6,700 utterances. Its own supplementary defines
> it as computed "with respect to the anchor orientation", from Scan2CAD 9DoF
> alignments. That is the *intrinsic* frame. "Allocentric" means the world frame
> — the opposite.**
>
> The words "camera", "viewer", "observer" and "point of view" appear nowhere in
> the 13-page supplementary. Sr3D contains no viewer-relative projective
> utterances at all — and viewer-relative is the dominant reading of "left of X"
> in ordinary English. A model evaluated on Sr3D is rewarded for one reading and
> penalised for the other, and no headline number says so.
>
> To be fair: it *is* documented, in the supplementary. The problem isn't
> concealment, it's that a single convention is applied to ~6,700 utterances
> under a name meaning the opposite frame.
>
> On my own five ScanNet++ scenes, forcing that intrinsic convention changes
> **34%** of answerable frame-dependent answers.
>
> Why this matters: when a grounding pipeline gets one of these wrong, the
> failure looks like a wrong object. So it gets counted as a perception error and
> attacked with a bigger detector. It isn't a perception error.
>
> The GIF is the whole idea in three seconds: same scene, same camera, same
> geometry. Only the frame changes.
>
> Not claimed: any accuracy figure. That needs hand-annotated frame labels, and
> 882 candidate queries are generated and waiting. Five scenes is thin, and
> ground-truth perception is generous. Both are stated in the README rather than
> buried.
>
> Code, method, and the related work this builds on: <link>

## Version B — longer, with the mechanics

Everything in A, plus:

**The bit that surprised me.** Fix the convention that a frame's `front` axis
points where "in front of" means and its `right` axis where "to the right of"
means. Then the object-centric frame is right-handed and the viewer-centric one
comes out **left-handed**. That handedness flip isn't a bug to normalise away —
it *is* the mirror error. It's why the mug on your left is on the chair's right,
and why a pipeline that builds one basis and reuses it for both axes is wrong for
one of them whichever basis it picks.

**The asymmetry that does the work.** `front`/`behind` default to the object's own
frame — "in front of the sofa" means the side the sofa faces, nobody means
"between me and the sofa". `left`/`right` default to the viewer's. One reused
basis is wrong for one of them.

**Ambiguity is an output, not an error.** When the plausible frames disagree, the
honest answer is "here's the default reading, and here's what the other reading
gives" — not a confident single object.

---

## Ambiguity detection — report per kind, never pooled

The system flags some form of ambiguity on **653 of 882 queries (74%)**. Quoted
pooled, that is a system that cries wolf. The breakdown is why pooling is the
wrong summary:

| kind | flagged | what it is |
|---|---|---|
| `anchor` | 401 | several objects match the anchor class equally well |
| `score_tie` | 297 | the top two candidates score within a whisker |
| `weak_match` | 114 | nothing in the scene really satisfies the query |
| **`frame`** | **96** | **the plausible reference frames disagree** |
| `ordinal_tie` | 28 | two candidates too close along the counting axis |
| `ordinal_degenerate` | 10 | the candidates aren't ordered along that axis at all |
| `level_even` | 5 | "the middle shelf" of an even number of shelves |

`anchor` and `score_tie` dominate for a reason that has nothing to do with
reference frames: a real office contains five keyboards, four tables and eleven
chairs, so "the keyboard" genuinely *is* ambiguous. Those flags are correct. But
they are a property of how many instances a room holds, not of the frame
resolver.

So a single pooled precision/recall would be dominated by the two kinds the
project does not claim, and would make the contribution look bad for the wrong
reason. **`frame` is the kind being claimed** (96 of 882), and the benchmark
scores each kind separately, with a `frame`-restricted-to-frame-dependent view as
the headline ambiguity metric and `pooled` printed last and explicitly
de-emphasised.

If someone quotes 74% at you, that is the number for "flagged anything at all",
and the composition above is the answer.

---

## What to say if pushed

**"Isn't this just Levinson's frames of reference?"** Yes, the trichotomy is
standard in linguistics and has been for decades — Levinson (2003) gives it a
chapter. The contribution is making it operational in a 3D pipeline and measuring
it, not discovering it. That's stated in the README and in
`docs/RELATED_WORK.md`.

**"Don't SpatialVLM / SpatialRGPT already do this?"** They do the grounding, not
the frame. SpatialRGPT computes left/right by traversing object nodes using "the
point cloud centroids and bounding boxes" — no frame named. SpatialVLM's frame is
the camera's, stated inside one template string. Literal quotes with page numbers
are in `docs/RELATED_WORK.md`.

**"Five scenes?"** Yes. It's thin, it's stated next to the headline number, and
per-scene numbers are in the report so the variance is visible. More scenes cost
43 MB each (mesh + segments + poses; the video isn't needed) and are the obvious
next step — but they don't substitute for annotation, which is what actually
blocks the accuracy number.

**"47 hand-tuned constants and no labelled data?"** Fair, and it's in the README's
"What is not established". The partial answer is `sqe robustness`: jitter all 43
query-time constants by ±30% and the unenriched disagreement rate stays in
2.0–6.8% (median 3.7%, configured 4.1%). So the *size* of the frame problem
doesn't hinge on my choices. It says nothing about whether my resolver's answers
are the ones a person would give — only annotation can.

**"Where does 4.1% come from, and didn't you say 18.8%?"** I did, and it was
wrong. That figure came from an item set my generator had enriched for frame
sensitivity — sensitive candidates sorted first, then capped — so it was a rate
over queries *selected for being frame-sensitive*, inflated 4.6×. The unenriched
population rate is 4.1%. Both runs are kept and every report now declares which
kind of sample it used. The enriched number still says something true — look for
frame-ambiguous queries and you find 96 in 512 — it just isn't a population rate.

**"Isn't the item set your own generator's?"** Yes, and that's the main
limitation of the measurement rather than of the idea. The recorded decision is
that the benchmark scores the field, not my resolver, which means items must come
from real Nr3D utterances. The Sr3D finding above needs none of my items, which
is why it leads.

---

## Not in this draft, deliberately

* **Any accuracy number.** Needs annotation.
* **Any minimal-pair model result.** The stimulus is built and the metric is
  calibrated (35 pairs, 35 non-contrastive controls; cue-following resolver 100%
  switched, pinned frames 100% frame-blind at 100% control-stability). No model
  has been run — it needs an API key. `sqe frame-probe --model <name>`.
* **The VLM baseline.** The harness is built and tested (`sqe vlm-baseline`) —
  hand a model the same scene graph, ask which object the sentence refers to,
  then classify its answer by which frame it matches. The prompt never mentions
  frames and object ids are shuffled, so a concentrated result would mean the
  model has absorbed one convention silently. It has **not been run** — it needs
  an API key. This would be the strongest version of the post, and it is one
  evening away.
