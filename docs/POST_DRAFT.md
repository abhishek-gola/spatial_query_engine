# Post draft — reference frames in 3D spatial queries

> Draft. Numbers are current as of the last run; regenerate with
> `./run_benchmark.sh <data>` before posting. **There is still no end-to-end
> accuracy number** — that needs hand-annotated frame labels. Nothing in this
> draft claims one.

**Asset:** `renders/gif/frame_switch.gif` — *"the first pillow from the left on
the bed"*, two pillows side by side, and the highlighted one jumps when the
reference frame changes. Lead with it. Boxes are drawn only on the objects the
relation involves, and occluded edges are removed against sensor depth. It is a
rendering of the resolver's own output on a real capture frame, not a screen
recording of the viewer; say so if anyone asks.

---

## Version A — short (LinkedIn)

> **"The first pillow from the left on the bed."** Two pillows, side by side.
>
> From where I'm standing, the first from the left is the near one. Counting from
> the bed's own left — the left of someone lying in it — it's the other one. Both
> are ordinary English. Both are right. The sentence doesn't say which is meant.
>
> [GIF: same scene, same camera, same geometry. Only the reference frame changes,
> and the answer moves.]
>
> **So I gave two models 35 pairs of sentences that differ in nothing but an
> explicit marker of whose "left" is meant.**
>
> *from where I am standing, the cabinet to the left of the bed*
> *the cabinet to the bed's own left*
>
> Same scene, same objects, same observer position, stated. Only one thing
> differs, and it is the one thing that determines the answer. On scenes where
> the two readings provably pick different objects.
>
> The stronger model got both arms right on **12 of 28** pairs. On **8** it gave
> the **identical answer to both** — told explicitly which frame to use, twice,
> and the instruction did nothing. Frontal relations were worse than lateral: it
> followed "the bed's own left" more often than "from the bed's point of view".
>
> And when no frame is marked at all, its silent default is the **viewer's** —
> 22 of 35.
>
> That last number is the interesting one, because:
>
> **Sr3D — 83,572 utterances, the standard benchmark for spatial reference in
> 3D — has a relation class named `allocentric` covering left/right/front/back,
> roughly 6,700 utterances. Its own supplementary defines it as computed "with
> respect to the anchor orientation", from PartNet front annotations and Scan2CAD
> 9DoF alignments, over "four oriented sections (front, back, left, and right)".
> That is the object's own frame. The words "camera", "viewer", "observer" and
> "point of view" appear nowhere in the 13-page supplementary.**
>
> So: the benchmark scores one reading. The model defaults to the other. Neither
> of them says which one out loud.
>
> (It *is* documented — filed under a name many readers will take to mean the
> room's axes. The problem isn't concealment, it's that a single convention
> covers ~6,700 utterances and no headline number mentions it.)
>
> Why this matters beyond the pedantry: when a grounding pipeline gets one of
> these wrong, the failure looks like a wrong object. So it gets logged as a
> perception error and attacked with a bigger detector. It is not a perception
> error, and a bigger detector will not fix it.
>
> What I built to measure it: a spatial query engine for indoor 3D scenes that
> treats the reference frame as an explicit parameter — five frames, an estimated
> intrinsic front per object, room-canonical axes with an ambiguity margin, and
> ambiguity reported as an output rather than swallowed. Plus the minimal-pair
> probe above, with a positive control (a system with its frame pinned scores
> 100% frame-blind, as it must) and a negative control (35 non-contrastive
> paraphrases, so "frame-blind" can be told apart from "unstable to any
> rephrasing").
>
> No end-to-end accuracy number yet — that needs hand-annotated frame labels, and
> 882 items are generated and waiting.
>
> On process: I drove this hard with agent tooling, and the commit history shows
> it. What is mine is the judgement — reading Sr3D's supplementary and finding the
> convention, the three bugs I only caught by rendering the boxes onto real camera
> frames and counting objects by eye, and retracting my own headline number when I
> found my generator had enriched the sample (18.8% → 4.1%, inflated 4.6×).
>
> Code, method, the audit trail, and the related work this builds on: <link>

---

## Version B — longer, with the mechanics

Everything in A, plus:

**The bit that surprised me.** Fix the convention that a frame's `front` axis
points where "in front of" means and its `right` axis points where "to the right
of" means. Then the object-centric frame is right-handed and the viewer-centric
one comes out **left-handed**. That handedness flip isn't a bug to normalise away
— it *is* the mirror error. It's why the mug on your left is on the chair's right,
and why a pipeline that builds one basis and reuses it for both axes is wrong for
one of them whichever basis it picks.

**The asymmetry that does the work.** `front`/`behind` default to the object's own
frame — "in front of the sofa" means the side the sofa faces, nobody means
"between me and the sofa". `left`/`right` default to the viewer's. One reused
basis is wrong for one of them. The probe result lines up with this: the model
followed an explicit *lateral* cue far more often than an explicit *frontal* one,
which is what you'd expect if the frontal default is entrenched enough to
override being told.

**Ambiguity is an output, not an error.** When the plausible frames disagree, the
honest answer is "here's the default reading, and here's what the other reading
gives" — not a confident single object.

---

## The probe, in full

35 minimal pairs, 35 non-contrastive controls, four ScanNet++ scenes. Protocol
and every design decision: `docs/FRAME_PROBE_PROTOCOL.md`. Full table:
`results/frame_probe/frame_probe.md`.

| system | pairs | switched correctly | frame blind | switched wrongly | partial | control stable |
|---|---|---|---|---|---|---|
| resolver, cue-following *(circularity check)* | 35 | 100% | 0% | 0% | 0% | 100% |
| resolver, frame pinned *(positive control)* | 35 | 0% | **100%** | 0% | 0% | 100% |
| Claude Opus 5 *(self-administered)* | 35 | 42.9% | **25.7%** | 11.4% | 20.0% | 80.0% |
| Claude Haiku 4.5 *(self-administered)* | 35 | 5.7% | **48.6%** | 14.3% | 25.7% | 37.1% |

Restricted to the pairs where that system answered its **own** control pair
consistently — the item-level version of the stability check, and the only
reading of `frame_blind` that survives a noisy system:

| system | pairs kept | switched correctly | frame blind | partial |
|---|---|---|---|---|
| Claude Opus 5 | 28 of 35 | 12 (42.9%) | **8 (28.6%)** | 5 (17.9%) |
| Claude Haiku 4.5 | 13 of 35 | 1 (7.7%) | **8 (61.5%)** | 3 (23.1%) |

**Do not quote the Haiku row as a finding.** Its control stability is 37% — it
gives different answers to two paraphrases that differ in nothing at all — so its
frame-blindness is unattributable: it could be "no frame to instruct" or just
"can't hold a long object list". That is exactly what the negative control is
for, and the honest report is that this arm failed its own precondition. Thirteen
items is also too few.

**Three things that cut against me, stated here rather than left to be found.**

1. The self-administered arm is a **pilot, not an independent measurement**.
   Blocking the trials means no answerer saw both arms of a pair, but the author
   of the stimulus and the answerer are the same model family, and the author
   knew what was being tested. The API harness for a genuine multi-vendor run is
   built (`sqe frame-probe --model frontier`) and waiting on a key.
2. On the **control** sentences — plain, unambiguous, no frame marker — the
   stronger model agreed with my resolver on only **16 of 35**. The controls test
   *stability*, not correctness, and this says that on ordinary sentences the
   model and I disagree about half the time. Which of us is right is exactly what
   hand annotation would settle and I have not done it. So the probe supports
   "the frame instruction often does nothing"; it does **not** support "the model
   is worse at spatial language than my resolver".
3. Four trials asked about an anchor that was **missing from the object list** —
   the listing filter dropped room-fixed objects and some anchors are room-fixed.
   Four answerers reported it independently, with no shared context and no access
   to the key. Fixed, tested, and the affected trials re-asked. Worth saying out
   loud because it is the strongest evidence the blinding was real: an answerer
   who could see the key would have had no reason to notice, and none to mention
   it.

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

**"You tested Claude with Claude."** Yes, and it is labelled that way in every
table. See caveat 1 above. The stimulus, the controls, the blocking and the
scoring are all vendor-independent, and the multi-vendor command is one
environment variable away from running.

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
is why it sits second.

---

## Not in this draft, deliberately

* **Any end-to-end accuracy number.** Needs annotation.
* **Any claim that the probe result generalises across vendors.** Two models,
  one family, self-administered. Labelled everywhere.
* **The VLM baseline.** The harness is built and tested (`sqe vlm-baseline`) —
  hand a model the same scene graph, ask which object the sentence refers to,
  then classify its answer by which frame it matches. It has **not been run** as
  an API sweep. The uncued-sentence column of the probe table already answers the
  same question on a smaller item set: 22 of 35 egocentric for the stronger
  model.
