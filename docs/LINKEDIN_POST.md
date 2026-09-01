# LinkedIn post — ready to paste

**Asset:** `renders/gif/frame_switch.gif` — *"the office chair to the right of
the monitor"*, an office, two chairs either side of a desk with a monitor on it.
**Not the whiteboard query**: every whiteboard-anchored query fails to compose,
and the reason is recorded under "Before posting". This one is the closest
composable match to the post's own hook — a screen-shaped anchor whose facing
everyone can read, a lateral relation, and two readings that are exact mirror
images.
**Length:** 2,343 characters over 15 short paragraphs. LinkedIn's limit is 3,000;
the first ~210 characters show before "see more", and the hook fits inside them.

The first five paragraphs establish the setting, the stake and what the project
*is*, before any number appears. A reader who stops after them still knows why
left and right are the hard case and what was built. Everything technical comes
after, in paragraph-sized pieces, so a skimmer can drop out at any point without
being left mid-argument.

Every number was verified against the ReferIt3D ECCV 2020 paper and its
supplementary PDF, and against the raw answer files in `results/self_probe_v2/`
(`opus_merged.json`; `results/self_probe/` holds the first pass, before four
trials with a missing anchor were re-asked).
Sources are listed under the post.

---

Say "bring me the mug to the left of the laptop" to a home robot, and before it can move it has to answer a question you never asked: left from whose point of view?

Yours. The laptop's own front. The room's axes. All three are real readings, and in a real room they point at different mugs.

Most spatial language is safe from this. "On the table", "nearest the door" — one answer, wherever you're standing. Left, right, in front of and behind are the exceptions. They only mean something relative to a point of view, and English almost never says which one.

So I built a spatial query engine for indoor 3D scenes. Walk a room with a phone, it builds a 3D map of the objects and which way each one faces, then you type a sentence and it points at one.

The reference frame is an explicit input rather than an accident. When two readings pick different objects, it says so instead of quietly choosing.

Then I checked whether a model has the same problem.

35 minimal pairs on real ScanNet++ scans: sentences differing only in a marker of whose "left" is meant, on layouts where the two readings provably pick different objects.

Given the scene as a list of objects and where the observer stands, Claude Opus read the unmarked sentence in the viewer's frame 22 times out of 35. The object's own frame: 4. Even on "in front of" and "behind", 9 to 3.

Now Sr3D, the standard template benchmark for spatial reference in 3D.

Its left/right/front/back class — 3,760 of its 83,572 utterances — is built from "the intrinsic self-orientation of an anchor". Search the 13-page supplementary for "camera", "viewer" or "egocentric": zero hits.

And the same paper measured that only 63% of the natural human utterances in its sibling dataset are view-independent.

So about a third of real spatial reference is viewer-relative, the benchmark contains none of it, and the model defaults to it 22 to 4.

Which matters because a wrong frame returns a wrong object. It looks exactly like a detection failure, so it gets logged as one and answered with a bigger detector.

Limits: self-administered pilot, 35 pairs, 5 scenes. Blocks went to isolated agents with no access to the key and the tests assert that, but stimulus author and answerer are the same model family. The multi-vendor run is built, waiting on a key.

Code, method and raw answers: <link>

---

## Sources for every claim above

| Claim | Source |
|---|---|
| 3,760 of 83,572 utterances | Main paper, Sr3D statistics table: Allocentric = 1,880 contexts / 3,760 utterances; total 41,786 / 83,572 |
| "the intrinsic self-orientation of an anchor" | Main paper, §Spatial References, item (iv) — verbatim |
| Scan2CAD 9DoF alignments, four oriented sections | Supplementary §2.5 "Allocentric Relations", p.9 — verbatim |
| Zero hits for camera / viewer / observer / point of view / egocentric | Supplementary, all 13 pages, text layer present on every page (also zero for viewpoint, deictic, perspective, view-dependent; "allocentric" appears 3x, all on p.9) |
| 63% of Nr3D utterances view-independent | Main paper, Nr3D analysis: "the identification of the target is view-independent … Although this attribute is not as prominent as the previous one (63%)" |
| 22 / 4 / 9-to-3 | `results/self_probe_v2/opus_merged.json`, neutral-arm trials, 35 pairs, scored against both frames' golds; reproduced in `results/frame_probe/frame_probe.json` under `neutral_implies` |

## Deliberately left out, and why

- **42.9% switched correctly / 25.7% frame blind.** Stratify by whether the model
  and this resolver agree on the unmarked sentence and the frame-blind rate is
  2 of 16 on the informative subset. Underpowered; do not publish as a headline.
- **The lateral-vs-frontal asymmetry.** It tracks baseline agreement (72% lateral
  vs 18% frontal), not cue-following, and attributing it to the front/back-is-
  intrinsic default is circular.
- **"Allocentric is a misnomer".** The main paper calls the class "intrinsic
  self-orientation" in its own definition. Nothing is mislabelled and the
  argument does not need it.
- **The 4.1% frame-disagreement rate and the 34% forced-intrinsic figure.** Both
  are properties of this generator and this unvalidated policy.

## Before posting

All three done — commit `HEAD`. What changed:

1. **Numbers fixed.** `README.md` and `docs/EXTERNAL_BENCHMARK_AUDIT.md` now cite
   Table 1 directly: allocentric = 1,880 contexts / **3,760 utterances (4.50%)**.
   The audit doc carries an explicit correction note saying the old "~8% /
   ≈6,700" was the `between` row read across, so the mistake is on the record
   rather than quietly deleted.
2. **Concealment framing gone.** Both documents now lead with view-independence
   as an explicit design goal, quoting "without a camera view dependency",
   "bypass camera view dependency" and the randomised speaker/listener cameras
   that "remove any camera view bias" — and state that the intrinsic frame is the
   only one left once the camera is out of the loop. The misnomer argument is
   deleted outright: the main paper's own definition says "intrinsic
   self-orientation", so there was nothing to complain about.
3. **GIF recropped, retitled, and re-shot on a different query.** Title 3.2x
   (cap height 6px → 21px at feed scale), anchor wireframe and green front axis
   gone, filled glow plus a heavy stroke on the answer, cross-image connector
   line dropped, cropped to the two candidates. `sqe gif --style poster` is now
   the default; `--style debug` restores the diagnostic view.

   **The whiteboard query does not work, and I could not make it work.** All 13
   frame-split queries with a board-family anchor were checked against every
   camera frame. The three that compose all have a weak anchor: two resolve
   "whiteboard" to `board#50`, which is a 0.63 x 0.58 x 0.09 m board lying flat on
   a table — captioning that "whiteboard" would be the misleading kind of figure —
   and the third to `whiteboard#11`, whose estimated front carries 0.20
   confidence. The one real whiteboard, `#20` at 2.01 x 0.03 x 1.25 m, never
   composes: of 2,473 frames sampled, 17 put it and both candidate tables in shot,
   separated and non-overlapping, and in every one of those 17 the losing table is
   between **4% and 24%** visible behind chairs and desks. A figure whose second
   reading is a box around an object you cannot see does not make the point. I did
   not lower the threshold to force it.

   So the hero is now **"the office chair to the right of the monitor"**
   (`0b031f3119`, anchor `monitor#34`, front confidence 0.54 from thin-face plus
   away-from-wall). It is the closest composable query to the post's opening line:
   a screen-shaped anchor whose facing any reader can see, a lateral relation, and
   the two readings are exact mirror images of each other — which is the
   handedness point in one picture. Frame selection is no longer hand-picked:
   `best_poster_view` scores anchor visibility, candidate separation,
   non-overlap and crop slack, and only about a dozen of the frame-split queries
   in the benchmark pass it.

Two of your other calls were also propagated, since leaving them only in this
file would have recreated the contradiction problem:

* **The pooled 42.9% / 25.7% is no longer a headline anywhere.** The harness now
  computes the stratification you described (`stratify_by_baseline`), the report
  prints it as the row to read — 12 switched, **2 frame-blind**, of the 16 pairs
  where the model and the resolver agree on the unmarked sentence — and marks the
  pooled figures not-quotable. There are tests for it.
* **The lateral-vs-frontal asymmetry is retracted** in the README and the post
  draft, with the reason stated: it tracks baseline agreement (72% vs 18%) and
  the conditioned frontal subset is n=3.
