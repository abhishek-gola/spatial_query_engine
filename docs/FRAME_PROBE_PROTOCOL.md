# Frame probe: what the experiment is, and what it is not

The question is narrow and answerable: **if you tell a system which reference
frame you mean, does its answer change?**

Not "can it do spatial reasoning". Not "does it know left from right". Just
whether the frame is a thing the system represents at all, such that an explicit
instruction can move it. A system that cannot be instructed into a frame will
silently apply one convention to every utterance, and every benchmark it is
scored on will reward exactly one reading of a sentence that has two.

## The stimulus

35 minimal pairs on four ScanNet++ scenes. Each pair is two sentences that
differ **only** in an explicit marker of whose "left" is meant:

| arm | sentence |
|---|---|
| egocentric | *from where I am standing, the cabinet to the left of the bed* |
| intrinsic | *the cabinet to the bed's own left* |

plus a third, uncued sentence — *the cabinet to the left of the bed* — which
records which convention the system falls back on when nothing is stated.

A pair is only generated when all of the following hold. Every one of these cuts
yield, and the cuts are the reason there are 35 pairs rather than 165:

* the anchor's class occurs **exactly once** in the scene, so the sentence
  identifies the anchor without an id;
* the target's class occurs **at least twice**, so the frame has something to
  choose between;
* the anchor has an estimated front with confidence ≥ 0.25, so the intrinsic
  reading exists at all;
* both arms parse to the same relation, target and anchor, and to the same
  viewpoint — the arms must differ in the frame and in nothing else;
* the two readings resolve to **different objects**, both above the answer-score
  threshold, both instances of the target class.

Each pair also records a **concrete observer position**, not a rule for finding
one. This matters more than it sounds: the egocentric gold answer is computed
from some viewpoint, and a system handed only the scene listing has no way to
know where that was. A pair that stored `viewpoint: best_view` would be marking
the egocentric arm against a position the answerer was never told.

## The two controls, which are the whole reason to believe any of it

**Positive control for the label.** The same resolver with its frame *pinned*,
ignoring the cue. It provably cannot switch, so it must score 100%
`frame_blind`. If it does not, the label is picking up something other than the
frame and no row in the table means anything. All four pinned frames score
100%.

**Circularity check.** The resolver *following* the cue must score ~100%
`switched_correctly` — the pairs were built by forcing its own frames, so this
only says the stimulus is well-formed and the scoring works. It is not evidence
that the resolver is right about anything, and it is labelled as such everywhere
it appears.

**Negative control for stability.** 35 non-contrastive pairs: two paraphrases,
equally awkward, matched for clause shape, differing in nothing that should
change the answer.

> *as far as I can tell, the cabinet to the left of the bed*
> *the cabinet to the left of the bed, if I am not mistaken*

A system **should** answer these identically. `frame_blind` only means "has no
frame to instruct" if control stability is high; a system that is merely unstable
to surface form would also give two different answers to the two cued arms, for
reasons that have nothing to do with reference frames. This is the check that
separates a finding from an artefact.

## Blocking: why the trials are split across separate answerers

A minimal pair measures nothing if the answerer sees both arms. Side by side,
anyone — model or person — can see that a contrast is being tested and reason
about the contrast rather than reading the sentence. So the 175 trials are
partitioned into five blocks:

| block | contents |
|---|---|
| `a` | the first arm of every pair |
| `b` | the second arm of every pair |
| `n` | the uncued sentence of every pair |
| `c1`, `c2` | the two paraphrases of every control |

No block holds two sentences drawn from the same pair or the same control, and
each block goes to a separate answerer with no shared context. Sentences are
shuffled within a block, trials are named by an opaque hash of the prompt, and
the block file contains nothing else: no frame name, no gold answer, no pair id.
The answer key is written outside the directory the answerer is given.

`tests/test_selfprobe.py` asserts the blocking rather than trusting it — an
export that put both arms in one block would still produce a clean-looking table,
and the table would be worthless.

## Running it

Against an API model, one command per vendor:

```bash
export ANTHROPIC_API_KEY=...     # or OPENAI_API_KEY / GEMINI_API_KEY
sqe frame-probe --root DATA --model frontier --preflight   # reachability first
sqe frame-probe --root DATA --model anthropic:claude-opus-5 openai:gpt-5
```

Against anything without an API reachable from here — including a person — export
the blocked trials, answer them elsewhere, and fold them back in:

```bash
sqe self-probe export --root DATA --out results/self_probe
# ... each block answered in isolation, written to results/self_probe/answers/
sqe self-probe merge --out results/self_probe
sqe frame-probe --root DATA --answers "some system=results/self_probe/answers_merged.json"
```

Both paths go through the same `run()` and the same `classify_pair()`. Nothing in
the scoring knows where the answers came from.

## What the self-administered run is worth, stated plainly

One arm of the results was collected by giving each block to an isolated Claude
Code agent — a fresh context, one block file, no access to the key, told to
answer from the prompt alone and not to compute the relation with a script.

Blocking removes the side-by-side comparison. It does **not** remove the fact
that the author of the stimulus and the answerer are the same model family, and
that the author knew what was being tested. So:

* this arm is a **pilot**, not an independent measurement, and it is labelled
  that way wherever the number appears;
* the discipline (no scripts, no cross-trial reasoning, no reading other files)
  is *instructed*, not enforced by the sandbox;
* an independent API run on models from other vendors is the version of this
  experiment that would carry weight, and the harness for it is built and waiting
  on a key.

One thing the run did establish beyond doubt is that the answerers were working
from the prompt: two of them, independently and with no shared context, reported
that four trials named an anchor — a heater, a wall clock — that was absent from
the object list. They were right. The listing filter dropped room-fixed objects,
and some anchors are room-fixed, so those four trials were unanswerable rather
than hard. An answerer with any view of the key would have had no reason to
notice, and no reason to say so. The bug is fixed
(`build_prompt` now re-inserts the anchor and every candidate), there is a test
for it, and the affected trials were re-asked.
