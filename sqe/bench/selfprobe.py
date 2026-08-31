"""Export the minimal-pair trials so something without an API can answer them.

The point of the minimal-pair design is that the two arms of a pair differ only
in an explicit marker of whose "left" is meant. That design is destroyed the
moment the answerer sees both arms: given the pair side by side, anyone -- model
or person -- can see that a contrast is being tested and reason about it, and the
result stops being evidence about how they read a single sentence.

So the export **blocks** the trials:

* block ``a`` -- the first arm of every pair
* block ``b`` -- the second arm of every pair
* block ``n`` -- the uncued sentence of every pair
* block ``c1``/``c2`` -- the two paraphrases of every control

No block contains two sentences drawn from the same pair or the same control.
Each block goes to a separate, isolated answerer. Sentences are shuffled within a
block, trials are named by an opaque hash of the prompt, and the block file
contains nothing else -- no frame name, no gold answer, no pair id. The mapping
back to pairs lives in a separate keymap the answerer never sees.

The answerer writes ``{"answers": {"<trial>": <id or null>}}``. `FileProbe` in
`frame_probe` then scores it through exactly the same path as an API model.

One honest caveat, recorded here because it belongs with the instrument: if the
answerer is the same model that wrote this stimulus, blocking removes the
side-by-side comparison but not the fact that the author knew what was being
tested. That run is a pilot, not an independent measurement, and the report says
so wherever the number appears.
"""

from __future__ import annotations

import json
import os
import random
from typing import Callable, Dict, List, Optional, Sequence

from .frame_probe import build_prompt, prompt_key

INSTRUCTIONS = """\
# Trial block: which object does the sentence refer to?

Each trial gives you a list of objects in a room and one sentence. Answer with
the id of the single object the sentence refers to.

Rules, and they matter:

* Answer **only** from what the trial text gives you. Do not open any other
  file, do not search the repository, do not look for a key or a gold answer.
* Answer each trial on its own. Do not compare trials to each other or try to
  work out what the set is testing.
* Answer every trial. If genuinely no object fits, answer `null` -- but prefer a
  best guess, since `null` is scored as a non-answer rather than as a mistake.

Write one JSON object to the output path:

    {"answers": {"<trial>": <id or null>, ...}}

Nothing else in the file.
"""


def export_trials(pairs: Sequence, controls: Sequence, scene_for: Callable,
                  out_dir: str, shuffle_ids: bool = True, seed: int = 0,
                  block_seed: int = 7,
                  skip_keys: Optional[set] = None) -> Dict:
    """Write blocked trial files plus a private keymap. Returns a manifest.

    `skip_keys` leaves already-answered trials out of the block files while
    keeping them in the keymap. A trial's key is a hash of its prompt, so fixing
    a prompt bug changes the keys of the affected trials only, and a top-up run
    can re-ask just those instead of the whole set.
    """
    skip_keys = skip_keys or set()
    trials_dir = os.path.join(out_dir, "trials")
    os.makedirs(trials_dir, exist_ok=True)
    blocks: Dict[str, List[dict]] = {k: [] for k in ("a", "b", "n", "c1", "c2")}
    keymap: Dict[str, dict] = {}

    def add(block: str, scene, text: str, item, meta: dict):
        prompt = build_prompt(scene, text, item, shuffle_ids, seed)
        key = prompt_key(prompt)
        blocks[block].append({"trial": key, "prompt": prompt})
        keymap[key] = dict(meta, text=text, block=block,
                           scene_id=item.scene_id)

    for p in pairs:
        scene = scene_for(p.scene_id)
        for slot, arm in zip(("a", "b"), p.arms):
            add(slot, scene, arm.text, p,
                {"kind": "pair_arm", "pair_id": p.id, "frame": arm.frame,
                 "gold": arm.answer_id})
        add("n", scene, p.neutral_text, p,
            {"kind": "pair_neutral", "pair_id": p.id,
             "frame_chosen_by_resolver": p.neutral_frame_chosen})
    for c in controls:
        scene = scene_for(c.scene_id)
        for slot, text in zip(("c1", "c2"), c.texts):
            add(slot, scene, text, c,
                {"kind": "control", "control_id": c.id,
                 "expected": c.expected_answer_id})

    rng = random.Random(block_seed)
    manifest = {"n_trials": len(keymap), "blocks": {}, "seed": seed,
                "block_seed": block_seed, "n_skipped": 0}
    for name, rows in blocks.items():
        before = len(rows)
        rows = [r for r in rows if r["trial"] not in skip_keys]
        manifest["n_skipped"] += before - len(rows)
        rng.shuffle(rows)
        path = os.path.join(trials_dir, f"block_{name}.json")
        with open(path, "w") as f:
            json.dump({"block": name, "n": len(rows),
                       "instructions": INSTRUCTIONS, "trials": rows}, f,
                      indent=1)
        manifest["blocks"][name] = {"path": path, "n": len(rows)}

    with open(os.path.join(trials_dir, "INSTRUCTIONS.md"), "w") as f:
        f.write(INSTRUCTIONS)
    # the keymap is the answer key: it lives OUTSIDE the trials directory the
    # answerer is pointed at.
    with open(os.path.join(out_dir, "keymap_private.json"), "w") as f:
        json.dump(keymap, f, indent=1)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    # a blocking violation would silently invalidate the whole run, so check
    seen: Dict[str, set] = {}
    for key, m in keymap.items():
        gid = m.get("pair_id") or m.get("control_id")
        seen.setdefault(m["block"], set())
        assert gid not in seen[m["block"]], \
            f"blocking violated: {gid} appears twice in block {m['block']}"
        seen[m["block"]].add(gid)
    return manifest


def merge_answers(paths: Sequence[str], out_path: str) -> Dict:
    """Combine per-block answer files into one, checking for collisions."""
    merged: Dict[str, Optional[int]] = {}
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        rows = d.get("answers", d)
        for k, v in rows.items():
            if isinstance(v, dict):
                v = v.get("id")
            if k in merged:
                raise ValueError(f"trial {k} answered in two blocks")
            merged[k] = None if v in (None, "null") else int(v)
    with open(out_path, "w") as f:
        json.dump({"answers": merged}, f, indent=1)
    return merged


def audit_answers(answers: Dict[str, Optional[int]], keymap_path: str) -> Dict:
    """What is missing, and does the coverage let the pairs be scored?"""
    with open(keymap_path) as f:
        keymap = json.load(f)
    missing = [k for k in keymap if k not in answers]
    extra = [k for k in answers if k not in keymap]
    by_block: Dict[str, Dict[str, int]] = {}
    for k, m in keymap.items():
        b = by_block.setdefault(m["block"], {"n": 0, "answered": 0,
                                             "null": 0})
        b["n"] += 1
        if k in answers:
            b["answered"] += 1
            if answers[k] is None:
                b["null"] += 1
    return {"n_expected": len(keymap), "n_answered": len(answers),
            "missing": missing, "extra": extra, "by_block": by_block}
