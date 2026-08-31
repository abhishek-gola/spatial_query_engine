"""Command line entry point.

    sqe scenes    --root DATA                  what is available
    sqe build     --root DATA --scene ID       build and cache a scene graph
    sqe query     --scene ID "the second mug from the left"
    sqe frames    --scene ID --anchor N        show every frame around an anchor
    sqe propose   --scene ID --out FILE        propose benchmark queries
    sqe annotate  --items FILE                 annotate them (blind by default)
    sqe evaluate  --items FILE --out DIR       run the benchmark and report
    sqe sensitivity --items FILE               how much the frame matters
    sqe render    --scene ID --root DATA "..."  boxes drawn on real frames
    sqe audit     --scene ID                   flag dubious annotations
    sqe viewer    --scene ID                   web viewer
    sqe selftest                               synthetic end-to-end check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

from . import cache as cache_mod
from .relations.base import RelationConfig


def _add_common(p):
    p.add_argument("--cache", default=cache_mod.DEFAULT_CACHE,
                   help="scene cache directory")
    p.add_argument("--dataset", default="scannetpp",
                   choices=["scannetpp", "synthetic", "arkitscenes"])
    p.add_argument("--tag", default="", help="cache tag, e.g. a perception mode")


def _scene_getter(args):
    """Load scenes from cache, building on demand if a data root is given."""
    from .pipeline import build
    loaded: Dict[str, object] = {}

    def get(scene_id: str):
        if scene_id in loaded:
            return loaded[scene_id]
        tag = args.tag or (getattr(args, "perception", "gt")
                           if getattr(args, "perception", "gt") != "gt" else "")
        if cache_mod.exists(args.cache, args.dataset, scene_id, tag):
            s = cache_mod.load(args.cache, args.dataset, scene_id, tag)
        else:
            root = getattr(args, "root", None)
            if args.dataset != "synthetic" and not root:
                raise FileNotFoundError(
                    f"scene {scene_id!r} is not in the cache at {args.cache} "
                    f"and no --root was given to build it from")
            s = build(args.dataset, root or "", scene_id,
                      perception=getattr(args, "perception", "gt"),
                      use_gt_fronts=getattr(args, "gt_fronts", False),
                      forward_convention=getattr(args, "forward", "composite"),
                      verbose=not getattr(args, "quiet", False))
            cache_mod.save(s, args.cache, tag)
        loaded[scene_id] = s
        return s
    return get


# --------------------------------------------------------------------------

def cmd_scenes(args):
    print(f"cache: {args.cache}")
    rows = cache_mod.list_cached(args.cache)
    if rows:
        for r in rows:
            stale = "  [STALE, rebuild]" if r["stale"] else ""
            print(f"  {r['key']:44s} {r['n_objects']:4d} objects{stale}")
    else:
        print("  (empty)")
    if args.root:
        if args.dataset == "scannetpp":
            from .data.scannetpp import ScanNetPPPaths, list_scenes
            ids = list_scenes(args.root)
            print(f"\n{args.dataset} at {args.root}: {len(ids)} usable scenes")
            for i in ids:
                p = ScanNetPPPaths(args.root, i)
                extras = []
                if os.path.exists(p.iphone_depth):
                    extras.append("iphone+depth")
                if os.path.exists(p.dslr_transforms):
                    extras.append("dslr")
                print(f"  {i}   {', '.join(extras)}")
        elif args.dataset == "arkitscenes":
            from .data.arkitscenes import list_scenes
            for i in list_scenes(args.root):
                print(f"  {i}")
    if args.dataset == "synthetic":
        from .data.synthetic import ROOMS, SCENE_IDS
        print(f"\nsynthetic rooms: {', '.join(sorted(ROOMS))}"
              f"   (scene ids: {', '.join(sorted(SCENE_IDS))})")
    return 0


def cmd_build(args):
    from .pipeline import build
    from .data.scannetpp import list_scenes as spp_scenes
    ids = args.scene or []
    if args.all:
        if args.dataset == "scannetpp":
            ids = spp_scenes(args.root)
        elif args.dataset == "synthetic":
            from .data.synthetic import ROOMS
            ids = sorted(ROOMS)
        else:
            from .data.arkitscenes import list_scenes
            ids = list_scenes(args.root)
    if not ids:
        print("nothing to build: give --scene ID or --all")
        return 2
    tag = args.tag or (args.perception if args.perception != "gt" else "")
    for sid in ids:
        if cache_mod.exists(args.cache, args.dataset, sid, tag) and not args.force:
            print(f"[skip] {sid} is already cached (use --force to rebuild)")
            continue
        print(f"\n=== building {sid} ({args.dataset}, perception={args.perception})")
        s = build(args.dataset, args.root or "", sid,
                  perception=args.perception, use_gt_fronts=args.gt_fronts,
                  forward_convention=args.forward, verbose=True)
        d = cache_mod.save(s, args.cache, tag)
        print(s.summary())
        print(f"cached -> {d}")
    return 0


def cmd_query(args):
    from .frames.policy import ViewpointSpec
    from .query.parser_rules import parse
    from .query.resolver import Resolver
    get = _scene_getter(args)
    scene = get(args.scene)
    vp = ViewpointSpec(mode=args.viewpoint, index=args.frame_index,
                      landmark=args.landmark,
                      position=(np.asarray(args.position, float)
                                if args.position else None))
    r = Resolver(scene, RelationConfig.load(args.config),
                 predicted=(args.perception != "gt"))
    for text in args.text:
        q = parse(text)
        res = r.resolve(q, vp, force_frame=args.force_frame)
        print()
        print(res.explain())
        if args.json:
            print(json.dumps(res.to_dict(), indent=1, default=float))
    return 0


def cmd_frames(args):
    from .frames.policy import ViewpointSpec, build_frames, decide_frame
    get = _scene_getter(args)
    scene = get(args.scene)
    anchor = scene.by_id(args.anchor)
    if anchor is None:
        print(f"no object #{args.anchor} in {args.scene}")
        return 2
    vp = ViewpointSpec(mode=args.viewpoint, index=args.frame_index,
                       landmark=args.landmark,
                       position=(np.asarray(args.position, float)
                                 if args.position else None))
    frames, rv = build_frames(scene, anchor, viewpoint=vp)
    print(f"anchor {anchor.short()}")
    print(f"  front: {'none' if anchor.front is None else np.round(anchor.front, 3)}"
          f"  confidence {anchor.front_confidence:.2f} [{anchor.front_method}]")
    print(f"  viewpoint: {rv.source}  ok={rv.ok} {rv.reason}")
    print()
    for kind, f in frames.items():
        hand = "right-handed" if f.handedness > 0 else "LEFT-handed"
        if f.available:
            print(f"  {kind:20s} right=({f.right[0]:+.2f},{f.right[1]:+.2f})  "
                  f"front=({f.front[0]:+.2f},{f.front[1]:+.2f})  "
                  f"conf {f.confidence:.2f}  {hand}")
        else:
            print(f"  {kind:20s} UNAVAILABLE: {f.reason}")
    print()
    for rel in ("left", "right", "front", "behind"):
        d = decide_frame(rel, anchor, scene, "", vp)
        print(f"  policy for {rel:7s} -> {d.chosen}"
              f"   ({'; '.join(p.kind + ':' + f'{p.prior:.2f}' for p in d.priors)})")
    return 0


def cmd_propose(args):
    from .bench.generate import propose_scene, to_items
    from .bench.schema import describe_split, write_jsonl
    from .frames.policy import ViewpointSpec
    get = _scene_getter(args)
    vp = ViewpointSpec(mode=args.viewpoint, index=args.frame_index,
                       landmark=args.landmark)
    items = []
    for sid in args.scene:
        scene = get(sid)
        props = propose_scene(scene, RelationConfig.load(args.config), vp,
                              args.max_projective, args.max_ordinal,
                              args.max_controls)
        got = to_items(scene, props, viewpoint=vp)
        print(f"{sid}: {len(got)} proposals "
              f"({sum(1 for p in props if p.frame_sensitive)} frame-sensitive)")
        items.extend(got)
    write_jsonl(items, args.out)
    print(f"\nwrote {len(items)} unannotated items -> {args.out}")
    print(json.dumps(describe_split(items), indent=1))
    print("\nNext: sqe annotate --items " + args.out)
    return 0


def cmd_annotate(args):
    from .bench.annotate import annotate
    from .bench.schema import read_jsonl
    from .query.resolver import Resolver
    get = _scene_getter(args)
    items = read_jsonl(args.items)
    resolvers: Dict[str, Resolver] = {}

    def resolver_for(sid):
        if sid not in resolvers:
            resolvers[sid] = Resolver(get(sid), RelationConfig.load(args.config))
        return resolvers[sid]

    annotate(items, get, args.out or args.items, annotator=args.annotator,
             show_prediction=args.show_prediction, resolver_for=resolver_for,
             start=args.start, only_unannotated=not args.all)
    return 0


def cmd_validate(args):
    from .bench.schema import describe_split, read_many
    get = _scene_getter(args)
    items = read_many(args.items)
    bad = 0
    for it in items:
        try:
            scene = get(it.scene_id)
        except Exception as exc:
            print(f"{it.id}: scene unavailable ({exc})")
            bad += 1
            continue
        problems = it.validate(scene)
        if problems:
            bad += 1
            print(f"{it.id}: " + "; ".join(problems))
    print(f"\n{len(items)} items, {bad} with problems")
    print(json.dumps(describe_split(items), indent=1))
    return 1 if bad else 0


def cmd_evaluate(args):
    from .bench.evaluate import (aggregate, attribute, render_report,
                                 run_condition, save_results)
    from .bench.schema import describe_split, read_many
    from .pipeline import build

    items = read_many(args.items)
    items = [i for i in items if i.target_ids or i.ambiguous]
    if not items:
        print("no annotated items found; run `sqe annotate` first")
        return 2
    if args.limit:
        items = items[: args.limit]
    split = describe_split(items)
    print(f"evaluating {len(items)} annotated items over "
          f"{split['n_scenes']} scenes")

    cfg = RelationConfig.load(args.config)
    gt_get = _scene_getter(argparse.Namespace(**{**vars(args),
                                                 "perception": "gt", "tag": ""}))
    scene_getters = {"gt": gt_get}
    if args.perception == "openvocab" or args.compare_perception:
        ov_args = argparse.Namespace(**{**vars(args), "perception": "openvocab",
                                        "tag": "openvocab"})
        scene_getters["openvocab"] = _scene_getter(ov_args)

    primary_perception = args.perception
    primary_get = scene_getters[primary_perception]

    conditions_out: Dict[str, Dict] = {}
    outcomes: Dict[str, List] = {}

    def run(name, getter, parse_mode, frame_mode, predicted):
        oc = run_condition(items, getter, parse_mode, frame_mode, cfg,
                           predicted_perception=predicted,
                           progress=not args.quiet)
        outcomes[name] = oc
        conditions_out[name] = aggregate(oc)
        return oc

    base = run("ours (policy frame)", primary_get, args.parse, "policy",
               primary_perception != "gt")
    for fk in args.baselines:
        run(f"fixed frame: {fk}", primary_get, args.parse, fk,
            primary_perception != "gt")
    oracle = run("oracle frame", primary_get, args.parse, "oracle",
                 primary_perception != "gt")
    gold = None
    if any(i.gold_parse for i in items):
        gold = run("gold parse", primary_get, "gold", "policy",
                   primary_perception != "gt")
    gtp = None
    if primary_perception != "gt":
        gtp = run("ground-truth perception", scene_getters["gt"], args.parse,
                  "policy", False)

    # which items even have a constructible annotated frame
    frame_available: Dict[str, bool] = {}
    from .frames.policy import build_frames
    for it in items:
        if it.frame in ("unspecified", "any", "world"):
            frame_available[it.id] = True
            continue
        try:
            scene = primary_get(it.scene_id)
        except Exception:
            frame_available[it.id] = True
            continue
        ok = True
        gq = it.gold_query()
        anchors = (gq.target.constraints[0].anchors if gq
                   and gq.target.constraints else [])
        if anchors and anchors[0].label:
            hits = scene.find(anchors[0].label)
            if hits:
                fr, _ = build_frames(scene, hits[0], (it.frame,),
                                     it.viewpoint_spec())
                ok = fr[it.frame].available
        frame_available[it.id] = ok

    conds = {}
    if gold is not None:
        conds["gold_parse"] = gold
    if gtp is not None:
        conds["gt_perception"] = gtp
    conds["oracle_frame"] = oracle
    attribution = attribute(base, conds, frame_available)

    report = render_report(split, conditions_out, attribution,
                           title=args.title)
    print()
    print(report)
    if args.out:
        save_results(args.out, split, conditions_out, attribution, outcomes,
                     report, meta={"items": args.items,
                                   "perception": primary_perception,
                                   "parse": args.parse,
                                   "config": args.config or "defaults"})
        print(f"\nwrote {args.out}/report.md, results.json, outcomes.jsonl")
    return 0


def cmd_audit(args):
    from .data.quality import audit_scene, format_audit
    get = _scene_getter(args)
    ids = args.scene
    if args.all:
        ids = [r["scene_id"] for r in cache_mod.list_cached(args.cache)
               if r["dataset"] == args.dataset]
    total_s = total_o = 0
    for sid in ids:
        scene = get(sid)
        a = audit_scene(scene)
        total_s += a["n_suspect"]
        total_o += a["n_objects"]
        print(format_audit(scene, max_rows=args.max_rows))
        print()
    if total_o:
        print(f"TOTAL {total_s} of {total_o} instances flagged "
              f"({100.0 * total_s / total_o:.1f}%)")
        print("Flagged instances stay in the scene but are excluded from "
              "benchmark proposal generation.")
    return 0


def cmd_render(args):
    """Project the 3-D boxes into real camera frames, to check the geometry."""
    import numpy as np
    from .viz.overlay import (render_pointcloud_view, render_query_overlay,
                              render_scene_frames, render_topdown,
                              scannetpp_frame_source)
    get = _scene_getter(args)
    os.makedirs(args.out, exist_ok=True)
    written = []
    for sid in args.scene:
        scene = get(sid)
        out_dir = os.path.join(args.out, sid)
        src = None
        if args.dataset == "scannetpp" and args.root:
            try:
                src = scannetpp_frame_source(args.root, sid)
            except Exception as exc:
                print(f"[render] no frame source for {sid}: {exc}")

        if args.text:
            from .frames.policy import ViewpointSpec
            from .query.parser_rules import parse
            from .query.resolver import Resolver
            r = Resolver(scene, RelationConfig.load(args.config))
            vp = ViewpointSpec(mode=args.viewpoint, index=args.frame_index,
                               landmark=args.landmark,
                               position=(np.asarray(args.position, float)
                                         if args.position else None))
            for n, text in enumerate(args.text):
                res = r.resolve(parse(text), vp)
                print()
                print(res.explain())
                hl = {}
                if res.target is not None:
                    hl[res.target.id] = "target"
                for a in res.anchors:
                    if a.obj is not None:
                        hl[a.obj.id] = "anchor"
                if len(res.candidates) > 1:
                    hl.setdefault(res.candidates[1].obj.id, "runner")
                slug = "".join(c if c.isalnum() else "_" for c in text)[:48]
                if src is not None and len(src):
                    p = render_query_overlay(
                        scene, src, res,
                        os.path.join(out_dir, f"q{n:02d}_{slug}.jpg"),
                        scale=args.scale, hidden_style=args.hidden_style,
                        max_distance=args.max_distance)
                    if p:
                        written.append(p)
                fr = (res.frame_decision.frame
                      if res.frame_decision is not None else None)
                eye = (res.frame_decision.viewpoint.eye
                       if res.frame_decision is not None else None)
                written.append(render_topdown(
                    scene, os.path.join(out_dir, f"q{n:02d}_{slug}_topdown.png"),
                    hl, viewpoint=eye, frame=fr,
                    caption=[f'"{text}"',
                             f"answer: "
                             f"{res.target.label if res.target else 'none'}"
                             f"   frame: {res.frame_used or 'frame-free'}"]))
                if src is None or not len(src):
                    written.append(render_pointcloud_view(
                        scene, os.path.join(out_dir, f"q{n:02d}_{slug}_view.png"),
                        highlight=hl,
                        caption=[f'"{text}"']))
        else:
            if src is not None and len(src):
                written += render_scene_frames(
                    scene, src, out_dir, n_frames=args.frames,
                    include_structure=args.structure, scale=args.scale,
                    hidden_style=args.hidden_style,
                    max_distance=args.max_distance,
                    max_objects=(args.max_objects or None))
            else:
                written.append(render_pointcloud_view(
                    scene, os.path.join(out_dir, "view.png")))
            written.append(render_topdown(
                scene, os.path.join(out_dir, "topdown.png")))

        if args.per_object and src is not None and len(src):
            from .scenegraph.objects import CameraTrajectory
            traj = CameraTrajectory(src.poses, src.Ks, src.image_size, src.names)
            labels = args.per_object
            picks = []
            for lab in labels:
                hits = scene.find(lab)
                if not hits:
                    print(f"[render] no {lab!r} in {sid}")
                    continue
                o = max(hits, key=lambda x: x.obb.volume)
                picks.append((lab, o, int(traj.best_view(o.obb, scene.up))))
            frames = src.rgb_many([i for _, _, i in picks])
            import cv2
            from .viz.overlay import render_frame_overlay
            for lab, o, i in picks:
                rgb = frames.get(i)
                if rgb is None:
                    continue
                img = render_frame_overlay(
                    scene, src, i, {o.id: "target"}, rgb,
                    max_distance=args.max_distance,
                    caption=[f"best view of {o.label} #{o.id} (amber), "
                             f"frame {i}",
                             "the amber box should sit on the object",
                             "solid = visible, faint dashes = occluded"],
                    scale=args.scale, hidden_style=args.hidden_style,
                    label_all=True,
                    max_objects=(args.max_objects or None))
                p = os.path.join(out_dir,
                                 f"best_{lab.replace(' ', '_')}.jpg")
                cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                written.append(p)

    print()
    print(f"wrote {len(written)} images:")
    for p in written:
        print(f"  {p}")
    return 0


def cmd_sensitivity(args):
    """How much the answer depends on the frame. Needs no annotation."""
    import json as _json
    from .bench.schema import read_many
    from .bench.sensitivity import measure, render, summarise
    get = _scene_getter(args)
    items = read_many(args.items)
    if args.limit:
        items = items[: args.limit]
    rows = measure(items, get, RelationConfig.load(args.config),
                   progress=not args.quiet)
    summary = summarise(rows)
    text = render(summary, args.title)
    print()
    print(text)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "sensitivity.md"), "w") as f:
            f.write(text)
        with open(os.path.join(args.out, "sensitivity.json"), "w") as f:
            _json.dump(summary, f, indent=1, default=float)
        print(f"\nwrote {args.out}/sensitivity.md and sensitivity.json")
    return 0


def cmd_viewer(args):
    from .viewer.server import serve
    get = _scene_getter(args)
    scenes = {sid: get(sid) for sid in args.scene}
    serve(scenes, host=args.host, port=args.port,
          cfg=RelationConfig.load(args.config),
          items_path=args.items, open_browser=not args.no_browser)
    return 0


def cmd_selftest(args):
    from .selftest import run_selftest
    return run_selftest(verbose=not args.quiet)


# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="sqe", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--config", default=None, help="relations.yaml override")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("scenes", help="list cached and available scenes")
    _add_common(p)
    p.add_argument("--root", default=None)
    p.set_defaults(func=cmd_scenes)

    p = sub.add_parser("build", help="build and cache scene graphs")
    _add_common(p)
    p.add_argument("--root", default=None)
    p.add_argument("--scene", nargs="*", default=[])
    p.add_argument("--all", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--perception", default="gt", choices=["gt", "openvocab"])
    p.add_argument("--gt-fronts", action="store_true",
                   help="take object fronts from ground truth (oracle)")
    p.add_argument("--forward", default="composite",
                   help="room canonical-forward convention")
    p.set_defaults(func=cmd_build)

    def _vp(p):
        p.add_argument("--viewpoint", default="best_view",
                       choices=["best_view", "nearest", "mean", "index",
                                "position", "landmark"])
        p.add_argument("--frame-index", type=int, default=None)
        p.add_argument("--landmark", default=None)
        p.add_argument("--position", nargs=3, type=float, default=None)

    p = sub.add_parser("query", help="resolve queries against a scene")
    _add_common(p)
    p.add_argument("--root", default=None)
    p.add_argument("--scene", required=True)
    p.add_argument("text", nargs="+")
    p.add_argument("--perception", default="gt", choices=["gt", "openvocab"])
    p.add_argument("--force-frame", default=None)
    p.add_argument("--json", action="store_true")
    p.add_argument("--gt-fronts", action="store_true")
    p.add_argument("--forward", default="composite")
    _vp(p)
    p.set_defaults(func=cmd_query)

    p = sub.add_parser("frames", help="show all reference frames for an anchor")
    _add_common(p)
    p.add_argument("--root", default=None)
    p.add_argument("--scene", required=True)
    p.add_argument("--anchor", type=int, required=True)
    p.add_argument("--perception", default="gt", choices=["gt", "openvocab"])
    p.add_argument("--gt-fronts", action="store_true")
    p.add_argument("--forward", default="composite")
    _vp(p)
    p.set_defaults(func=cmd_frames)

    p = sub.add_parser("propose", help="propose benchmark queries")
    _add_common(p)
    p.add_argument("--root", default=None)
    p.add_argument("--scene", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--perception", default="gt", choices=["gt", "openvocab"])
    p.add_argument("--gt-fronts", action="store_true")
    p.add_argument("--forward", default="composite")
    p.add_argument("--max-projective", type=int, default=25,
                   help="per relation, so x4 for left/right/front/behind")
    p.add_argument("--max-ordinal", type=int, default=30)
    p.add_argument("--max-controls", type=int, default=70)
    _vp(p)
    p.set_defaults(func=cmd_propose)

    p = sub.add_parser("annotate", help="annotate benchmark items")
    _add_common(p)
    p.add_argument("--root", default=None)
    p.add_argument("--items", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--annotator", default=os.environ.get("USER", ""))
    p.add_argument("--show-prediction", action="store_true",
                   help="NOT blind; stamps the items so the report can say so")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--all", action="store_true",
                   help="revisit already-annotated items too")
    p.add_argument("--perception", default="gt", choices=["gt", "openvocab"])
    p.add_argument("--gt-fronts", action="store_true")
    p.add_argument("--forward", default="composite")
    p.set_defaults(func=cmd_annotate)

    p = sub.add_parser("validate", help="check benchmark items")
    _add_common(p)
    p.add_argument("--root", default=None)
    p.add_argument("--items", nargs="+", required=True)
    p.add_argument("--perception", default="gt", choices=["gt", "openvocab"])
    p.add_argument("--gt-fronts", action="store_true")
    p.add_argument("--forward", default="composite")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("evaluate", help="run the benchmark and write a report")
    _add_common(p)
    p.add_argument("--root", default=None)
    p.add_argument("--items", nargs="+", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--perception", default="gt", choices=["gt", "openvocab"])
    p.add_argument("--compare-perception", action="store_true",
                   help="also build the other perception mode for attribution")
    p.add_argument("--parse", default="rules", choices=["rules", "gold", "llm"])
    p.add_argument("--baselines", nargs="*",
                   default=["egocentric", "intrinsic", "world"],
                   help="fixed-frame baselines to report")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--title", default="Spatial query benchmark")
    p.add_argument("--gt-fronts", action="store_true")
    p.add_argument("--forward", default="composite")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("sensitivity",
                       help="how much the answer depends on the frame "
                            "(needs no annotation)")
    _add_common(p)
    p.add_argument("--root", default=None)
    p.add_argument("--items", nargs="+", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--title", default="Frame sensitivity")
    p.add_argument("--perception", default="gt", choices=["gt", "openvocab"])
    p.add_argument("--gt-fronts", action="store_true")
    p.add_argument("--forward", default="composite")
    p.set_defaults(func=cmd_sensitivity)

    p = sub.add_parser("audit", help="flag dubious ground-truth annotations")
    _add_common(p)
    p.add_argument("--root", default=None)
    p.add_argument("--scene", nargs="*", default=[])
    p.add_argument("--all", action="store_true")
    p.add_argument("--max-rows", type=int, default=20)
    p.add_argument("--perception", default="gt", choices=["gt", "openvocab"])
    p.add_argument("--gt-fronts", action="store_true")
    p.add_argument("--forward", default="composite")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("render",
                       help="project 3D boxes into real frames to verify the "
                            "geometry")
    _add_common(p)
    p.add_argument("--root", default=None,
                   help="dataset root; needed for RGB frames")
    p.add_argument("--scene", nargs="+", required=True)
    p.add_argument("--out", default="renders")
    p.add_argument("text", nargs="*",
                   help="queries to render; omit for a scene overview")
    p.add_argument("--frames", type=int, default=6)
    p.add_argument("--per-object", nargs="*", default=None,
                   metavar="LABEL",
                   help="also render the best view of each of these classes")
    p.add_argument("--structure", action="store_true",
                   help="draw walls, floor and ceiling too")
    p.add_argument("--hidden-style", default="dashed",
                   choices=["dashed", "dim", "hide"],
                   help="how to draw the occluded parts of a box")
    p.add_argument("--max-objects", type=int, default=12,
                   help="draw only the nearest N objects (0 for no limit)")
    p.add_argument("--max-distance", type=float, default=6.0)
    p.add_argument("--scale", type=float, default=0.55)
    p.add_argument("--perception", default="gt", choices=["gt", "openvocab"])
    p.add_argument("--gt-fronts", action="store_true")
    p.add_argument("--forward", default="composite")
    _vp(p)
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("viewer", help="serve the web viewer")
    _add_common(p)
    p.add_argument("--root", default=None)
    p.add_argument("--scene", nargs="+", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--items", default=None,
                   help="benchmark jsonl to annotate from the viewer")
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--perception", default="gt", choices=["gt", "openvocab"])
    p.add_argument("--gt-fronts", action="store_true")
    p.add_argument("--forward", default="composite")
    p.set_defaults(func=cmd_viewer)

    p = sub.add_parser("selftest", help="end-to-end check on synthetic rooms")
    p.set_defaults(func=cmd_selftest)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    try:
        return args.func(args) or 0
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
