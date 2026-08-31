#!/usr/bin/env bash
#
# Build the scene graphs, check the frame logic, propose queries, and -- once
# queries are annotated -- run the benchmark and write the report.
#
#   ./run_benchmark.sh /path/to/scannetpp
#
# The script is idempotent: scenes already cached are skipped, and annotations
# already present are never overwritten. Run it, annotate some queries, run it
# again.
#
# Options
#   --scenes "a b c"     only these scene ids (default: every usable scene)
#   --python PATH        interpreter to use (default: auto-detect)
#   --perception MODE    gt | openvocab   (default: gt)
#   --items FILE         benchmark jsonl (default: benchmark/queries/<name>.jsonl)
#   --out DIR            results directory (default: results/<timestamp>)
#   --cache DIR          scene cache (default: ./cache)
#   --skip-render        do not write the box-overlay verification images
#   --skip-build         assume the cache is already built
#   --skip-propose       do not generate new proposals
#   --propose-only       stop after proposing
#   --rebuild            force a rebuild of cached scenes
#   --setup              create the conda env and install dependencies first
#   --limit N            evaluate only the first N annotated items
#   -h, --help

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

DATA_ROOT=""
SCENES=""
PY=""
PERCEPTION="gt"
ITEMS=""
OUT=""
CACHE="$HERE/cache"
SKIP_BUILD=0
SKIP_PROPOSE=0
SKIP_RENDER=0
PROPOSE_ONLY=0
REBUILD=0
DO_SETUP=0
LIMIT=""

die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
say() { printf '\n\033[1;36m==>\033[0m \033[1m%s\033[0m\n' "$*"; }
note() { printf '    %s\n' "$*"; }

usage() { sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)      usage ;;
    --scenes)       SCENES="${2:-}"; shift 2 ;;
    --python)       PY="${2:-}"; shift 2 ;;
    --perception)   PERCEPTION="${2:-}"; shift 2 ;;
    --items)        ITEMS="${2:-}"; shift 2 ;;
    --out)          OUT="${2:-}"; shift 2 ;;
    --cache)        CACHE="${2:-}"; shift 2 ;;
    --limit)        LIMIT="${2:-}"; shift 2 ;;
    --skip-build)   SKIP_BUILD=1; shift ;;
    --skip-render)  SKIP_RENDER=1; shift ;;
    --skip-propose) SKIP_PROPOSE=1; shift ;;
    --propose-only) PROPOSE_ONLY=1; shift ;;
    --rebuild)      REBUILD=1; shift ;;
    --setup)        DO_SETUP=1; shift ;;
    -*)             die "unknown option $1 (try --help)" ;;
    *)              if [[ -z "$DATA_ROOT" ]]; then DATA_ROOT="$1"; shift;
                    else die "unexpected argument $1"; fi ;;
  esac
done

# ---------------------------------------------------------------- python
if [[ $DO_SETUP -eq 1 ]]; then
  say "creating the conda env 'sqe'"
  command -v conda >/dev/null || die "conda not found; install it or use --python"
  conda create -y -n sqe python=3.11
  # shellcheck disable=SC1091
  eval "$(conda shell.bash hook)"
  conda activate sqe
  pip install -e ".[iphone]"
  PY="$(command -v python)"
fi

if [[ -z "$PY" ]]; then
  for cand in \
      "${CONDA_PREFIX:-}/bin/python" \
      "$HOME/miniconda3/envs/sqe/bin/python" \
      "$HOME/miniforge3/envs/sqe/bin/python" \
      "$HOME/anaconda3/envs/sqe/bin/python" \
      "$(command -v python3 || true)"; do
    [[ -n "$cand" && -x "$cand" ]] || continue
    if "$cand" -c 'import numpy, scipy, plyfile, yaml' >/dev/null 2>&1; then
      PY="$cand"; break
    fi
  done
fi
[[ -n "$PY" ]] || die "no interpreter with numpy+scipy+plyfile+pyyaml found.
    Run: ./run_benchmark.sh --setup /path/to/data
    or:  ./run_benchmark.sh --python /path/to/python /path/to/data"

say "environment"
note "python:  $PY  ($("$PY" -V 2>&1))"
"$PY" - <<'PYCHECK'
import importlib.util, sys
need = ["numpy", "scipy", "plyfile", "yaml"]
opt = {"lz4": "iPhone depth frames", "torch": "open-vocabulary perception",
       "transformers": "open-vocabulary perception"}
missing = [m for m in need if not importlib.util.find_spec(m)]
if missing:
    print("    MISSING REQUIRED: " + ", ".join(missing)); sys.exit(1)
print("    required: all present")
for m, why in opt.items():
    have = importlib.util.find_spec(m) is not None
    print(f"    optional: {m:14s} {'yes' if have else 'no '}   ({why})")
PYCHECK

export SQE_CACHE="$CACHE"
RUN="$PY -m sqe --config $HERE/configs/relations.yaml"

# ---------------------------------------------------------------- selftest
say "self-test (synthetic rooms, hand-derived frame expectations)"
if ! $RUN --quiet selftest; then
  die "the self-test failed. The frame logic is broken; fix that before
    trusting any benchmark number produced from it."
fi

# ---------------------------------------------------------------- data
if [[ $SKIP_BUILD -eq 0 || $SKIP_PROPOSE -eq 0 ]]; then
  [[ -n "$DATA_ROOT" ]] || die "give the dataset folder:
    ./run_benchmark.sh /path/to/scannetpp
    (or pass --skip-build --skip-propose to work purely from the cache)"
  [[ -d "$DATA_ROOT" ]] || die "not a directory: $DATA_ROOT"
fi

if [[ -z "$SCENES" && -n "$DATA_ROOT" ]]; then
  say "discovering scenes in $DATA_ROOT"
  SCENES="$($PY - "$DATA_ROOT" <<'PYSCENES'
import sys
from sqe.data.scannetpp import list_scenes
print(" ".join(list_scenes(sys.argv[1])))
PYSCENES
)"
  [[ -n "$SCENES" ]] || die "no usable ScanNet++ scenes under $DATA_ROOT.
    Each scene needs scans/mesh_aligned_0.05.ply, scans/segments.json
    and scans/segments_anno.json."
fi
note "scenes: $SCENES"
N_SCENES=$(wc -w <<<"$SCENES" | tr -d ' ')

# ---------------------------------------------------------------- build
if [[ $SKIP_BUILD -eq 0 ]]; then
  say "building scene graphs (perception=$PERCEPTION)"
  FORCE=""
  [[ $REBUILD -eq 1 ]] && FORCE="--force"
  # shellcheck disable=SC2086
  $RUN build --root "$DATA_ROOT" --cache "$CACHE" --perception "$PERCEPTION" \
       --scene $SCENES $FORCE
else
  note "skipping build"
fi

say "cached scenes"
$RUN scenes --cache "$CACHE" | sed 's/^/    /'

# ---------------------------------------------------------------- audit
say "auditing the ground-truth annotations"
note "instances whose fitted box disagrees with the annotated box, or whose"
note "size is implausible for their label, are flagged and left out of the"
note "benchmark proposals. A query about a mislabelled instance is"
note "unanswerable for reasons that have nothing to do with spatial reasoning."
# shellcheck disable=SC2086
$RUN audit --cache "$CACHE" --scene $SCENES --max-rows 4 | sed 's/^/    /'

# ---------------------------------------------------------------- verify
if [[ $SKIP_RENDER -eq 0 && -n "$DATA_ROOT" ]]; then
  say "rendering verification images"
  note "3-D boxes projected into real camera frames. Open a few: if the boxes"
  note "do not sit on the objects, the pose or intrinsic path is wrong and no"
  note "benchmark number from this build means anything."
  FIRST_SCENE="${SCENES%% *}"
  if $RUN render --cache "$CACHE" --root "$DATA_ROOT" --scene "$FIRST_SCENE" \
        --out renders/verify --frames 4 \
        --per-object chair table monitor door 2>&1 | tail -12 | sed 's/^/    /'
  then :; else note "rendering skipped (needs opencv: pip install opencv-python)"; fi
fi

# ---------------------------------------------------------------- items
mkdir -p benchmark/queries
if [[ -z "$ITEMS" ]]; then
  ITEMS="benchmark/queries/scannetpp_${N_SCENES}scenes.jsonl"
fi
PROPOSALS="benchmark/queries/proposals_${N_SCENES}scenes.jsonl"

if [[ $SKIP_PROPOSE -eq 0 ]]; then
  if [[ -s "$ITEMS" ]]; then
    note "an annotation file already exists: $ITEMS"
    note "leaving it alone; delete it or pass --items to start a new one"
  fi
  if [[ -s "$PROPOSALS" ]]; then
    note "proposals already exist: $PROPOSALS ($(wc -l <"$PROPOSALS" | tr -d ' ') items)"
  else
    say "proposing benchmark queries"
    # shellcheck disable=SC2086
    $RUN propose --cache "$CACHE" --scene $SCENES --out "$PROPOSALS"
  fi
fi

# ------------------------------------------------------- frame sensitivity
# Needs no annotation, so it runs on every invocation. It is not accuracy: it
# measures how much the answer depends on which frame is used, which is the
# precondition for the accuracy claim.
if [[ -s "$PROPOSALS" || -s "$ITEMS" ]]; then
  SENS_IN="$PROPOSALS"
  [[ -s "$ITEMS" ]] && SENS_IN="$ITEMS"
  say "frame sensitivity (no annotation needed)"
  mkdir -p results/sensitivity
  $RUN --quiet sensitivity --cache "$CACHE" --items "$SENS_IN" \
       --out results/sensitivity \
       --title "Frame sensitivity: $N_SCENES scenes, perception=$PERCEPTION" \
       2>&1 | grep -E "queries,|disagree on|^\| (egocentric|intrinsic|addressee|world|ordinal|projective)" \
       | sed 's/^/    /'
  note "full report: results/sensitivity/sensitivity.md"
fi

if [[ $PROPOSE_ONLY -eq 1 ]]; then
  say "done (proposals only)"
  note "annotate them with:"
  note "  $PY -m sqe annotate --items $PROPOSALS --out $ITEMS"
  note "or in the browser:"
  note "  $PY -m sqe viewer --cache $CACHE --scene ${SCENES%% *} --items $ITEMS"
  exit 0
fi

# ---------------------------------------------------------------- evaluate
N_ANNOTATED=0
if [[ -s "$ITEMS" ]]; then
  N_ANNOTATED="$($PY - "$ITEMS" <<'PYCOUNT'
import json, sys
n = 0
for line in open(sys.argv[1]):
    line = line.strip()
    if not line:
        continue
    d = json.loads(line)
    if d.get("target_ids") or d.get("ambiguous"):
        n += 1
print(n)
PYCOUNT
)"
fi

if [[ "$N_ANNOTATED" -lt 1 ]]; then
  say "nothing annotated yet, so there is nothing to evaluate"
  cat <<EOM

    Proposals are in:  $PROPOSALS
    Annotate them into: $ITEMS

    In the terminal (blind by default, with a top-down map):
      $PY -m sqe annotate --items $PROPOSALS --out $ITEMS

    Or in the browser, clicking the target object:
      $PY -m sqe viewer --cache $CACHE --scene ${SCENES%% *} --items $ITEMS

    Then run this script again to get the report.

    Aim for 300-500 items. Annotate the 'hard' ones first: those are the
    frame-sensitive queries the report turns on. What matters most is that
    the 'frame' field says which reading the sentence means, and that
    genuinely ambiguous queries are marked as such rather than forced to a
    single answer.
EOM
  exit 0
fi

say "validating $N_ANNOTATED annotated items"
$RUN validate --cache "$CACHE" --items "$ITEMS" || \
  note "validation reported problems (continuing; they are listed above)"

if [[ -z "$OUT" ]]; then
  OUT="results/$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$OUT"

say "evaluating -> $OUT"
LIMIT_ARG=""
[[ -n "$LIMIT" ]] && LIMIT_ARG="--limit $LIMIT"
COMPARE=""
[[ "$PERCEPTION" != "gt" ]] && COMPARE="--compare-perception"
# shellcheck disable=SC2086
$RUN evaluate --cache "$CACHE" --items "$ITEMS" --out "$OUT" \
     --perception "$PERCEPTION" $COMPARE $LIMIT_ARG \
     --title "Spatial query benchmark ($N_SCENES scenes, perception=$PERCEPTION)"

say "done"
note "report:   $OUT/report.md"
note "raw:      $OUT/results.json, $OUT/outcomes.jsonl"
note "viewer:   $PY -m sqe viewer --cache $CACHE --scene ${SCENES%% *}"
