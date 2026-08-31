"""Category priors: which object classes have an intrinsic front, which can
support things, which can contain things, and what geometric cue identifies the
front.

This table is small, hand-written and deliberately visible. It is the honest
place for knowledge that a pipeline otherwise smuggles in as a threshold. An
unknown label falls through to `has_front = False`, which makes the intrinsic
reference frame *unavailable* rather than silently wrong -- the query then comes
back flagged instead of confidently misresolved.

`front_cues` are tried in order by `sqe.perception.orientation`:

* `backrest`     -- tall thin vertical mass on one side; front faces away from it
* `away_from_wall` -- the object is placed against a wall; front faces the room
* `thin_face`    -- a flat panel (screen, picture, door); front is the large face
* `long_side`    -- front is normal to the longer horizontal side
* `open_face`    -- the face with the largest cavity / fewest points
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class CategoryPrior:
    name: str
    has_front: bool = False
    front_cues: Tuple[str, ...] = ()
    support_surface: bool = False       # things can be "on" it
    container: bool = False            # things can be "in" it
    room_fixed: bool = False           # part of the room shell
    movable_small: bool = False        # a plausible target for fine queries
    typical_height: Optional[float] = None   # metres, for sanity checks
    #: Whether `typical_height` is tight enough to flag an annotation by. True
    #: only for classes whose height really is constrained -- a chair is always
    #: about 0.9 m, whereas "lamp" spans a 10 cm desk lamp and a 1.6 m floor
    #: lamp and "shelf" spans a shoe rack and a full bookcase. Applying the test
    #: to those produced false alarms on perfectly good annotations.
    height_stable: bool = False


def _c(name, **kw) -> CategoryPrior:
    return CategoryPrior(name=name, **kw)


_SEATING = dict(has_front=True, front_cues=("backrest", "away_from_wall"),
                support_surface=True)
_CASEGOODS = dict(has_front=True, front_cues=("away_from_wall", "open_face", "long_side"),
                  support_surface=True, container=True)
_SCREEN = dict(has_front=True, front_cues=("thin_face", "away_from_wall"))
_PANEL = dict(has_front=True, front_cues=("thin_face", "away_from_wall"), room_fixed=True)
_VESSEL = dict(container=True, movable_small=True)

CATEGORIES: Dict[str, CategoryPrior] = {c.name: c for c in [
    # ---- seating & sleeping -------------------------------------------------
    _c("chair", **_SEATING, typical_height=0.9, height_stable=True),
    _c("office chair", **_SEATING, typical_height=1.0, height_stable=True),
    _c("armchair", **_SEATING, typical_height=0.85, height_stable=True),
    _c("sofa", **_SEATING, typical_height=0.8, height_stable=True),
    _c("bench", **_SEATING, typical_height=0.45, height_stable=True),
    _c("stool", support_surface=True, typical_height=0.5),
    _c("bed", has_front=True, front_cues=("away_from_wall", "long_side"),
       support_surface=True, typical_height=0.6),

    # ---- tables & work surfaces --------------------------------------------
    _c("table", support_surface=True, typical_height=0.75, height_stable=True),
    _c("desk", has_front=True, front_cues=("away_from_wall", "long_side"),
       support_surface=True, typical_height=0.75),
    _c("coffee table", support_surface=True, typical_height=0.4, height_stable=True),
    _c("dining table", support_surface=True, typical_height=0.75, height_stable=True),
    _c("side table", support_surface=True, typical_height=0.6),
    _c("counter", support_surface=True, room_fixed=True, typical_height=0.9, height_stable=True),
    _c("countertop", support_surface=True, room_fixed=True, typical_height=0.9, height_stable=True),
    _c("kitchen counter", support_surface=True, room_fixed=True, typical_height=0.9, height_stable=True),
    _c("windowsill", support_surface=True, room_fixed=True),

    # ---- storage ------------------------------------------------------------
    # no typical_height: "shelf" spans a shoe rack and a full bookcase
    _c("shelf", **_CASEGOODS),
    _c("bookshelf", **_CASEGOODS),
    _c("shelving unit", **_CASEGOODS),
    _c("cabinet", **_CASEGOODS, typical_height=0.9),
    _c("kitchen cabinet", **_CASEGOODS, typical_height=0.9),
    _c("wardrobe", **_CASEGOODS, typical_height=2.0),
    _c("closet", **_CASEGOODS, room_fixed=True),
    _c("dresser", **_CASEGOODS, typical_height=0.9),
    _c("nightstand", **_CASEGOODS, typical_height=0.6),
    _c("sideboard", **_CASEGOODS, typical_height=0.85),
    _c("tv stand", **_CASEGOODS, typical_height=0.5),
    _c("drawer", has_front=True, front_cues=("away_from_wall",), container=True),
    _c("box", container=True, support_surface=True, movable_small=True),
    _c("basket", **_VESSEL),
    _c("suitcase", container=True, has_front=True, front_cues=("thin_face",)),
    _c("backpack", container=True, movable_small=True),
    _c("bag", container=True, movable_small=True),

    # ---- appliances ---------------------------------------------------------
    _c("refrigerator", **_CASEGOODS, typical_height=1.8, height_stable=True),
    _c("oven", **_CASEGOODS, typical_height=0.9, height_stable=True),
    _c("microwave", has_front=True, front_cues=("thin_face", "away_from_wall"),
       container=True, support_surface=True, typical_height=0.3),
    _c("dishwasher", **_CASEGOODS, typical_height=0.85, height_stable=True),
    _c("washing machine", **_CASEGOODS, typical_height=0.85, height_stable=True),
    _c("stove", has_front=True, front_cues=("away_from_wall",), support_surface=True),
    _c("range hood", has_front=True, front_cues=("away_from_wall",)),
    _c("printer", has_front=True, front_cues=("away_from_wall",), typical_height=0.3),

    # ---- screens & wall panels ---------------------------------------------
    _c("tv", **_SCREEN, typical_height=0.7),
    _c("monitor", **_SCREEN, typical_height=0.5),
    # an open laptop is geometrically a tiny chair: upright screen behind, work
    # surface in front, which is exactly what the `backrest` cue reads
    # no typical_height: a closed laptop is 3 cm and an open one 25 cm
    _c("laptop", has_front=True, front_cues=("backrest", "thin_face")),
    _c("keyboard", has_front=True, front_cues=("long_side",), movable_small=True),
    _c("whiteboard", **_PANEL),
    _c("mirror", **_PANEL),
    _c("picture", **_PANEL),
    _c("poster", **_PANEL),
    _c("clock", has_front=True, front_cues=("thin_face", "away_from_wall")),

    # ---- room shell ---------------------------------------------------------
    _c("wall", room_fixed=True),
    _c("floor", support_surface=True, room_fixed=True),
    _c("ceiling", room_fixed=True),
    _c("door", **_PANEL),
    _c("doorway", room_fixed=True),
    _c("window", **_PANEL),
    _c("curtain", room_fixed=True),
    _c("blinds", room_fixed=True),
    _c("stairs", room_fixed=True, has_front=True, front_cues=("long_side",)),
    _c("radiator", has_front=True, front_cues=("away_from_wall",), room_fixed=True),
    _c("heater", has_front=True, front_cues=("away_from_wall",), room_fixed=True),
    _c("rug", room_fixed=True),
    _c("vent", room_fixed=True),

    # ---- bathroom -----------------------------------------------------------
    _c("toilet", has_front=True, front_cues=("away_from_wall",), typical_height=0.75, height_stable=True),
    _c("sink", has_front=True, front_cues=("away_from_wall",), container=True),
    _c("bathtub", container=True, has_front=True, front_cues=("away_from_wall",)),
    _c("shower", room_fixed=True, container=True),
    _c("towel", movable_small=True),

    # ---- small movable objects ---------------------------------------------
    _c("mug", **_VESSEL, typical_height=0.1),
    _c("cup", **_VESSEL, typical_height=0.1),
    _c("glass", **_VESSEL, typical_height=0.12),
    _c("bottle", **_VESSEL),
    _c("can", **_VESSEL, typical_height=0.12),
    _c("bowl", **_VESSEL, typical_height=0.08),
    _c("plate", movable_small=True, support_surface=True),
    _c("pot", **_VESSEL),
    _c("pan", **_VESSEL),
    _c("jar", **_VESSEL),
    _c("book", has_front=True, front_cues=("thin_face",), movable_small=True),
    # no typical_height: desk lamps and floor lamps differ by 4x
    _c("lamp", movable_small=True),
    _c("floor lamp", movable_small=True, typical_height=1.5),
    _c("plant", movable_small=True, typical_height=0.5),
    _c("vase", **_VESSEL),
    _c("trash can", **_VESSEL, typical_height=0.4),
    _c("pillow", movable_small=True),
    _c("cushion", movable_small=True),
    _c("blanket", movable_small=True),
    _c("phone", has_front=True, front_cues=("thin_face",), movable_small=True),
    _c("remote", has_front=True, front_cues=("long_side",), movable_small=True),
    _c("mouse", movable_small=True),
    _c("pen", movable_small=True),
    _c("paper", movable_small=True),
    _c("shoe", has_front=True, front_cues=("long_side",), movable_small=True),
    _c("clothes", movable_small=True),
    _c("speaker", has_front=True, front_cues=("thin_face",), movable_small=True),
    _c("candle", movable_small=True),
    _c("tissue box", container=True, movable_small=True),

    # ---- labels that actually occur in the ScanNet++ scenes ----------------
    # Added after reading the annotation vocabulary rather than guessed. The
    # important ones are the room-fixed fittings: without these, a ceiling lamp
    # folds onto "lamp" and becomes a movable object that queries can pick as a
    # target, and a power socket becomes a plausible referent for "the thing on
    # the left".
    _c("ceiling lamp", room_fixed=True),
    _c("fake ceiling", room_fixed=True),
    _c("power socket", room_fixed=True),
    _c("wall outlet", room_fixed=True),
    _c("light switch", room_fixed=True),
    _c("thermostat", room_fixed=True),
    _c("electrical duct", room_fixed=True),
    _c("blind rail", room_fixed=True),
    _c("door frame", room_fixed=True),
    _c("window frame", room_fixed=True),
    _c("pillar", room_fixed=True),
    _c("pipe", room_fixed=True),
    _c("wall clock", has_front=True, front_cues=("thin_face", "away_from_wall"),
       room_fixed=True),
    _c("computer tower", has_front=True, front_cues=("away_from_wall", "long_side"),
       typical_height=0.42),
    _c("projector", has_front=True, front_cues=("long_side",)),
    _c("storage cabinet", **_CASEGOODS, typical_height=1.2),
    _c("cardboard box", container=True, support_surface=True, movable_small=True),
    _c("whiteboard eraser", movable_small=True),
    _c("toilet paper", movable_small=True),
    _c("cleaning trolley", container=True, support_surface=True),
    _c("floor cleaner", movable_small=True),
    _c("tripod"),
    _c("jacket", movable_small=True),
    _c("jar", **_VESSEL),
]}

# Label synonyms folded onto the canonical names above. Left side is what a
# dataset or a user might say; right side is a key of CATEGORIES.
SYNONYMS: Dict[str, str] = {
    "couch": "sofa", "settee": "sofa", "loveseat": "sofa",
    "television": "tv", "tv screen": "tv", "telly": "tv", "television set": "tv",
    "computer monitor": "monitor", "display": "monitor", "screen": "monitor",
    "fridge": "refrigerator", "freezer": "refrigerator", "icebox": "refrigerator",
    "book shelf": "bookshelf", "bookcase": "bookshelf", "book case": "bookshelf",
    "shelves": "shelf", "rack": "shelf", "storage shelf": "shelf",
    "night stand": "nightstand", "bedside table": "nightstand",
    "chest of drawers": "dresser", "commode": "dresser",
    "waste bin": "trash can", "wastebasket": "trash can", "bin": "trash can",
    "garbage can": "trash can", "rubbish bin": "trash can", "dustbin": "trash can",
    "trashcan": "trash can", "trash bin": "trash can",
    "coffee mug": "mug", "tea cup": "cup", "teacup": "cup",
    "water bottle": "bottle", "flask": "bottle",
    "potted plant": "plant", "houseplant": "plant", "pot plant": "plant",
    "office desk": "desk", "work desk": "desk", "writing desk": "desk",
    "swivel chair": "office chair", "desk chair": "office chair",
    "dining chair": "chair", "seat": "chair",
    "notebook computer": "laptop", "laptop computer": "laptop", "macbook": "laptop",
    "washer": "washing machine", "washing mashine": "washing machine",
    "cooker": "stove", "hob": "stove", "cooktop": "stove",
    "wash basin": "sink", "basin": "sink", "washbasin": "sink",
    "wc": "toilet", "lavatory": "toilet",
    "photo": "picture", "painting": "picture", "photograph": "picture",
    "framed picture": "picture", "artwork": "picture", "wall art": "picture",
    "cabinet door": "cabinet", "cupboard": "cabinet", "kitchen cupboard": "cabinet",
    "tv cabinet": "tv stand", "media console": "tv stand",
    "office table": "desk", "side board": "sideboard",
    "carpet": "rug", "mat": "rug", "doormat": "rug",
    "light": "ceiling lamp", "lamp shade": "lamp", "standing lamp": "floor lamp",
    "table lamp": "lamp", "desk lamp": "lamp", "pendant light": "ceiling lamp",
    "window blind": "blinds", "window blinds": "blinds", "shade": "blinds",
    "radiator heater": "radiator", "heating": "radiator",
    "cell phone": "phone", "mobile phone": "phone", "smartphone": "phone",
    "remote control": "remote", "tv remote": "remote",
    "computer mouse": "mouse",
    "waste paper basket": "trash can",
    "stairway": "stairs", "staircase": "stairs", "step": "stairs",
    "board": "whiteboard", "blackboard": "whiteboard",
    # ScanNet++ spellings
    "ceiling light": "ceiling lamp", "pipes": "pipe",
    "office table": "desk", "computer": "computer tower",
    "pc": "computer tower", "desktop computer": "computer tower",
    "socket": "power socket", "plug socket": "power socket",
    "duct": "electrical duct", "electric duct": "electrical duct",
    "column": "pillar",
}

_PLURAL_EXCEPTIONS = {"glasses": "glass", "shelves": "shelf", "boxes": "box",
                      "shoes": "shoe", "dishes": "plate", "clothes": "clothes",
                      "stairs": "stairs", "blinds": "blinds", "trousers": "clothes"}

_ARTICLES = re.compile(r"^(the|a|an|this|that|some)\s+")
_NONWORD = re.compile(r"[^a-z0-9 ]+")


#: A synonym key that is also a category name is dead code: the category lookup
#: runs first and the mapping never fires. Checked at import so the two tables
#: cannot silently drift apart -- this caught "ceiling light" being both its own
#: category and a synonym for "ceiling lamp".
_COLLISIONS = sorted(set(SYNONYMS) & set(CATEGORIES))
if _COLLISIONS:
    raise RuntimeError(
        f"these names are both a category and a synonym key, so the synonym "
        f"can never fire: {_COLLISIONS}")

_DANGLING = sorted({v for v in SYNONYMS.values()} - set(CATEGORIES))
if _DANGLING:
    raise RuntimeError(
        f"these synonym targets are not categories: {_DANGLING}")


def normalize_label(label: str) -> str:
    """Fold a raw dataset or user label onto a canonical category name.

    Handles articles, punctuation, ScanNet++-style `foo/bar` and `foo_bar`
    labels, simple plurals and the synonym table. Returns a lowercase string
    which may still be outside `CATEGORIES` -- that is fine and expected.
    """
    if label is None:
        return ""
    s = str(label).strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = s.split("/")[0].strip()          # ScanNet++ writes "chair/armchair"
    s = _NONWORD.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = _ARTICLES.sub("", s)
    if not s:
        return ""
    if s in CATEGORIES:
        return s
    if s in SYNONYMS:
        return SYNONYMS[s]
    if s in _PLURAL_EXCEPTIONS:
        return _PLURAL_EXCEPTIONS[s]
    if s.endswith("ies") and len(s) > 4:
        cand = s[:-3] + "y"
        if cand in CATEGORIES or cand in SYNONYMS:
            return SYNONYMS.get(cand, cand)
    if s.endswith("es") and (s[:-2] in CATEGORIES or s[:-2] in SYNONYMS):
        return SYNONYMS.get(s[:-2], s[:-2])
    if s.endswith("s") and not s.endswith("ss"):
        cand = s[:-1]
        if cand in CATEGORIES or cand in SYNONYMS:
            return SYNONYMS.get(cand, cand)
    # multi-word fallback: the last word often carries the head noun
    words = s.split()
    if len(words) > 1:
        head = words[-1]
        if head in CATEGORIES:
            return head
        if head in SYNONYMS:
            return SYNONYMS[head]
    return s


def prior(label: str) -> CategoryPrior:
    """Category prior for a label, with a permissive unknown default."""
    key = normalize_label(label)
    got = CATEGORIES.get(key)
    if got is not None:
        return got
    return CategoryPrior(name=key or "unknown")


def has_intrinsic_front(label: str) -> bool:
    return prior(label).has_front


def is_support_surface(label: str) -> bool:
    return prior(label).support_surface


def is_container(label: str) -> bool:
    return prior(label).container


def is_room_fixed(label: str) -> bool:
    return prior(label).room_fixed


def front_cues(label: str) -> Tuple[str, ...]:
    return prior(label).front_cues


def label_matches(query_label: str, object_label: str) -> float:
    """Soft label agreement in [0, 1], for the closed-vocabulary path.

    Exact canonical match scores 1. A shared head noun scores 0.6. Anything
    else scores 0. The open-vocabulary path uses feature similarity instead and
    ignores this function.
    """
    q, o = normalize_label(query_label), normalize_label(object_label)
    if not q or not o:
        return 0.0
    if q == o:
        return 1.0
    qw, ow = q.split(), o.split()
    if qw[-1] == ow[-1]:
        return 0.6
    # "shelf" should match "bookshelf": a compound whose head is the query word
    # is a subtype of it. Scored above the generic-substring case because the
    # resolver needs it to survive a 0.6 threshold -- "on the middle shelf" must
    # find a bookshelf.
    if o.endswith(q) or q.endswith(o):
        return 0.7
    if q in o or o in q:
        return 0.5
    return 0.0


def known_categories() -> List[str]:
    return sorted(CATEGORIES)
