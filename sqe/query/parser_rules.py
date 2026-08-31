"""A deterministic, offline parser for spatial referring expressions.

Why a rule parser at all, when an LLM would be more flexible: because the
benchmark has to separate parsing failures from spatial-reasoning failures, and
a parser whose behaviour drifts between runs makes that impossible. This one is
a pure function of its input. `sqe.query.parser_llm` is the flexible option and
is reported as a separate condition, alongside the gold-parse condition.

Structure of the parse:

1. strip the question wrapper ("which of these is ...", "point at ...");
2. lift out and remove explicit frame markers, so they cannot be mistaken for
   noun phrases;
3. split the remainder into a chain ``NP (relation NP)*`` by longest-match on
   the relation lexicon;
4. attach each relation to the nearest preceding noun phrase -- low attachment,
   which is the usual reading, with the alternative recorded as a note when
   there is more than one relation;
5. parse each noun phrase into ordinal / superlative / attributes / head noun.

Two things are handled before relation splitting because they contain words that
look like relations: "the second X **from the left**", where "from the left"
belongs to the ordinal and is not a relation, and "**between** A **and** B",
which is the only relation taking two anchors.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from ..categories import CATEGORIES, SYNONYMS, normalize_label
from ..relations.base import REGISTRY, canonical_relation
from ..relations.comparative import SIZE_ADJECTIVES, SUPERLATIVES
from ..relations.ordinal import EXTREME_WORDS, FROM_WORDS, parse_ordinal_word
from ..frames.cues import POSSESSIVE_RE, _trim_noun_phrase, extract_cues
from .schema import Constraint, LevelSpec, OrdinalSpec, Phrase, Query

# --------------------------------------------------------------------------
# lexicons
# --------------------------------------------------------------------------

WRAPPERS = [
    r"^(?:which|what|where)(?:\s+(?:one|object|thing|item))?\s+(?:is|are|of these is)\s+",
    r"^(?:which|what)\s+",
    r"^(?:please\s+)?(?:find|locate|point (?:at|to)|show me|select|highlight|"
    r"identify|give me|tell me|get|pick(?: out)?|click(?: on)?)\s+",
    r"^(?:can you|could you|i want|i need)\s+(?:to\s+)?(?:find|show me|see)?\s*",
    r"^(?:the\s+)?(?:location|position)\s+of\s+",
]

TRAILING = [r"\s*\?$", r"\s*\.$", r"^\s*", r"\s*$"]

#: Epistemic hedges, removed wherever they appear. They carry no spatial content,
#: and left in place a trailing one lands inside the anchor noun phrase -- "the
#: mug to the left of the laptop, as far as I can tell" parsed its anchor as
#: "laptop as far as i can tell". Closed list on purpose: a general subordinate-
#: clause stripper would eat real content.
HEDGES = [
    r"as far as i can tell",
    r"if i am not mistaken",
    r"if i'm not mistaken",
    r"as best i can tell",
    r"i think",
    r"i believe",
    r"i would say",
    r"presumably",
    r"apparently",
]

ATTRIBUTES = {
    # colours
    "red", "green", "blue", "yellow", "black", "white", "grey", "gray",
    "brown", "orange", "pink", "purple", "silver", "gold", "beige", "cream",
    "dark", "light", "bright",
    # materials and finishes
    "wooden", "wood", "metal", "metallic", "glass", "plastic", "leather",
    "fabric", "steel", "ceramic", "cardboard",
    # states that behave like attributes
    "open", "closed", "empty", "full", "folded",
}

DETERMINERS = {"the", "a", "an", "this", "that", "these", "those", "some",
               "any", "its", "his", "her", "their", "my", "our", "one"}

GENERIC_NOUNS = {"thing", "things", "object", "objects", "item", "items",
                 "one", "ones", "something", "anything"}

#: Residue left behind after the question wrapper and the relation phrase are
#: removed -- "what **is** to the left of X" leaves a stray "is". Without this,
#: the target noun phrase parses as the class "is".
FILLERS = {"is", "are", "was", "be", "to", "of", "at", "there", "here", "it",
           "that", "which", "what", "does", "do", "sits", "sitting", "located",
           "positioned", "placed", "stands", "standing",
           # intensifiers that modify a relation without changing it
           "immediately", "directly", "just", "exactly", "roughly",
           "approximately", "slightly", "further", "way"}

#: Bare positional adjectives in front of a noun ("the left monitor"). In
#: English these are modifiers, not relations, and they mean "the extreme one on
#: that side" -- so they become an ordinal rather than a spatial constraint.
POSITIONAL_ADJ = {"left": "left", "right": "right", "top": "up",
                  "upper": "up", "bottom": "down", "lower": "down",
                  "near": "front", "nearer": "front", "far": "behind",
                  "further": "behind", "farther": "behind"}

#: Head nouns that name a surface inside another object rather than an object.
LEVEL_NOUNS = {"shelf", "shelves", "level", "levels", "tier", "tiers",
               "row", "rows", "drawer", "drawers", "rack", "step", "layer"}

LEVEL_WORDS = {"top": ("top", None), "topmost": ("top", None),
               "upper": ("top", None), "highest": ("top", None),
               "bottom": ("bottom", None), "bottommost": ("bottom", None),
               "lowest": ("bottom", None), "lower": ("bottom", None),
               "middle": ("middle", None), "centre": ("middle", None),
               "center": ("middle", None), "central": ("middle", None)}

PLURAL_HINTS = {"two", "three", "four", "both", "all", "several", "many"}

#: Relation phrases sorted longest-first, so "in front of" wins over "in".
_REL_PHRASES: List[str] = []


#: Bare words that must NOT be treated as relations when splitting the chain.
#: "the **left** monitor" is a modifier and "the second mug from the **left**"
#: is an ordinal origin; letting either match as a relation split the sentence
#: in the wrong place. Multi-word forms like "left of" are still relations.
#: "under" and "over" stay in: they are unambiguous spatial prepositions.
#: "at", "by", "around" and "within" come out because they are far more often
#: sentence glue than a relation.
BARE_NON_RELATIONS = {"left", "right", "front", "left side", "right side",
                      "left hand side", "right hand side", "beyond", "before",
                      "at", "by", "around", "within",
                      # bare superlatives: these are ordinals on their own
                      "nearest", "farthest", "furthest"}


def _build_relation_phrases() -> List[str]:
    phrases = set()
    for name, spec in REGISTRY.items():
        if spec.family == "ordinal":
            continue
        phrases.add(name.replace("_", " "))
        for a in spec.aliases:
            phrases.add(a)
    # surface variants the lexicon does not spell out
    phrases |= {"to the left of", "to the right of", "on the left of",
                "on the right of", "left of", "right of", "in front of",
                "behind", "on top of", "underneath", "next to", "close to",
                "far from", "inside of", "in between", "between",
                "to the left", "to the right",
                # two-word superlative relations. The bare forms ("nearest",
                # "farthest") are deliberately absent: on their own they are
                # ordinals -- "the nearest chair" -- and splitting the sentence
                # on them loses the head noun.
                "nearest to", "furthest from", "farthest from", "closest to"}
    phrases -= BARE_NON_RELATIONS
    return sorted(phrases, key=lambda p: (-len(p.split()), -len(p)))


def relation_phrases() -> List[str]:
    global _REL_PHRASES
    if not _REL_PHRASES:
        _REL_PHRASES = _build_relation_phrases()
    return _REL_PHRASES


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _clean(text: str) -> str:
    s = " ".join(str(text or "").lower().split())
    s = s.replace("’", "'")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    s = " ".join(s.split())
    for pat in WRAPPERS:
        new = re.sub(pat, "", s)
        if new != s:
            s = " ".join(new.split())
            break
    for h in HEDGES:
        s = re.sub(r"\b" + h + r"\b", " ", s)
    s = " ".join(s.split())
    # "the chair on the left" is not a relation with the anchor "left"; it means
    # the leftmost chair. Rewrite the post-nominal form into the superlative the
    # ordinal extractor already understands. The "of" lookahead keeps the real
    # relation "on the left of the desk" intact.
    s = re.sub(r"\b(?:on|to|at|towards|toward) the (left|right)\b(?!\s+of)",
               r"\1most", s)
    return s.strip()


def _strip_cue_spans(text: str, cues) -> Tuple[str, List[str]]:
    """Remove explicit frame-marker spans so they cannot be parsed as nouns."""
    out, removed = text, []
    for c in cues:
        if c.rule in ("viewer_phrase", "landmark_viewpoint", "addressee_phrase",
                      "intrinsic_phrase", "landmark_viewpoint_from"):
            ev = c.evidence
            if ev and ev in out:
                out = out.replace(ev, " ")
                removed.append(ev)
    # A possessive marker ("the laptop's left") must keep its noun: rewrite it
    # into the equivalent "left of the laptop" so the chain splitter sees a
    # relation. The captured group has to be trimmed to the real head noun --
    # a lazy regex on "the mug on the laptop's left" otherwise captures
    # "the mug on the laptop" and the rewrite comes out as nonsense.
    m = POSSESSIVE_RE.search(out)
    if m:
        noun = _trim_noun_phrase(m.group(1))
        side = {"back": "behind", "rear": "behind"}.get(m.group(2), m.group(2))
        if noun:
            captured = m.group(1)
            cut = captured.rfind(noun)
            prefix = captured[:cut] if cut > 0 else ""
            before = out[:m.start()] + prefix
            # "the second mug from the bookshelf's own left" -- here the
            # possessive sits in an ordinal's counting-origin slot, so the side
            # word has to stay put and only the owner comes out. Rewriting it to
            # "left of the bookshelf" would destroy the ordinal.
            if re.search(r"\bfrom (?:the )?$", before):
                out = before + side + out[m.end():]
            else:
                out = before + f"{side} of the {noun}" + out[m.end():]
            removed.append(m.group(0))
    return " ".join(out.split()), removed


def _lift_from_origin(body: str, has_ordinal: bool):
    """Pull "from the left" / "from the door" out of the sentence.

    Only when an ordinal word is present. "The second chair **from the door**"
    counts outwards from the door; "the chair, seen **from the door**" describes
    where the speaker stands. Same words, different job, and the deciding
    evidence is whether anything is being counted.

    Both patterns are anchored to closed vocabularies. A permissive
    ``([a-z]+(?: [a-z]+)?)`` capture reads "from the left on the middle shelf"
    as the landmark "left on", which then eats the relation.
    """
    if not has_ordinal:
        return body, None, None
    m = re.search(r"\bfrom (?:the )?(left|right|front|back|rear|top|bottom|"
                  r"near|far)\b(?:\s+(?:side|end|wall))?", body)
    if m:
        reduced = " ".join((body[:m.start()] + " " + body[m.end():]).split())
        return reduced, FROM_WORDS.get(m.group(1), "left"), None
    m = re.search(r"\bfrom (?:the )?([a-z]+)\b", body)
    if m:
        token = m.group(1)
        if token in DETERMINERS or token in FILLERS or canonical_relation(token):
            return body, None, None
        reduced = " ".join((body[:m.start()] + " " + body[m.end():]).split())
        return reduced, None, token
    return body, None, None


#: Surface forms that make a proximity relation a ranking rather than a test.
SUPERLATIVE_SURFACE = ("nearest", "closest", "furthest", "farthest")


def _split_chain(text: str) -> List[Tuple[str, Optional[str], str]]:
    """Split into [(np, relation_following_it, surface_form), ...].

    The last element's relation is None. Longest-match on the relation lexicon,
    scanning left to right over word boundaries. The surface form is kept
    because "nearest to" and "near" map to the same relation but do not mean the
    same thing.
    """
    words = text.split()
    parts: List[Tuple[str, Optional[str], str]] = []
    buf: List[str] = []
    i = 0
    phrases = relation_phrases()
    while i < len(words):
        matched = None
        for p in phrases:
            pw = p.split()
            if words[i:i + len(pw)] == pw:
                matched = (p, len(pw))
                break
        if matched:
            rel = canonical_relation(matched[0])
            if rel:
                # An empty buffer is fine and means the sentence opens with the
                # relation -- "to the left of the laptop" -- which is a query
                # for any object in that position.
                parts.append((" ".join(buf), rel, matched[0]))
                buf = []
                i += matched[1]
                continue
        buf.append(words[i])
        i += 1
    parts.append((" ".join(buf), None, ""))
    return parts


def _has_ordinal_word(text: str) -> bool:
    for w in text.split():
        if w in EXTREME_WORDS or w in LEVEL_WORDS:
            return True
        if parse_ordinal_word(w) is not None:
            return True
    return False


def _extract_level(words: List[str], from_word: Optional[str] = None
                   ) -> Tuple[List[str], Optional[LevelSpec]]:
    """Turn "the middle shelf" into a LevelSpec when the head noun is a surface.

    Runs *before* the ordinal extractor, because "the second shelf from the
    bottom" is a level and not an ordinal over objects, and whichever extractor
    goes first consumes the word "second".
    """
    ws = list(words)
    head = ws[-1] if ws else ""
    if head not in LEVEL_NOUNS:
        return ws, None
    from_bottom = from_word != "up"
    for i, w in enumerate(ws[:-1]):
        if w in LEVEL_WORDS:
            kind, _ = LEVEL_WORDS[w]
            ws.pop(i)
            return ws, LevelSpec(word=kind, middle=(kind == "middle"),
                                 index=None, from_bottom=(kind != "top"))
        idx = parse_ordinal_word(w)
        if idx is not None:
            ws.pop(i)
            return ws, LevelSpec(word=w, index=idx, from_bottom=from_bottom)
    return ws, None


def _extract_ordinal(words: List[str], from_word: Optional[str] = None,
                     from_landmark: Optional[str] = None
                     ) -> Tuple[List[str], Optional[OrdinalSpec]]:
    """Pull an ordinal out of a noun phrase's word list.

    Handles "second", "leftmost", "middle", and the bare positional modifier in
    "the left monitor". The counting origin has already been lifted out of the
    sentence by `_lift_from_origin` and arrives here as an argument.
    """
    ws = list(words)

    # a superlative-as-ordinal: leftmost, nearest, topmost
    for i, w in enumerate(ws):
        if w in EXTREME_WORDS:
            side, idx = EXTREME_WORDS[w]
            ws.pop(i)
            return ws, OrdinalSpec(word=w, index=idx,
                                   from_word=from_word or side,
                                   from_landmark=from_landmark)

    # an explicit ordinal word
    for i, w in enumerate(ws):
        if w in ("middle", "centre", "center", "central"):
            ws.pop(i)
            return ws, OrdinalSpec(word=w, index=None, middle=True,
                                   from_word=from_word or "left",
                                   from_landmark=from_landmark)
        idx = parse_ordinal_word(w)
        if idx is not None and w not in DETERMINERS:
            ws.pop(i)
            return ws, OrdinalSpec(word=w, index=idx,
                                   from_word=from_word or "left",
                                   from_landmark=from_landmark)

    # a bare positional modifier directly in front of the head noun
    if len(ws) >= 2 and ws[-2] in POSITIONAL_ADJ:
        w = ws.pop(-2)
        return ws, OrdinalSpec(word=w, index=0,
                               from_word=from_word or POSITIONAL_ADJ[w],
                               from_landmark=from_landmark)

    if from_word or from_landmark:
        # "the chair from the door" with no ordinal: read as the nearest one
        return ws, OrdinalSpec(word="", index=0, from_word=from_word or "left",
                               from_landmark=from_landmark)
    return ws, None


def parse_phrase(text: str) -> Phrase:
    """Parse one noun phrase, including its own counting origin.

    The "from the left" lift happens here rather than at sentence level. Doing
    it globally attached the origin to the first noun phrase, which is wrong for
    "the bottle on the second shelf **from the bottom**" -- there the origin
    belongs to the shelf, not the bottle. Chain splitting is safe with "from the
    left" still in place, because neither "from" nor a bare "left" is a relation.
    """
    raw = " ".join(str(text or "").split())
    # "the middle shelf of the bookshelf" is one noun phrase naming a surface
    # inside another object. Without this the head noun comes out as
    # "bookshelf" and "middle" becomes an ordinal over bookshelves, of which
    # there is one -- so the query silently resolves to the wrong shelf.
    owner = ""
    m = re.match(r"^(.*?\b(?:" + "|".join(sorted(LEVEL_NOUNS)) + r"))\s+of\s+"
                 r"(?:the\s+)?(.+)$", raw)
    if m:
        raw, owner = m.group(1), m.group(2).strip()

    body, from_word, from_landmark = _lift_from_origin(raw,
                                                       _has_ordinal_word(raw))
    ws = [w for w in body.split() if w]

    plural = any(w in PLURAL_HINTS for w in ws)
    ws, level = _extract_level(ws, from_word)
    ws, ordinal = _extract_ordinal(ws, from_word, from_landmark)
    if level is not None:
        ordinal = None            # a level already says which surface

    ws = [w for w in ws if w not in DETERMINERS and w not in FILLERS]

    superlative, size_word = None, None
    kept: List[str] = []
    attributes: List[str] = []
    for w in ws:
        if w in SUPERLATIVES and superlative is None:
            superlative = w
        elif w in SIZE_ADJECTIVES and size_word is None:
            size_word = w
        elif w in ATTRIBUTES:
            attributes.append(w)
        elif w in PLURAL_HINTS:
            continue
        else:
            kept.append(w)

    head_text = " ".join(kept).strip()
    label: Optional[str] = None
    if head_text and head_text not in GENERIC_NOUNS:
        if not plural and head_text.endswith("s"):
            plural = (head_text not in CATEGORIES and head_text not in SYNONYMS
                      and not head_text.endswith("ss"))
        label = normalize_label(head_text)
        if label in GENERIC_NOUNS or not label:
            label = None
    if level is not None and label is None:
        label = "shelf"
    if owner:
        # the phrase refers to a surface of `owner`, so the object to find is
        # the owner and the level narrows it down
        if level is None:
            level = LevelSpec(word="", index=None)
        label = normalize_label(owner)
        head_text = owner

    return Phrase(label=label, text=head_text or raw, attributes=attributes,
                  size_word=size_word, superlative=superlative,
                  ordinal=ordinal, level=level, plural=plural)


def _split_two_anchors(text: str) -> List[str]:
    """"the laptop and the keyboard" -> two phrases."""
    parts = re.split(r"\s+and\s+", text, maxsplit=1)
    return [p.strip() for p in parts if p.strip()]


def parse(text: str) -> Query:
    """Parse a spatial query into a `Query`. Never raises."""
    original = str(text or "")
    cleaned = _clean(original)
    notes: List[str] = []

    cues = extract_cues(cleaned)
    # "from the door" is a counting origin when the sentence is counting, and a
    # viewpoint otherwise. The ambiguous bare form is dropped here so that the
    # per-phrase lift can claim it; the explicit forms ("seen from the door")
    # are never dropped.
    if _has_ordinal_word(cleaned):
        dropped = [c for c in cues if c.rule == "landmark_viewpoint_from"]
        if dropped:
            cues = [c for c in cues if c.rule != "landmark_viewpoint_from"]
            notes.append(f"'{dropped[0].evidence}' read as an ordinal's counting "
                         f"origin rather than as the speaker's position, because "
                         f"the query counts")

    body, removed = _strip_cue_spans(cleaned, cues)
    if removed:
        notes.append("frame markers lifted out of the noun phrases: "
                     + "; ".join(removed))

    chain = _split_chain(body)
    phrases: List[Phrase] = []
    relations: List[Optional[str]] = []
    surfaces: List[str] = []
    for np_text, rel, surface in chain:
        phrases.append(parse_phrase(np_text))
        relations.append(rel)
        surfaces.append(surface)

    # right-branching structure with low attachment
    target = phrases[0] if phrases else Phrase()
    cursor = target
    i = 0
    while i < len(relations) - 1:
        rel = relations[i]
        if rel is None:
            break
        anchor_phrase = phrases[i + 1]
        anchors = [anchor_phrase]
        if rel == "between":
            pieces = _split_two_anchors(chain[i + 1][0])
            if len(pieces) == 2:
                anchors = [parse_phrase(pieces[0]), parse_phrase(pieces[1])]
                phrases[i + 1] = anchors[0]
                anchor_phrase = anchors[0]
            else:
                notes.append("'between' needs two anchors joined by 'and'; "
                             "only one was found")
        sup = any(surfaces[i].startswith(w) for w in SUPERLATIVE_SURFACE)
        cursor.constraints.append(Constraint(relation=rel, anchors=anchors,
                                             superlative=sup))
        cursor = anchor_phrase
        i += 1

    if sum(1 for r in relations if r) > 1:
        notes.append("more than one relation: attached low (each relation "
                     "modifies the nearest preceding noun phrase); the high "
                     "attachment reading is also possible")

    frame_hint = cues[0].kind if cues else None
    viewpoint_hint = next((c.viewpoint_hint for c in cues if c.viewpoint_hint), None)

    confidence = 1.0
    if not target.label and not target.constraints and not target.ordinal:
        confidence = 0.3
        notes.append("no head noun and no relation were recognised")
    elif not target.label:
        confidence = 0.75
        notes.append("the target has no explicit class; treating it as "
                     "'any object'")

    return Query(text=original, target=target, frame_hint=frame_hint,
                 viewpoint_hint=viewpoint_hint,
                 frame_cues=[c.to_dict() for c in cues],
                 expects_multiple=target.plural, parser="rules",
                 parse_confidence=confidence, notes=notes)
