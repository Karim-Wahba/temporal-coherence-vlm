"""
linguistic/syntactic.py
-----------------------
Lightweight phrase-structure + modifier-head extraction for short referring
expressions. Uses NLTK POS tagging + a regex chunk grammar — no spaCy required.

Returns a compact dict per expression so it can be injected into the OPRO
prompt alongside IoU / MassGT:

  {
    "head_noun":  "bike",
    "modifiers":  [("black", "amod"), ("a", "det")],
    "verbs":      [{"verb": "riding", "subj": "man", "obj": "bike"}],
    "prep_phrases": [{"prep": "on", "obj": "left"}],
    "deps":       [
      ("a",     "det",  "bike"),
      ("black", "amod", "bike"),
      ...
    ],
    "summary":    "head=bike  amod=[black]  det=[a]"
  }

Single-line `summary` is what we inject into the prompt for compactness; the
full dict is also stored for the outer-loop meta-analysis.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import nltk
from nltk import RegexpParser, Tree


# ── POS-tag groupings ─────────────────────────────────────────────────────────

_NOUN_TAGS = {"NN", "NNS", "NNP", "NNPS"}
_VERB_TAGS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
_ADJ_TAGS  = {"JJ", "JJR", "JJS"}
_DET_TAGS  = {"DT", "PRP$", "WDT"}
_PREP_TAGS = {"IN", "TO"}
_ADV_TAGS  = {"RB", "RBR", "RBS"}

# Grammar: NP = (DT|PRP$)? (ADJ|VBN|VBG)* (NN+)
#          PP = IN NP
#          VP = VB.* (NP|PP)*
_GRAMMAR = r"""
  NP: {<DT|PRP\$|WDT>?<JJ.*|VBN|VBG>*<NN.*>+}
  PP: {<IN|TO><NP>}
  VP: {<VB.*><NP|PP>*}
"""


# ── Tokenize + POS without depending on `punkt` ───────────────────────────────

def _tokenize(text: str) -> List[str]:
    # Word-level split that doesn't require punkt
    return re.findall(r"[A-Za-z']+", text.lower())


def _pos_tag(text: str) -> List[Tuple[str, str]]:
    return nltk.pos_tag(_tokenize(text))


# ── Phrase + dependency extraction ────────────────────────────────────────────

def _walk_chunks(tree: Tree, label: str) -> List[Tree]:
    out: List[Tree] = []
    for sub in tree:
        if isinstance(sub, Tree):
            if sub.label() == label:
                out.append(sub)
            out.extend(_walk_chunks(sub, label))
    return out


def _np_head(np: Tree) -> Optional[str]:
    """Best guess at the head noun of an NP chunk.

    NLTK off-the-shelf tagging makes two systematic mistakes on referring
    expressions:
      - tags gerunds ('jumping', 'swimming') as NN when they should be verb-like
      - tags compound-y nouns ('goldfish', 'pickup') as JJ
    Heuristic: prefer the rightmost noun-or-adjective whose surface form does
    NOT end in -ing/-ed. If none exists, fall back to the last NN.
    """
    candidates = [
        w for w, t in np.leaves()
        if t in _NOUN_TAGS or t in _ADJ_TAGS
    ]
    if not candidates:
        return None
    non_participle = [w for w in candidates
                      if not (w.endswith("ing") or w.endswith("ed"))]
    if non_participle:
        return non_participle[-1]
    return candidates[-1]


def _np_modifiers(np: Tree) -> List[Tuple[str, str]]:
    """(word, relation) for non-noun tokens in the NP."""
    mods: List[Tuple[str, str]] = []
    for w, t in np.leaves():
        if t in _ADJ_TAGS:                 mods.append((w, "amod"))
        elif t in {"VBN", "VBG"}:          mods.append((w, "amod_participle"))
        elif t in _DET_TAGS:               mods.append((w, "det"))
    return mods


def extract_structure(expression: str) -> dict:
    """Parse one expression and return a compact phrase-structure dict."""
    tags = _pos_tag(expression)
    if not tags:
        return {
            "head_noun":  None,
            "modifiers":  [],
            "verbs":      [],
            "prep_phrases": [],
            "deps":       [],
            "summary":    "(empty)",
        }

    parser = RegexpParser(_GRAMMAR)
    tree = parser.parse(tags)

    nps = _walk_chunks(tree, "NP")
    vps = _walk_chunks(tree, "VP")
    pps = _walk_chunks(tree, "PP")

    # Head noun: rightmost NP's head, falling back to last NN in the sentence.
    head_noun: Optional[str] = None
    if nps:
        head_noun = _np_head(nps[0])         # leftmost NP — the matrix subject/head
    if not head_noun:
        nouns = [w for w, t in tags if t in _NOUN_TAGS]
        head_noun = nouns[-1] if nouns else None

    modifiers: List[Tuple[str, str]] = []
    if nps:
        modifiers = _np_modifiers(nps[0])
    # Avoid reporting the head word as its own modifier (happens when the
    # tagger mislabels a noun like 'goldfish' as JJ).
    if head_noun is not None:
        modifiers = [(w, r) for w, r in modifiers if w != head_noun]

    # Verbs: each VP yields one record (verb + first object NP if any)
    verb_records: List[Dict[str, Optional[str]]] = []
    for vp in vps:
        verb_token = next((w for w, t in vp.leaves() if t in _VERB_TAGS), None)
        if not verb_token:
            continue
        obj_np = next((sub for sub in vp if isinstance(sub, Tree) and sub.label() == "NP"), None)
        obj_head = _np_head(obj_np) if obj_np else None
        verb_records.append({
            "verb": verb_token,
            "subj": head_noun,
            "obj":  obj_head,
        })

    # Prepositional phrases: "in the middle", "on the left", "with patches"
    prep_records: List[Dict[str, Optional[str]]] = []
    for pp in pps:
        prep_token = next((w for w, t in pp.leaves() if t in _PREP_TAGS), None)
        obj_np = next((sub for sub in pp if isinstance(sub, Tree) and sub.label() == "NP"), None)
        obj_head = _np_head(obj_np) if obj_np else None
        if prep_token:
            prep_records.append({"prep": prep_token, "obj": obj_head})

    # Build dependency triples
    deps: List[Tuple[str, str, str]] = []
    for w, rel in modifiers:
        if head_noun:
            deps.append((w, rel, head_noun))
    for vr in verb_records:
        if vr["subj"]: deps.append((vr["verb"], "nsubj", vr["subj"]))
        if vr["obj"]:  deps.append((vr["verb"], "dobj",  vr["obj"]))
    for pr in prep_records:
        if pr["obj"] and head_noun:
            deps.append((pr["prep"], "prep", head_noun))
            deps.append((pr["obj"],  "pobj", pr["prep"]))

    return {
        "head_noun":   head_noun,
        "modifiers":   modifiers,
        "verbs":       verb_records,
        "prep_phrases": prep_records,
        "deps":        deps,
        "summary":     _format_summary(head_noun, modifiers, verb_records, prep_records),
    }


# ── One-line summary for the OPRO prompt ──────────────────────────────────────

def _format_summary(
    head: Optional[str],
    mods: List[Tuple[str, str]],
    verbs: List[Dict],
    preps: List[Dict],
) -> str:
    parts: List[str] = []
    parts.append(f"head={head or '?'}")

    adj = [w for w, r in mods if r in ("amod", "amod_participle")]
    if adj:
        parts.append(f"adj=[{','.join(adj)}]")

    if verbs:
        v_strs = []
        for v in verbs:
            s = v["verb"]
            if v.get("obj"): s += f"->{v['obj']}"
            v_strs.append(s)
        parts.append(f"verb=[{','.join(v_strs)}]")

    if preps:
        p_strs = [f"{p['prep']}({p['obj'] or '?'})" for p in preps]
        parts.append(f"prep=[{','.join(p_strs)}]")

    return "  ".join(parts)
