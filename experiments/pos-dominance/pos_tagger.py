"""
pos_tagger.py
-------------
POS-aware labeling for expression content words.

Splits the former monolithic 'label_noun' bucket into finer POS categories:
  label_noun      NOUN, PROPN
  label_adj       ADJ
  label_verb      VERB, AUX, PART
  label_adv       ADV
  label_other_pos everything else that passed the content-word filter

Uses spaCy (en_core_web_sm) → NLTK averaged_perceptron_tagger → heuristic
fallback (returns 'label_other_pos' for everything).
"""

import re
from typing import Dict

# spaCy pos_ tag → experiment category
_SPACY_MAP: Dict[str, str] = {
    "NOUN":  "label_noun",
    "PROPN": "label_noun",
    "ADJ":   "label_adj",
    "VERB":  "label_verb",
    "AUX":   "label_verb",
    "PART":  "label_verb",
    "ADV":   "label_adv",
}

# NLTK Penn-Treebank tag → experiment category
_NLTK_MAP: Dict[str, str] = {
    "NN": "label_noun", "NNS": "label_noun",
    "NNP": "label_noun", "NNPS": "label_noun",
    "JJ": "label_adj", "JJR": "label_adj", "JJS": "label_adj",
    "VB": "label_verb", "VBD": "label_verb", "VBG": "label_verb",
    "VBN": "label_verb", "VBP": "label_verb", "VBZ": "label_verb",
    "RB": "label_adv",  "RBR": "label_adv",  "RBS": "label_adv",
}

_tagger_used: str = "none"


def build_pos_map(expression: str) -> Dict[str, str]:
    """
    Return word → POS-category for every alphabetic word in *expression*.
    Words whose POS cannot be determined get 'label_other_pos'.
    """
    global _tagger_used

    # ── spaCy ────────────────────────────────────────────────────────────────
    try:
        import spacy  # type: ignore
        for model_name in ("en_core_web_sm", "en_core_web_md", "en_core_web_lg"):
            try:
                nlp = spacy.load(model_name)
                break
            except OSError:
                continue
        else:
            raise ImportError("no spaCy model found")
        doc = nlp(expression)
        pos_map: Dict[str, str] = {
            tok.text.lower(): _SPACY_MAP.get(tok.pos_, "label_other_pos")
            for tok in doc
        }
        _tagger_used = f"spaCy:{model_name}"
        return pos_map
    except Exception:
        pass

    # ── NLTK ─────────────────────────────────────────────────────────────────
    try:
        import nltk  # type: ignore
        # download tokeniser and tagger data silently on first use
        for resource in ("punkt_tab", "averaged_perceptron_tagger_eng"):
            try:
                nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource
                               else f"taggers/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)
        from nltk import pos_tag, word_tokenize
        tagged = pos_tag(word_tokenize(expression))
        pos_map = {word.lower(): _NLTK_MAP.get(tag, "label_other_pos")
                   for word, tag in tagged}
        _tagger_used = "NLTK"
        return pos_map
    except Exception:
        pass

    # ── fallback ─────────────────────────────────────────────────────────────
    words = re.findall(r"[a-zA-Z]+", expression.lower())
    _tagger_used = "fallback"
    return {w: "label_other_pos" for w in words}


def token_to_pos_category(token_clean: str, pos_map: Dict[str, str]) -> str:
    """
    Map a cleaned (BPE-stripped, lowercase) token to its POS category.

    Exact match first; then prefix matching for subword tokens (min 3 chars).
    Falls back to 'label_other_pos' when no match is found.
    """
    w = token_clean.lower()
    if w in pos_map:
        return pos_map[w]
    if len(w) >= 3:
        for word, cat in pos_map.items():
            if word.startswith(w) or w.startswith(word):
                return cat
    return "label_other_pos"


def tagger_info() -> str:
    """Return a human-readable string describing which tagger was used."""
    return _tagger_used
