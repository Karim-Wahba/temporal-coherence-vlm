"""
_paths.py
---------
Adds sibling experiment folders to sys.path so opro-meta modules can import
existing infrastructure (DAVISVOTLoader, QwenVOTRunner, token_parser, metrics).
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_EXPS = _HERE.parent
_REF_DAVIS = _EXPS / "Ref-DAVIS"
_GS_MAX    = _EXPS / "grounding-stability-max"
_POSDOM    = _EXPS / "pos-dominance"

for p in [_REF_DAVIS, _REF_DAVIS / "benchmark", _GS_MAX, _POSDOM]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
