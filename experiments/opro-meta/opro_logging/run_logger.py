"""
opro_logging/run_logger.py
--------------------------
Append-only structured logger. One JSONL line per recorded event.

Events emitted by the inner/outer loops:
  run_start             {run_id, config, time}
  clip_start            {seq, obj, seed_expression, iteration_budget}
  candidate_proposed    {seq, obj, iter, expression, rationale}
  candidate_evaluated   {seq, obj, iter, expression, mean_iou, mass_in_gt,
                         mass_in_pred, token_category_breakdown}
  clip_done             {seq, obj, best_expression, best_iou, history_len}
  meta_prompt           {version, prompt_text, source_clips}
  run_end               {run_id, time}
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class RunLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", buffering=1)  # line-buffered

    def log(self, event: str, **fields: Any) -> None:
        rec = {"ts": time.time(), "event": event, **fields}
        self._fh.write(json.dumps(rec, default=str) + "\n")

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_log(path: str | Path) -> list[dict]:
    """Load every record in a JSONL log."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out
