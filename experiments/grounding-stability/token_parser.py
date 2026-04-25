"""
token_parser.py
---------------
Parses the VOT model output to find which generated tokens correspond to
each frame's label value, so TAM heatmaps can be attributed per frame.
"""

import json
import re
from typing import List, Tuple


def parse_frame_labels(
    gen_text: str,
    fps: float = 24.0,
    sample_rate: int = 1,
) -> List[Tuple[int, str]]:
    """
    Parse the model's JSON output into (sampled_frame_idx, label_str) pairs
    in generation order.

    Handles two output formats:
      Video mode:  [{"frame": 0, "bbox_2d": [...], "label": "swan"}, ...]
      Image mode:  [{"time": 0.33, "bbox_2d": [...], "label": "swan"}, ...]

    For time-based entries, sampled_frame_idx = round(time * fps / sample_rate).

    Returns an empty list on parse failure.
    """
    text = gen_text.strip()
    text = re.sub(r"```[a-z]*", "", text).strip().strip("`")
    try:
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            entries = json.loads(text[start:end])
            if isinstance(entries, list):
                result = []
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    label = entry.get("label", "")
                    if not label:
                        continue
                    if "frame" in entry:
                        frame_idx = int(entry["frame"])
                    elif "time" in entry:
                        t = float(entry["time"])
                        frame_idx = int(round(t * fps / sample_rate))
                    else:
                        continue
                    if frame_idx >= 0:
                        result.append((frame_idx, str(label).strip()))
                return result
    except Exception:
        pass
    return []


def find_label_token_indices(
    gen_tokens: List[str],
    parsed_entries: List[Tuple[int, str]],
) -> List[Tuple[int, List[int]]]:
    """
    For each (frame_idx, label_str) in parsed_entries (in generation order),
    find all gen_token indices whose text falls between the opening and closing
    quote of the label value in the concatenated token stream.

    Uses quote-boundary detection rather than label string length to handle
    tokenizer space-prefix quirks (e.g. " swan" vs "swan").

    Returns: [(frame_idx, [tok_idx, ...]), ...]
    Frames whose label cannot be located get an empty index list.
    """
    # Concatenate tokens to build a searchable string + per-token char spans
    raw = "".join(gen_tokens)
    spans: List[Tuple[int, int]] = []
    pos = 0
    for tok in gen_tokens:
        spans.append((pos, pos + len(tok)))
        pos += len(tok)

    results: List[Tuple[int, List[int]]] = []
    search_from = 0

    for frame_idx, label_str in parsed_entries:
        key_pos = raw.find('"label"', search_from)
        if key_pos == -1:
            results.append((frame_idx, []))
            continue

        colon_pos = raw.find(":", key_pos + 7)
        if colon_pos == -1:
            results.append((frame_idx, []))
            continue

        open_quote = raw.find('"', colon_pos + 1)
        if open_quote == -1:
            results.append((frame_idx, []))
            continue

        close_quote = raw.find('"', open_quote + 1)
        if close_quote == -1:
            results.append((frame_idx, []))
            continue

        val_start = open_quote + 1
        val_end = close_quote

        # Collect every token that overlaps the value region
        token_idxs = [
            i for i, (s, e) in enumerate(spans)
            if s < val_end and e > val_start
        ]

        results.append((frame_idx, token_idxs))
        # Advance past this "label" so the next search finds the next entry
        search_from = key_pos + 7

    return results
