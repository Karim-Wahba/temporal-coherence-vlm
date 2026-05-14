"""
main.py
-------
Orchestrator for opro-meta.

Flow:
  1. Load YAML config.
  2. Load Ref-DAVIS clips.
  3. Build the grounder (Qwen3-VL) and the optimizer LLM client (Qwen3 text /
     Anthropic / OpenAI).
  4. For each clip: run the inner OPRO loop, log every step.
  5. If outer_loop.enabled: distill a meta-prompt from the trajectories.
  6. If evaluation.enabled: run the meta-prompt against held-out clips.

Usage:
  python main.py --config configs/default.yaml
  python main.py --config configs/default.yaml --max_clips 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import _paths  # noqa: F401  -- side-effect: extends sys.path

import yaml

from data.ref_davis_loader import RefDavisClipLoader
from grounding.qwen3vl_runner import Qwen3VLGrounder
from opro_metrics.grounding_metrics import compute_metrics  # noqa: F401  (ensures import path)
from optimizer.inner_loop import InnerLoopConfig, run_clip
from optimizer.llm_client import make_client
from optimizer.outer_loop import distill_meta_prompt, evaluate_meta_prompt
from opro_logging.run_logger import RunLogger


def _load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _maybe_override(cfg: dict, args: argparse.Namespace) -> dict:
    if args.max_clips is not None:
        cfg["data"]["max_clips"] = args.max_clips
    if args.out_dir is not None:
        cfg["run"]["out_dir"] = args.out_dir
    if args.backend is not None:
        cfg["llm"]["backend"] = args.backend
        # By default, --backend applies to BOTH inner and meta. Pass
        # --meta_backend explicitly to keep them different.
        if args.meta_backend is None:
            cfg.setdefault("meta_llm", {})
            cfg["meta_llm"]["backend"] = args.backend
    if args.meta_backend is not None:
        cfg.setdefault("meta_llm", {})
        cfg["meta_llm"]["backend"] = args.meta_backend
    if args.skip_tam:
        cfg["grounder"]["skip_tam"] = True
    return cfg


def _llm_config_for(role_cfg: dict) -> tuple[str, dict]:
    """Return (backend, kwargs) for a single LLM role block."""
    backend = role_cfg.get("backend")
    if not backend:
        return None, {}
    kwargs = dict(role_cfg.get(backend, {}) or {})
    return backend, kwargs


def _configs_equivalent(a: dict, b: dict) -> bool:
    ba, ka = _llm_config_for(a)
    bb, kb = _llm_config_for(b)
    if not ba or not bb:
        return False
    if ba != bb:
        return False
    # Same backend → equivalent if model_id matches (everything else is per-call adjustable)
    return ka.get("model_id") == kb.get("model_id")


def _build_role_llm(role_cfg: dict, role_name: str):
    backend, kwargs = _llm_config_for(role_cfg)
    if not backend:
        return None
    print(f"[{role_name}] backend={backend}  model_id={kwargs.get('model_id', '<default>')}")
    return make_client(backend, **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",   default="configs/default.yaml")
    ap.add_argument("--max_clips", type=int, default=None)
    ap.add_argument("--out_dir",  default=None)
    ap.add_argument("--backend",  default=None,
                    help="Override the LLM backend. Applies to BOTH inner and "
                         "meta unless --meta_backend is also given.")
    ap.add_argument("--meta_backend", default=None,
                    help="Override meta_llm.backend independently of --backend.")
    ap.add_argument("--skip_tam", action="store_true",
                    help="Skip TAM extraction in the grounder. Drops MassGT/MassPred "
                         "but roughly halves per-iteration wall time.")
    args = ap.parse_args()

    cfg = _maybe_override(_load_config(args.config), args)
    out_dir = Path(cfg["run"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"opro_{time.strftime('%Y%m%d_%H%M%S')}"
    (out_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    logger = RunLogger(out_dir / f"{run_id}.jsonl")
    logger.log("run_start", run_id=run_id, config=cfg)

    print("=" * 70)
    print(f"opro-meta run {run_id}")
    print(f"output dir: {out_dir}")
    print("=" * 70)

    # ── Data ────────────────────────────────────────────────────────────────
    loader = RefDavisClipLoader(
        davis_root=cfg["data"]["davis_root"],
        split=cfg["data"]["split"],
        sequences=cfg["data"]["sequences"],
        max_clips=cfg["data"]["max_clips"],
    )
    clips = list(loader)
    print(f"\nLoaded {len(clips)} clip(s)")
    for c in clips:
        print(f"  - {c.seq_name} obj{c.obj_id}  \"{c.seed_expression}\"")

    if not clips:
        sys.exit("No clips matched the loader filter.")

    # ── Grounder ────────────────────────────────────────────────────────────
    gcfg = cfg["grounder"]
    grounder = Qwen3VLGrounder(
        model_id=gcfg["model_id"],
        sample_rate=gcfg["sample_rate"],
        max_new_tokens=gcfg["max_new_tokens"],
        video_mode=gcfg["video_mode"],
        seed=cfg["run"]["seed"],
        skip_tam=gcfg.get("skip_tam", False),
    )

    # ── LLM clients (inner + meta) ──────────────────────────────────────────
    print()
    inner_llm = _build_role_llm(cfg["llm"], role_name="inner-llm")

    meta_cfg = cfg.get("meta_llm") or {}
    if not meta_cfg.get("backend"):
        meta_llm = inner_llm
        print("[meta-llm]  reusing inner-llm (meta_llm.backend not configured)")
    elif _configs_equivalent(cfg["llm"], meta_cfg):
        meta_llm = inner_llm
        print("[meta-llm]  reusing inner-llm instance (same backend + model_id)")
    else:
        meta_llm = _build_role_llm(meta_cfg, role_name="meta-llm")

    # ── Inner loop ──────────────────────────────────────────────────────────
    inner_cfg = InnerLoopConfig(
        n_candidates=cfg["inner_loop"]["n_candidates"],
        n_iterations=cfg["inner_loop"]["n_iterations"],
        early_stop_no_improve=cfg["inner_loop"]["early_stop_no_improve"],
        history_top_n=cfg["inner_loop"].get("history_top_n", 5),
        history_bottom_n=cfg["inner_loop"].get("history_bottom_n", 5),
    )
    clip_results = []
    for i, clip in enumerate(clips, 1):
        print(f"\n[{i}/{len(clips)}] {clip.seq_name} obj{clip.obj_id}")
        res = run_clip(clip, grounder, inner_llm, inner_cfg, logger=logger)
        clip_results.append(res)
        # Persist after every clip
        (out_dir / "clip_results.json").write_text(
            json.dumps([c.to_dict() for c in clip_results], indent=2, default=str)
        )

    # ── Outer loop (meta-prompt distillation) ───────────────────────────────
    if cfg["outer_loop"]["enabled"] and len(clip_results) >= cfg["outer_loop"]["source_clips_min"]:
        print("\n── Distilling meta-prompt ──")
        meta_dir = out_dir / cfg["outer_loop"]["meta_prompt_dir"]
        # Use meta_llm-specific max_new_tokens if set; else outer_loop.meta_max_new_tokens
        meta_kwargs = meta_cfg.get(meta_cfg.get("backend"), {}) if meta_cfg.get("backend") else {}
        meta_max_tokens = (
            meta_kwargs.get("max_new_tokens")
            or meta_kwargs.get("max_tokens")
            or cfg["outer_loop"].get("meta_max_new_tokens", 4096)
        )
        meta = distill_meta_prompt(
            clip_results=clip_results,
            llm_client=meta_llm,
            out_dir=meta_dir,
            max_new_tokens=meta_max_tokens,
            logger=logger,
        )

        # ── Held-out evaluation ────────────────────────────────────────────
        # Uses inner_llm (the meta-distilled prompt is fed to the inner loop).
        if cfg["evaluation"]["enabled"] and meta.get("inner_system_prompt"):
            print("\n── Evaluating meta-prompt on held-out clips ──")
            holdout_loader = RefDavisClipLoader(
                davis_root=cfg["data"]["davis_root"],
                split=cfg["data"]["split"],
                max_clips=cfg["evaluation"]["holdout_clips"]
                          + (cfg["data"]["max_clips"] or 0),
            )
            seen = {(c.seq_name, c.obj_id) for c in clips}
            holdout = [c for c in holdout_loader if (c.seq_name, c.obj_id) not in seen][
                : cfg["evaluation"]["holdout_clips"]
            ]
            evals = evaluate_meta_prompt(
                holdout_clips=holdout,
                grounder=grounder,
                llm_client=inner_llm,
                meta=meta,
                n_iters_for_meta=1,
                n_candidates=cfg["inner_loop"]["n_candidates"],
                logger=logger,
            )
            (out_dir / "meta_eval.json").write_text(
                json.dumps([c.to_dict() for c in evals], indent=2, default=str)
            )

    logger.log("run_end", run_id=run_id)
    logger.close()
    print(f"\nDone. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
