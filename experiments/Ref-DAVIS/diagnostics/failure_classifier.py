"""
failure_classifier.py
---------------------
Rule-based classifier that maps per-sequence metric results to failure modes.

Failure modes
-------------
  LOST_TRACK      : IoU drops to ~0 for ≥3 consecutive frames after initially tracking
  NEVER_FOUND     : IoU ≈ 0 from frame 1 (model never localized the object)
  PARTIAL_TRACK   : Mean IoU 0.1–0.4 (partially correct but imprecise)
  IDENTITY_SWAP   : Sudden spatial jump in attention centroid mid-sequence
  OCCLUSION_FAIL  : IoU drops sharply during known/inferred occlusion
  TEMPORAL_COLLAPSE: Attention concentrates on 1-2 frames regardless of generation step
  ATTENTION_DRIFT : Attention centroid does not follow the object trajectory
  SCALE_FAILURE   : IoU inversely correlated with object scale change
  UNSTABLE        : High J-variance with no clear directional failure
  SUCCESS         : Mean IoU > 0.5 and no major failure flags

Each sequence gets a primary failure mode + a list of secondary flags.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np


class FailureMode(str, Enum):
    SUCCESS = "SUCCESS"
    NEVER_FOUND = "NEVER_FOUND"
    LOST_TRACK = "LOST_TRACK"
    PARTIAL_TRACK = "PARTIAL_TRACK"
    IDENTITY_SWAP = "IDENTITY_SWAP"
    OCCLUSION_FAIL = "OCCLUSION_FAIL"
    TEMPORAL_COLLAPSE = "TEMPORAL_COLLAPSE"
    ATTENTION_DRIFT = "ATTENTION_DRIFT"
    SCALE_FAILURE = "SCALE_FAILURE"
    UNSTABLE = "UNSTABLE"


@dataclass
class FailureResult:
    seq_name: str
    exp_id: str
    expression: str
    primary_failure: FailureMode
    secondary_flags: List[FailureMode] = field(default_factory=list)
    # Diagnostic scores
    mean_J: float = 0.0
    J_decay: float = 0.0
    J_variance: float = 0.0
    collapse_rate: float = 0.0
    mean_drift_error: Optional[float] = None
    mean_overlap: Optional[float] = None
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "seq_name": self.seq_name,
            "exp_id": self.exp_id,
            "expression": self.expression,
            "primary_failure": self.primary_failure.value,
            "secondary_flags": [f.value for f in self.secondary_flags],
            "mean_J": self.mean_J,
            "J_decay": self.J_decay,
            "J_variance": self.J_variance,
            "collapse_rate": self.collapse_rate,
            "mean_drift_error": self.mean_drift_error,
            "mean_overlap": self.mean_overlap,
            "notes": self.notes,
        }


class FailureClassifier:
    """
    Classifies a sequence into failure modes given metric results
    and optional TAM diagnostic results.

    Parameters
    ----------
    j_lost_threshold : J below this for 3+ consecutive frames = LOST_TRACK
    j_never_threshold: Mean J below this from start = NEVER_FOUND
    j_partial_max    : Mean J in this range = PARTIAL_TRACK
    collapse_threshold: collapse_rate above this = TEMPORAL_COLLAPSE flag
    drift_threshold  : mean_drift_error above this (in TAM px) = ATTENTION_DRIFT
    variance_threshold: J_variance above this = UNSTABLE
    """

    def __init__(
        self,
        j_lost_threshold: float = 0.1,
        j_never_threshold: float = 0.05,
        j_partial_max: float = 0.4,
        collapse_threshold: float = 0.5,
        drift_threshold: float = 5.0,
        variance_threshold: float = 0.15,
        jump_threshold: float = 8.0,
    ):
        self.j_lost_threshold = j_lost_threshold
        self.j_never_threshold = j_never_threshold
        self.j_partial_max = j_partial_max
        self.collapse_threshold = collapse_threshold
        self.drift_threshold = drift_threshold
        self.variance_threshold = variance_threshold
        self.jump_threshold = jump_threshold

    def _consecutive_low(self, j_per_frame: List[float], threshold: float, min_run: int = 3) -> bool:
        """True if there are min_run consecutive frames with J < threshold."""
        run = 0
        for j in j_per_frame:
            if j < threshold:
                run += 1
                if run >= min_run:
                    return True
            else:
                run = 0
        return False

    def _centroid_jump(self, drift_result: Optional[dict]) -> bool:
        """True if attention centroid has a sudden large jump mid-sequence."""
        if drift_result is None or "centroid_velocity" not in drift_result:
            return False
        v = drift_result["centroid_velocity"]
        if len(v) == 0:
            return False
        valid = v[~np.isnan(v)]
        if len(valid) == 0:
            return False
        mean_v = valid.mean()
        return bool((valid > mean_v * 5).any() and (valid > self.jump_threshold).any())

    def classify(
        self,
        seq_name: str,
        exp_id: str,
        expression: str,
        metrics: dict,
        tam_collapse: Optional[dict] = None,
        tam_drift: Optional[dict] = None,
        tam_identity: Optional[dict] = None,
    ) -> FailureResult:
        """
        Parameters
        ----------
        metrics : dict from metrics.compute_sequence_metrics()
        tam_collapse : dict from tam_analyzer.temporal_collapse() or None
        tam_drift    : dict from tam_analyzer.attention_drift() or None
        tam_identity : dict from tam_analyzer.identity_confusion() or None
        """
        j_per_frame = np.array(metrics["J_per_frame"])
        mean_J = metrics["mean_J"]
        J_decay = metrics["J_decay"]
        J_var = metrics["J_variance"]

        flags: List[FailureMode] = []
        notes_parts = []

        # ── Pull TAM scalars ────────────────────────────────────────────────
        collapse_rate = tam_collapse.get("collapse_rate", 0.0) if tam_collapse else 0.0
        mean_drift_err = tam_drift.get("mean_drift_error") if tam_drift else None
        mean_overlap = tam_identity.get("mean_overlap") if tam_identity else None

        # ── Primary failure detection (in priority order) ────────────────────

        # 1. Never found (J ≈ 0 from start)
        first_third = j_per_frame[:max(1, len(j_per_frame) // 3)]
        if first_third.mean() < self.j_never_threshold and mean_J < 0.1:
            primary = FailureMode.NEVER_FOUND
            notes_parts.append(f"First-third mean J={first_third.mean():.3f}")

        # 2. Lost track (initially good, then falls)
        elif (
            j_per_frame[0] > 0.3
            and self._consecutive_low(j_per_frame.tolist(), self.j_lost_threshold)
        ):
            primary = FailureMode.LOST_TRACK
            notes_parts.append(f"J[0]={j_per_frame[0]:.2f} then dropped")

        # 3. Temporal collapse (TAM-based)
        elif collapse_rate > self.collapse_threshold:
            primary = FailureMode.TEMPORAL_COLLAPSE
            notes_parts.append(f"collapse_rate={collapse_rate:.2f}")

        # 4. Identity swap (centroid jump)
        elif self._centroid_jump(tam_drift):
            primary = FailureMode.IDENTITY_SWAP
            if tam_drift:
                v = tam_drift["centroid_velocity"]
                valid = v[~np.isnan(v)]
                notes_parts.append(f"max_velocity={valid.max():.1f}px")

        # 5. Partial tracking
        elif self.j_never_threshold <= mean_J < self.j_partial_max:
            primary = FailureMode.PARTIAL_TRACK
            notes_parts.append(f"mean_J={mean_J:.3f}")

        # 6. Attention drift (TAM-based, object found but attention wanders)
        elif mean_drift_err is not None and mean_drift_err > self.drift_threshold:
            primary = FailureMode.ATTENTION_DRIFT
            notes_parts.append(f"drift_error={mean_drift_err:.1f}px")

        # 7. Unstable (high variance, no directional failure)
        elif J_var > self.variance_threshold:
            primary = FailureMode.UNSTABLE
            notes_parts.append(f"J_var={J_var:.3f}")

        # 8. Success
        elif mean_J >= 0.5:
            primary = FailureMode.SUCCESS

        else:
            primary = FailureMode.PARTIAL_TRACK
            notes_parts.append(f"mean_J={mean_J:.3f} (catch-all)")

        # ── Secondary flags ──────────────────────────────────────────────────
        if primary != FailureMode.TEMPORAL_COLLAPSE and collapse_rate > self.collapse_threshold:
            flags.append(FailureMode.TEMPORAL_COLLAPSE)
        if primary != FailureMode.ATTENTION_DRIFT and mean_drift_err and mean_drift_err > self.drift_threshold:
            flags.append(FailureMode.ATTENTION_DRIFT)
        if mean_overlap is not None and mean_overlap > 0.3:
            flags.append(FailureMode.IDENTITY_SWAP)
            notes_parts.append(f"attn_overlap={mean_overlap:.2f}")
        if J_decay < -0.3:
            flags.append(FailureMode.LOST_TRACK)
            notes_parts.append(f"J_decay={J_decay:.3f}")
        if J_var > self.variance_threshold and primary not in (FailureMode.UNSTABLE,):
            flags.append(FailureMode.UNSTABLE)

        return FailureResult(
            seq_name=seq_name,
            exp_id=exp_id,
            expression=expression,
            primary_failure=primary,
            secondary_flags=flags,
            mean_J=mean_J,
            J_decay=J_decay,
            J_variance=J_var,
            collapse_rate=collapse_rate,
            mean_drift_error=mean_drift_err,
            mean_overlap=mean_overlap,
            notes="; ".join(notes_parts),
        )

    def classify_vot(
        self,
        seq_name: str,
        exp_id: str,
        expression: str,
        metrics: dict,
    ) -> FailureResult:
        """
        VOT equivalent of classify(). Uses iou_per_frame / mean_iou / iou_decay /
        iou_variance in place of J metrics. TAM is not supported for VOT.
        """
        iou_per_frame = np.array(metrics["iou_per_frame"])
        mean_iou = metrics["mean_iou"]
        iou_decay = metrics["iou_decay"]
        iou_var = metrics["iou_variance"]

        flags: List[FailureMode] = []
        notes_parts = []

        first_third = iou_per_frame[:max(1, len(iou_per_frame) // 3)]
        if first_third.mean() < self.j_never_threshold and mean_iou < 0.1:
            primary = FailureMode.NEVER_FOUND
            notes_parts.append(f"First-third mean IoU={first_third.mean():.3f}")
        elif (
            iou_per_frame[0] > 0.3
            and self._consecutive_low(iou_per_frame.tolist(), self.j_lost_threshold)
        ):
            primary = FailureMode.LOST_TRACK
            notes_parts.append(f"IoU[0]={iou_per_frame[0]:.2f} then dropped")
        elif self.j_never_threshold <= mean_iou < self.j_partial_max:
            primary = FailureMode.PARTIAL_TRACK
            notes_parts.append(f"mean_IoU={mean_iou:.3f}")
        elif iou_var > self.variance_threshold:
            primary = FailureMode.UNSTABLE
            notes_parts.append(f"iou_var={iou_var:.3f}")
        elif mean_iou >= 0.5:
            primary = FailureMode.SUCCESS
        else:
            primary = FailureMode.PARTIAL_TRACK
            notes_parts.append(f"mean_IoU={mean_iou:.3f} (catch-all)")

        if iou_decay < -0.3:
            flags.append(FailureMode.LOST_TRACK)
            notes_parts.append(f"iou_decay={iou_decay:.3f}")
        if iou_var > self.variance_threshold and primary != FailureMode.UNSTABLE:
            flags.append(FailureMode.UNSTABLE)

        return FailureResult(
            seq_name=seq_name,
            exp_id=exp_id,
            expression=expression,
            primary_failure=primary,
            secondary_flags=flags,
            mean_J=mean_iou,
            J_decay=iou_decay,
            J_variance=iou_var,
            notes="; ".join(notes_parts),
        )

    def summarize(self, results: List[FailureResult]) -> dict:
        """Aggregate failure mode distribution across all sequences."""
        total = len(results)
        if total == 0:
            return {}
        counts = {}
        for mode in FailureMode:
            n = sum(1 for r in results if r.primary_failure == mode)
            counts[mode.value] = {"count": n, "pct": 100 * n / total}

        # Secondary flag distribution
        secondary_counts = {}
        for mode in FailureMode:
            n = sum(1 for r in results if mode in r.secondary_flags)
            if n > 0:
                secondary_counts[mode.value] = n

        return {
            "total": total,
            "primary_distribution": counts,
            "secondary_distribution": secondary_counts,
            "success_rate": counts.get("SUCCESS", {}).get("pct", 0.0),
        }
