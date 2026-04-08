"""
Temporal Analysis Module for Video Activation Maps

Computes temporal coherence metrics on per-frame activation maps
to assess whether MLLMs genuinely track objects across video frames.

Usage:
    from temporal_analysis import compute_temporal_coherence, evaluate_tracking

    # per_frame_maps: list of np.ndarray, each shape (H, W)
    # (extracted from TAM's video pipeline for a specific token)
    metrics = compute_temporal_coherence(per_frame_maps)

    # With ground truth trajectory
    tracking_metrics = evaluate_tracking(per_frame_maps, gt_trajectory, frame_size=(448, 448))
"""

import json
import numpy as np
from pathlib import Path


def compute_temporal_coherence(per_frame_maps):
    """
    Compute temporal coherence metrics for per-frame activation maps of a single token.

    Args:
        per_frame_maps: list of np.ndarray, each shape (H, W).
            Activation maps for the same token across video frames.

    Returns:
        dict with:
            - map_consistency: mean cosine similarity between consecutive maps
            - map_consistency_per_frame: list of per-pair cosine similarities
            - peak_trajectory: list of (row, col) peak locations per frame
            - spatial_smoothness: mean Euclidean displacement between consecutive peaks
            - displacement_per_frame: list of per-pair displacements
            - intensity_stability: std of peak activation values
            - peak_values: list of peak activation values per frame
    """
    n_frames = len(per_frame_maps)
    if n_frames < 2:
        raise ValueError("Need at least 2 frames for temporal analysis")

    # 1. Map consistency: cosine similarity between consecutive maps
    cos_sims = []
    for i in range(n_frames - 1):
        flat_a = per_frame_maps[i].flatten().astype(np.float64)
        flat_b = per_frame_maps[i + 1].flatten().astype(np.float64)
        norm_a = np.linalg.norm(flat_a)
        norm_b = np.linalg.norm(flat_b)
        if norm_a > 0 and norm_b > 0:
            sim = np.dot(flat_a, flat_b) / (norm_a * norm_b)
        else:
            sim = 0.0
        cos_sims.append(float(sim))

    # 2. Peak trajectory: argmax location per frame
    peaks = []
    peak_values = []
    for m in per_frame_maps:
        idx = np.unravel_index(np.argmax(m), m.shape)
        peaks.append((int(idx[0]), int(idx[1])))
        peak_values.append(float(m[idx]))

    # 3. Spatial smoothness: displacement between consecutive peaks
    displacements = []
    for i in range(len(peaks) - 1):
        dy = peaks[i + 1][0] - peaks[i][0]
        dx = peaks[i + 1][1] - peaks[i][1]
        displacements.append(float(np.sqrt(dy ** 2 + dx ** 2)))

    return {
        'map_consistency': float(np.mean(cos_sims)),
        'map_consistency_per_frame': cos_sims,
        'peak_trajectory': peaks,
        'spatial_smoothness': float(np.mean(displacements)),
        'displacement_per_frame': displacements,
        'intensity_stability': float(np.std(peak_values)),
        'peak_values': peak_values,
    }


def evaluate_tracking(per_frame_maps, gt_trajectory, map_shape=None):
    """
    Evaluate tracking quality against ground truth trajectory.

    Args:
        per_frame_maps: list of np.ndarray, each shape (H, W)
        gt_trajectory: list of dicts with keys:
            - frame_idx (int)
            - x (int): pixel x coordinate in original image
            - y (int): pixel y coordinate in original image
            - visible (bool): whether the object is visible
        map_shape: (map_H, map_W) if different from image size.
            Used to scale ground truth coordinates to activation map coordinates.

    Returns:
        dict with:
            - position_accuracy: mean distance between predicted and GT peaks (in map pixels)
            - position_accuracy_visible: same but only for visible frames
            - occlusion_recovery: cosine sim between pre-occlusion and post-occlusion maps
            - temporal_coherence: full coherence metrics from compute_temporal_coherence()
    """
    coherence = compute_temporal_coherence(per_frame_maps)

    map_h, map_w = per_frame_maps[0].shape

    # Build GT lookup
    gt_by_frame = {entry['frame_idx']: entry for entry in gt_trajectory}

    # Compute position accuracy
    distances = []
    distances_visible = []
    for i, m in enumerate(per_frame_maps):
        if i not in gt_by_frame:
            continue
        gt = gt_by_frame[i]

        # Scale GT coords to map coords if needed
        if map_shape:
            orig_h, orig_w = map_shape
        else:
            orig_h, orig_w = map_h, map_w

        gt_row = int(gt['y'] * map_h / orig_h)
        gt_col = int(gt['x'] * map_w / orig_w)

        # Predicted peak
        pred_row, pred_col = np.unravel_index(np.argmax(m), m.shape)

        dist = np.sqrt((pred_row - gt_row) ** 2 + (pred_col - gt_col) ** 2)
        distances.append(float(dist))

        if gt.get('visible', True):
            distances_visible.append(float(dist))

    # Occlusion recovery: compare maps before and after occlusion
    occlusion_recovery = None
    occluded_frames = [e['frame_idx'] for e in gt_trajectory if not e.get('visible', True)]
    if occluded_frames:
        first_occluded = min(occluded_frames)
        last_occluded = max(occluded_frames)
        pre = first_occluded - 1
        post = last_occluded + 1
        if pre >= 0 and post < len(per_frame_maps):
            flat_pre = per_frame_maps[pre].flatten().astype(np.float64)
            flat_post = per_frame_maps[post].flatten().astype(np.float64)
            norm_pre = np.linalg.norm(flat_pre)
            norm_post = np.linalg.norm(flat_post)
            if norm_pre > 0 and norm_post > 0:
                occlusion_recovery = float(np.dot(flat_pre, flat_post) / (norm_pre * norm_post))

    return {
        'position_accuracy': float(np.mean(distances)) if distances else None,
        'position_accuracy_visible': float(np.mean(distances_visible)) if distances_visible else None,
        'occlusion_recovery': occlusion_recovery,
        'temporal_coherence': coherence,
    }


def load_trajectory(trajectory_path):
    """Load ground truth trajectory from JSON file."""
    with open(trajectory_path) as f:
        return json.load(f)


def print_report(metrics, title="Temporal Analysis Report"):
    """Print a readable report of temporal coherence metrics."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

    tc = metrics.get('temporal_coherence', metrics)

    print(f"\n  Map Consistency (cosine sim):  {tc['map_consistency']:.4f}")
    print(f"  Spatial Smoothness (px disp):  {tc['spatial_smoothness']:.2f}")
    print(f"  Intensity Stability (std):     {tc['intensity_stability']:.4f}")
    print(f"  Peak Trajectory:               {tc['peak_trajectory']}")

    if 'position_accuracy' in metrics and metrics['position_accuracy'] is not None:
        print(f"\n  Position Accuracy (all):       {metrics['position_accuracy']:.2f} px")
        print(f"  Position Accuracy (visible):   {metrics['position_accuracy_visible']:.2f} px")

    if 'occlusion_recovery' in metrics and metrics['occlusion_recovery'] is not None:
        print(f"  Occlusion Recovery (cos sim):  {metrics['occlusion_recovery']:.4f}")

    print(f"\n{'=' * 60}\n")


if __name__ == '__main__':
    # Quick test with dummy data
    np.random.seed(42)

    # Simulate a "tracking" model: peak moves smoothly
    maps_tracking = []
    for i in range(10):
        m = np.zeros((16, 16))
        row = 8
        col = 2 + i  # moves from left to right
        m[max(0, row-1):row+2, max(0, col-1):col+2] = 1.0
        m += np.random.rand(16, 16) * 0.1  # small noise
        maps_tracking.append(m)

    # Simulate a "re-detection" model: peak jumps randomly
    maps_redetect = []
    for i in range(10):
        m = np.zeros((16, 16))
        row = np.random.randint(4, 12)
        col = np.random.randint(4, 12)
        m[max(0, row-1):row+2, max(0, col-1):col+2] = 1.0
        m += np.random.rand(16, 16) * 0.1
        maps_redetect.append(m)

    print_report(compute_temporal_coherence(maps_tracking), "Simulated TRACKING Model")
    print_report(compute_temporal_coherence(maps_redetect), "Simulated RE-DETECTION Model")
