"""
tam_analyzer.py
---------------
Diagnostic experiments on TAM attention maps for temporal coherence analysis.

Each function takes the output dict from TAMRunner.run() plus optional GT data
and returns a structured result dict + any per-experiment scalar metrics.

Experiments
-----------
  1. attention_drift()           - centroid of attention vs GT object centroid
  2. temporal_collapse()         - does attention ignore most frames?
  3. identity_confusion()        - (multi-call) do two objects share attention?
  4. occlusion_recovery()        - attention before vs after occlusion
  5. prompt_temporal_binding()   - does attention shift when prompt changes?
  6. pearson_temporal_coherence() - Pearson-corr coherence for the first noun
"""

import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


# ─── Utilities ───────────────────────────────────────────────────────────────

def _map_centroid(frame_map: np.ndarray) -> Tuple[float, float]:
    """
    Compute (cx, cy) centroid of a 2D attention map.
    Returns (nan, nan) if map is empty.
    """
    total = frame_map.sum()
    if total < 1e-8:
        return (float("nan"), float("nan"))
    h, w = frame_map.shape
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cx = float((xs * frame_map).sum() / total)
    cy = float((ys * frame_map).sum() / total)
    return cx, cy


def _mask_centroid(mask: np.ndarray) -> Tuple[float, float]:
    """Centroid of a binary mask."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return (float("nan"), float("nan"))
    return float(xs.mean()), float(ys.mean())


def _active_tokens(tam_result: dict, skip_special: bool = True) -> List[int]:
    """Indices of generated tokens that are not special tokens."""
    indices = []
    for i, tok in enumerate(tam_result["gen_tokens"]):
        if skip_special and ("<|" in tok or tok.strip() == ""):
            continue
        if tam_result["tam_maps"][i] is not None:
            indices.append(i)
    return indices


def _mean_map_for_tokens(tam_result: dict, token_indices: List[int]) -> Optional[np.ndarray]:
    """Element-wise mean of TAM maps for given token indices. Returns (T,H,W)."""
    maps = [tam_result["tam_maps"][i] for i in token_indices
            if tam_result["tam_maps"][i] is not None]
    if not maps:
        return None
    stacked = np.stack(maps, axis=0)
    return stacked.mean(axis=0)


# ─── Experiment 1: Attention Drift ───────────────────────────────────────────

def attention_drift(
    tam_result: dict,
    gt_masks: Optional[List[np.ndarray]] = None,
    frame_size: Optional[Tuple[int, int]] = None,
) -> dict:
    """
    For each generated token, compute the centroid of TAM attention in
    each frame, yielding a (num_tokens, T, 2) trajectory.

    If gt_masks provided, compare to GT centroid trajectory and compute
    mean centroid displacement error per frame.

    Returns
    -------
    dict:
        "tam_centroids"     : (num_active_tokens, T, 2)  [cx, cy] in TAM coords
        "per_frame_centroid": (T, 2) mean centroid across all active tokens
        "gt_centroids"      : (T, 2) or None
        "displacement_error": (T,) mean L2 error at each frame (if gt provided)
        "mean_drift_error"  : float or None
        "centroid_velocity" : (T-1,) L2 distance between consecutive centroids
        "velocity_variance" : float  (high = erratic/unstable attention)
    """
    T = tam_result["vision_shape"][0]
    H_tam = tam_result["vision_shape"][1]
    W_tam = tam_result["vision_shape"][2]

    active = _active_tokens(tam_result)
    if not active:
        return {"error": "no active tokens"}

    # Per-token, per-frame centroids in TAM space
    tam_centroids = np.full((len(active), T, 2), np.nan)
    for tok_i, i in enumerate(active):
        m = tam_result["tam_maps"][i]
        if m is None or m.shape[0] != T:
            continue
        for t in range(T):
            cx, cy = _map_centroid(m[t].astype(np.float32))
            tam_centroids[tok_i, t] = [cx, cy]

    # Mean centroid across tokens per frame (nanmean)
    per_frame_centroid = np.nanmean(tam_centroids, axis=0)  # (T, 2)

    # GT centroids (rescaled to TAM space)
    gt_centroids = None
    displacement_error = None
    mean_drift_error = None

    if gt_masks is not None and frame_size is not None:
        H_frame, W_frame = frame_size
        scale_x = W_tam / W_frame
        scale_y = H_tam / H_frame

        gt_centroids = np.full((T, 2), np.nan)
        for t in range(min(T, len(gt_masks))):
            cx, cy = _mask_centroid(gt_masks[t])
            gt_centroids[t] = [cx * scale_x, cy * scale_y]

        # L2 error between mean TAM centroid and GT centroid per frame
        diff = per_frame_centroid - gt_centroids  # (T, 2)
        displacement_error = np.sqrt((diff ** 2).sum(axis=1))  # (T,)
        valid = ~np.isnan(displacement_error)
        mean_drift_error = float(displacement_error[valid].mean()) if valid.any() else None

    # Centroid velocity: frame-to-frame movement of mean centroid
    delta = np.diff(per_frame_centroid, axis=0)  # (T-1, 2)
    velocity = np.sqrt((delta ** 2).sum(axis=1))  # (T-1,)
    valid_v = ~np.isnan(velocity)
    velocity_variance = float(velocity[valid_v].var()) if valid_v.any() else 0.0

    return {
        "tam_centroids": tam_centroids,           # (N_tokens, T, 2)
        "per_frame_centroid": per_frame_centroid,  # (T, 2)
        "gt_centroids": gt_centroids,              # (T, 2) or None
        "displacement_error": displacement_error,  # (T,) or None
        "mean_drift_error": mean_drift_error,      # float or None
        "centroid_velocity": velocity,             # (T-1,)
        "velocity_variance": velocity_variance,    # float
    }


# ─── Experiment 2: Temporal Collapse ─────────────────────────────────────────

def temporal_collapse(tam_result: dict, mass_threshold: float = 0.8) -> dict:
    """
    Detects whether the model collapses all attention to one or few frames.

    "Collapse" = more than `mass_threshold` fraction of total attention
    concentrated on a single frame, for most tokens.

    Returns
    -------
    dict:
        "frame_mass"         : (num_tokens, T) fractional mass per frame per token
        "dominant_frame"     : (num_tokens,) index of most-attended frame per token
        "dominant_mass"      : (num_tokens,) mass on that frame
        "collapse_rate"      : float — fraction of tokens where dominant mass > threshold
        "dominant_frame_hist": (T,) how often each frame is dominant
        "is_collapsed"       : bool — True if collapse_rate > 0.5
        "temporal_entropy"   : (num_tokens,) entropy of frame_mass distribution
                               (low entropy = collapsed to few frames)
        "mean_entropy"       : float
    """
    frame_mass = tam_result["frame_mass"]  # (num_tokens, T)
    T = tam_result["vision_shape"][0]
    num_tokens = frame_mass.shape[0]

    active = _active_tokens(tam_result)
    if not active:
        return {"error": "no active tokens"}

    active_mass = frame_mass[active]  # (N_active, T)

    dominant_frame = active_mass.argmax(axis=1)
    dominant_mass = active_mass.max(axis=1)
    collapse_rate = float((dominant_mass > mass_threshold).mean())

    dominant_frame_hist = np.bincount(dominant_frame, minlength=T)

    # Shannon entropy of each token's frame distribution
    eps = 1e-10
    entropy = -np.sum(
        active_mass * np.log(active_mass + eps), axis=1
    )  # (N_active,)
    # Normalise by log(T) so max entropy = 1
    max_entropy = np.log(T) if T > 1 else 1.0
    norm_entropy = entropy / max_entropy

    return {
        "frame_mass": active_mass,
        "dominant_frame": dominant_frame,
        "dominant_mass": dominant_mass,
        "collapse_rate": collapse_rate,
        "dominant_frame_hist": dominant_frame_hist,
        "is_collapsed": collapse_rate > 0.5,
        "temporal_entropy": norm_entropy,
        "mean_entropy": float(norm_entropy.mean()),
    }


# ─── Experiment 3: Identity Confusion ────────────────────────────────────────

def identity_confusion(
    tam_result_obj1: dict,
    tam_result_obj2: dict,
    gt_masks_obj1: Optional[List[np.ndarray]] = None,
    gt_masks_obj2: Optional[List[np.ndarray]] = None,
    frame_size: Optional[Tuple[int, int]] = None,
) -> dict:
    """
    Given TAM results for two expressions on the same video, detect whether
    the model's attention regions overlap (identity confusion).

    Returns
    -------
    dict:
        "cross_attention_overlap": (T,) mean IoU between obj1 and obj2 TAM maps
                                    per frame (high = confusion)
        "mean_overlap"           : float
        "confusion_rate"         : fraction of frames with overlap > 0.3
        "obj1_mean_map"          : (T, H, W) mean TAM map for obj1
        "obj2_mean_map"          : (T, H, W) mean TAM map for obj2
    """
    T = tam_result_obj1["vision_shape"][0]

    active1 = _active_tokens(tam_result_obj1)
    active2 = _active_tokens(tam_result_obj2)

    map1 = _mean_map_for_tokens(tam_result_obj1, active1)  # (T, H, W)
    map2 = _mean_map_for_tokens(tam_result_obj2, active2)

    if map1 is None or map2 is None:
        return {"error": "insufficient TAM maps"}

    # Binarize at 50th percentile to get attention regions
    def binarize(m):
        thresh = np.percentile(m, 75)
        return (m > thresh).astype(np.float32)

    overlap = np.zeros(T)
    for t in range(T):
        b1 = binarize(map1[t])
        b2 = binarize(map2[t])
        inter = (b1 * b2).sum()
        union = np.maximum(b1, b2).sum()
        overlap[t] = inter / (union + 1e-8)

    return {
        "cross_attention_overlap": overlap,
        "mean_overlap": float(overlap.mean()),
        "confusion_rate": float((overlap > 0.3).mean()),
        "obj1_mean_map": map1,
        "obj2_mean_map": map2,
    }


# ─── Experiment 4: Occlusion Recovery ────────────────────────────────────────

def occlusion_recovery(
    tam_result: dict,
    occlusion_frames: List[int],
    gt_masks: Optional[List[np.ndarray]] = None,
    frame_size: Optional[Tuple[int, int]] = None,
) -> dict:
    """
    Compare attention centroid before, during, and after occlusion.

    Parameters
    ----------
    occlusion_frames : list of frame indices where object is occluded

    Returns
    -------
    dict:
        "pre_occ_centroid"   : (2,) mean centroid before occlusion
        "post_occ_centroid"  : (2,) mean centroid after occlusion
        "centroid_jump"      : L2 distance between pre and post centroids
        "during_occ_entropy" : mean temporal entropy during occlusion
        "recovery_error"     : if gt provided, centroid error after occlusion
        "frozen_prior_score" : how similar post-occ attention is to pre-occ
                               (high = model "remembered" last position)
    """
    T = tam_result["vision_shape"][0]
    drift = attention_drift(tam_result, gt_masks, frame_size)
    per_frame_centroid = drift["per_frame_centroid"]  # (T, 2)

    occ_set = set(occlusion_frames)
    pre_frames = [t for t in range(T) if t < min(occ_set, default=0) and t not in occ_set]
    post_frames = [t for t in range(T) if t > max(occ_set, default=T) and t not in occ_set]

    pre_centroid = (
        np.nanmean(per_frame_centroid[pre_frames], axis=0)
        if pre_frames else np.array([np.nan, np.nan])
    )
    post_centroid = (
        np.nanmean(per_frame_centroid[post_frames], axis=0)
        if post_frames else np.array([np.nan, np.nan])
    )

    diff = post_centroid - pre_centroid
    centroid_jump = float(np.sqrt((diff ** 2).sum())) if not np.isnan(diff).any() else None

    # Temporal entropy during occlusion vs outside
    tc = temporal_collapse(tam_result)
    active = _active_tokens(tam_result)
    occ_token_entropies = []
    non_occ_entropies = []
    # Use frame mass to split tokens by which frame they attend most
    for tok_i, i in enumerate(active):
        if tok_i >= len(tc["temporal_entropy"]):
            break
        dom_frame = int(tc["dominant_frame"][tok_i])
        ent = tc["temporal_entropy"][tok_i]
        if dom_frame in occ_set:
            occ_token_entropies.append(ent)
        else:
            non_occ_entropies.append(ent)

    during_occ_entropy = float(np.mean(occ_token_entropies)) if occ_token_entropies else None
    outside_occ_entropy = float(np.mean(non_occ_entropies)) if non_occ_entropies else None

    # Frozen prior: cosine similarity between pre-occ and post-occ mean maps
    frozen_prior_score = None
    active_indices = active
    pre_map = _mean_map_for_tokens(
        tam_result, [i for i in active_indices
                     if i < len(tam_result["tam_maps"])]
    )
    # Simplified: compare centroid similarity
    if centroid_jump is not None and centroid_jump < 5.0:
        frozen_prior_score = 1.0 - centroid_jump / (tam_result["vision_shape"][2])
    elif centroid_jump is not None:
        frozen_prior_score = max(0.0, 1.0 - centroid_jump / tam_result["vision_shape"][2])

    return {
        "pre_occ_centroid": pre_centroid,
        "post_occ_centroid": post_centroid,
        "centroid_jump": centroid_jump,
        "during_occ_entropy": during_occ_entropy,
        "outside_occ_entropy": outside_occ_entropy,
        "frozen_prior_score": frozen_prior_score,
        "pre_frames": pre_frames,
        "post_frames": post_frames,
        "occlusion_frames": list(occ_set),
    }


# ─── Experiment 5: Prompt Temporal Binding ────────────────────────────────────

def prompt_temporal_binding(
    tam_results_by_prompt: Dict[str, dict],
    target_frames_by_prompt: Dict[str, List[int]],
) -> dict:
    """
    Given TAM results for different prompts targeting different temporal regions,
    check whether attention shifts appropriately.

    Parameters
    ----------
    tam_results_by_prompt    : {"beginning": tam_result, "end": tam_result, ...}
    target_frames_by_prompt  : {"beginning": [0,1,2], "end": [10,11,12], ...}

    Returns
    -------
    dict:
        "binding_scores"    : {prompt: float} — fraction of attention mass
                              on target frames
        "mean_binding"      : float — mean binding score across prompts
        "prompt_comparison" : {prompt: dominant_frame} 
        "is_steerable"      : bool — True if binding scores differ significantly
                              across prompts (model responds to temporal language)
    """
    binding_scores = {}
    dominant_frames = {}

    for prompt, tam_result in tam_results_by_prompt.items():
        target_frames = target_frames_by_prompt.get(prompt, [])
        frame_mass = tam_result["frame_mass"]  # (num_tokens, T)
        active = _active_tokens(tam_result)
        if not active or not target_frames:
            binding_scores[prompt] = 0.0
            continue

        active_mass = frame_mass[active]  # (N_active, T)
        target_mass = active_mass[:, target_frames].sum(axis=1)  # (N_active,)
        binding_scores[prompt] = float(target_mass.mean())
        dominant_frames[prompt] = int(active_mass.mean(axis=0).argmax())

    mean_binding = float(np.mean(list(binding_scores.values()))) if binding_scores else 0.0

    # Check steerability: std of binding scores across prompts
    scores_arr = np.array(list(binding_scores.values()))
    is_steerable = bool(scores_arr.std() > 0.1) if len(scores_arr) > 1 else False

    return {
        "binding_scores": binding_scores,
        "mean_binding": mean_binding,
        "dominant_frames": dominant_frames,
        "is_steerable": is_steerable,
        "binding_std": float(scores_arr.std()) if len(scores_arr) > 1 else 0.0,
    }


# ─── Experiment 6: Pearson Temporal Coherence ─────────────────────────────────

def compute_temporal_coherence(float_maps: List[np.ndarray]) -> float:
    """Compute temporal coherence from a list of per-frame attention maps.

    Uses Pearson correlation between consecutive normalised attention maps,
    rescaled from [-1, 1] to [0, 1].

    Args:
        float_maps: Ordered list of 2-D float64 arrays (one per video frame).

    Returns:
        Coherence score in [0, 1].  Higher = more temporally coherent.
        Returns 1.0 for a single frame (trivially coherent).
        Returns 0.0 when all maps are spatially uniform.
    """
    if len(float_maps) < 2:
        return 1.0

    correlations: List[float] = []
    for i in range(len(float_maps) - 1):
        m1 = np.asarray(float_maps[i]).flatten().astype(np.float64)
        m2 = np.asarray(float_maps[i + 1]).flatten().astype(np.float64)
        m1c = m1 - m1.mean()
        m2c = m2 - m2.mean()
        n1 = np.linalg.norm(m1c)
        n2 = np.linalg.norm(m2c)
        if n1 < 1e-8 or n2 < 1e-8:
            correlations.append(0.0)
        else:
            corr = float(np.dot(m1c, m2c) / (n1 * n2))  # Pearson ∈ [-1, 1]
            correlations.append((corr + 1.0) / 2.0)      # rescale to [0, 1]

    return float(np.mean(correlations))


def _group_tokens_into_words(gen_tokens: List[str]) -> List[dict]:
    """Group sub-word tokens into whole words.

    Returns a list of dicts, each with:
        "word"          : str  (stripped merged word)
        "token_indices" : List[int]  (0-based indices into gen_tokens)
    """
    word_groups: List[dict] = []
    current_word = ""
    current_indices: List[int] = []

    for i, raw in enumerate(gen_tokens):
        if "<|" in raw:
            continue
        is_new = (
            len(current_indices) == 0
            or raw.startswith(" ")
            or raw.startswith("\n")
            or (len(raw.strip()) == 1 and not raw.strip().isalnum())
        )
        if is_new and current_indices:
            word_groups.append({"word": current_word.strip(), "token_indices": current_indices})
            current_word = raw
            current_indices = [i]
        else:
            current_word += raw
            current_indices.append(i)

    if current_indices:
        word_groups.append({"word": current_word.strip(), "token_indices": current_indices})

    return [wg for wg in word_groups
            if wg["word"] and not all(c in " \n\t" for c in wg["word"])]


def _find_first_noun_in_groups(word_groups: List[dict]) -> Tuple[Optional[int], Optional[str]]:
    """Return (word_group_index, word) for the first object noun.

    Uses NLTK POS tagging.  Returns (None, None) if NLTK is unavailable or
    no noun is found.
    """
    try:
        import nltk  # noqa: PLC0415
        for resource in ("averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"):
            try:
                nltk.data.find(f"taggers/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)
        words = [wg["word"] for wg in word_groups]
        for j, (word, pos) in enumerate(nltk.pos_tag(words)):
            if pos in ("NN", "NNS", "NNP", "NNPS") and len(word) > 1:
                return j, word
    except Exception:
        pass
    return None, None


def _merge_word_maps(
    word_groups: List[dict],
    tam_maps: List[Optional[np.ndarray]],
) -> List[Optional[np.ndarray]]:
    """For each word, merge TAM maps across its sub-word tokens (element-wise max).

    Returns a list parallel to word_groups; each entry is (T, H, W) float32 or None.
    """
    word_maps: List[Optional[np.ndarray]] = []
    for wg in word_groups:
        maps = [tam_maps[ti] for ti in wg["token_indices"]
                if ti < len(tam_maps) and tam_maps[ti] is not None
                and isinstance(tam_maps[ti], np.ndarray) and tam_maps[ti].ndim == 3]
        if not maps:
            word_maps.append(None)
            continue
        merged = maps[0].astype(np.float32).copy()
        for m in maps[1:]:
            m32 = m.astype(np.float32)
            if m32.shape == merged.shape:
                merged = np.maximum(merged, m32)
            else:
                resized = np.stack([
                    cv2.resize(m32[t], (merged.shape[2], merged.shape[1]))
                    for t in range(m32.shape[0])
                ])
                merged = np.maximum(merged, resized)
        word_maps.append(merged)
    return word_maps


def attention_mass_in_gt(
    coherence_result: dict,
    gt_masks: List[np.ndarray],
    frame_size: Tuple[int, int],
    vision_shape: Tuple[int, int, int],
    lost_track_threshold: float = 0.25,
) -> dict:
    """Measure how much of the model's attention falls inside the GT region per frame.

    For each frame t and each word's merged TAM map, computes:

        GAM(t) = sum(map[t] * gt_mask_tam[t]) / (sum(map[t]) + ε)

    i.e. the fraction of total attention mass inside the GT bounding box.
    GAM = 1 → all attention on the object.  GAM = 0 → completely off-target.

    The mean GAM across content words is computed per frame, giving a temporal
    signal that drops when the model loses track.

    Parameters
    ----------
    coherence_result      : output of ``pearson_temporal_coherence()``
    gt_masks              : list of (H, W) binary uint8 arrays, one per frame
    frame_size            : (H, W) of original frames
    vision_shape          : (T, H_tam, W_tam)
    lost_track_threshold  : GAM below this value counts as "lost" (default 0.25)

    Returns
    -------
    dict:
        "per_frame_gam"       : np.ndarray (T,) — mean GAM across content words per frame
        "target_per_frame_gam": np.ndarray (T,) — GAM for the target noun only
        "word_per_frame_gam"  : dict[str → np.ndarray (T,)] — per-word GAM traces
        "mean_gam"            : float — mean over frames (overall on-target rate)
        "gam_decay"           : float — linear slope (frames/unit); negative = losing track
        "lost_track_rate"     : float — fraction of frames where mean GAM < threshold
        "gt_masks_tam"        : List[np.ndarray (H_tam, W_tam) bool]
    """
    T, H_tam, W_tam = vision_shape
    H_frame, W_frame = frame_size
    word_groups = coherence_result.get("word_groups", [])
    word_maps = coherence_result.get("word_maps", [])
    target_idx = coherence_result.get("target_word_idx", 0)

    STOPWORDS = {
        "the", "and", "for", "are", "was", "has", "its", "this", "that",
        "with", "from", "they", "their", "which", "have", "been", "also",
        "into", "can", "not", "but", "all", "some", "more", "very",
    }

    # Project GT masks to TAM resolution
    gt_masks_tam: List[np.ndarray] = []
    for t in range(T):
        if t < len(gt_masks) and gt_masks[t] is not None and gt_masks[t].sum() > 0:
            resized = cv2.resize(
                gt_masks[t].astype(np.float32),
                (W_tam, H_tam),
                interpolation=cv2.INTER_NEAREST,
            )
            gt_masks_tam.append(resized > 0.5)
        else:
            gt_masks_tam.append(np.ones((H_tam, W_tam), dtype=bool))

    def _gam_for_map(wm: np.ndarray) -> np.ndarray:
        """Return (T,) GAM trace for one word map."""
        gam = np.zeros(T, dtype=np.float64)
        for t in range(min(T, wm.shape[0])):
            m = wm[t].astype(np.float64)
            total = m.sum()
            if total < 1e-8:
                gam[t] = 0.0
            else:
                gam[t] = float(m[gt_masks_tam[t]].sum() / total)
        return gam

    # Compute per-word GAM; only keep content words for the mean
    word_per_frame_gam: Dict[str, np.ndarray] = {}
    content_gams: List[np.ndarray] = []

    for wg, wm in zip(word_groups, word_maps):
        if wm is None or wm.shape[0] < 1:
            continue
        w = wg["word"]
        gam = _gam_for_map(wm)
        word_per_frame_gam[w] = gam
        if (len(w) > 2
                and w.replace("-", "").replace("'", "").isalpha()
                and w.lower() not in STOPWORDS):
            content_gams.append(gam)

    # Mean across content words per frame
    if content_gams:
        per_frame_gam = np.mean(content_gams, axis=0)
    else:
        per_frame_gam = np.zeros(T, dtype=np.float64)

    # Target noun trace
    if target_idx < len(word_maps) and word_maps[target_idx] is not None:
        target_per_frame_gam = _gam_for_map(word_maps[target_idx])
    else:
        target_per_frame_gam = per_frame_gam.copy()

    # Linear decay slope (fit over frame indices)
    xs = np.arange(T, dtype=np.float64)
    if T > 1:
        coeffs = np.polyfit(xs, per_frame_gam, 1)
        gam_decay = float(coeffs[0])   # negative → attention drifting away over time
    else:
        gam_decay = 0.0

    mean_gam = float(per_frame_gam.mean())
    lost_track_rate = float((per_frame_gam < lost_track_threshold).mean())

    return {
        "per_frame_gam": per_frame_gam,
        "target_per_frame_gam": target_per_frame_gam,
        "word_per_frame_gam": word_per_frame_gam,
        "mean_gam": mean_gam,
        "gam_decay": gam_decay,
        "lost_track_rate": lost_track_rate,
        "gt_masks_tam": gt_masks_tam,
    }


def pearson_coherence_in_gt_region(
    coherence_result: dict,
    gt_masks: List[np.ndarray],
    frame_size: Tuple[int, int],
    vision_shape: Tuple[int, int, int],
) -> dict:
    """Compute per-word Pearson temporal coherence restricted to the GT region.

    For each word, the attention map is masked to the GT object region (GT mask
    projected to TAM resolution) before computing pairwise Pearson correlation
    between consecutive frames.  Only pixels inside the GT region are used, so
    the score reflects how consistently the model attends *to the object itself*
    rather than the background.

    Parameters
    ----------
    coherence_result : output of ``pearson_temporal_coherence()``
        Must contain ``word_groups``, ``word_maps``, ``target_word_idx``.
    gt_masks : list of (H, W) binary uint8 arrays, one per frame.
    frame_size : (H, W) of the original video frames.
    vision_shape : (T, H_tam, W_tam) from the TAM runner output.

    Returns
    -------
    dict:
        "word_scores_in_gt"       : List[float|None] — per-word coherence inside GT
        "target_coherence_in_gt"  : float|None — coherence for the target noun
        "target_per_pair_in_gt"   : List[float] — per-pair values for target noun
        "gt_masks_tam"            : List[ndarray (H_tam, W_tam) bool]
    """
    word_groups = coherence_result.get("word_groups", [])
    word_maps = coherence_result.get("word_maps", [])
    target_word_idx = coherence_result.get("target_word_idx", 0)

    T, H_tam, W_tam = vision_shape
    H_frame, W_frame = frame_size

    # Project GT masks to TAM resolution
    gt_masks_tam: List[np.ndarray] = []
    for t in range(T):
        if t < len(gt_masks) and gt_masks[t] is not None and gt_masks[t].sum() > 0:
            resized = cv2.resize(
                gt_masks[t].astype(np.float32),
                (W_tam, H_tam),
                interpolation=cv2.INTER_NEAREST,
            )
            gt_masks_tam.append(resized > 0.5)
        else:
            # Fallback: no GT info → use the full map (same as unconstrained)
            gt_masks_tam.append(np.ones((H_tam, W_tam), dtype=bool))

    def _pearson_in_region(wm: np.ndarray) -> Tuple[List[float], Optional[float]]:
        """Compute per-pair Pearson for one word map restricted to GT region."""
        if wm is None or wm.shape[0] < 2:
            return [], None
        per_pair: List[float] = []
        for t in range(T - 1):
            # Union of consecutive frames' GT regions so we don't lose signal
            # at frame boundaries where the object moves slightly.
            region = gt_masks_tam[t] | gt_masks_tam[t + 1]  # (H_tam, W_tam) bool
            n_pix = int(region.sum())
            if n_pix < 4:
                per_pair.append(0.0)
                continue
            m1 = wm[t].astype(np.float64)[region]
            m2 = wm[t + 1].astype(np.float64)[region]
            m1c = m1 - m1.mean()
            m2c = m2 - m2.mean()
            n1 = np.linalg.norm(m1c)
            n2 = np.linalg.norm(m2c)
            if n1 < 1e-8 or n2 < 1e-8:
                per_pair.append(0.0)
            else:
                corr = float(np.dot(m1c, m2c) / (n1 * n2))
                per_pair.append((corr + 1.0) / 2.0)
        score = float(np.mean(per_pair)) if per_pair else None
        return per_pair, score

    # Compute for every word
    word_scores_in_gt: List[Optional[float]] = []
    for wm in word_maps:
        _, score = _pearson_in_region(wm)
        word_scores_in_gt.append(score)

    # Per-pair trace for the target noun specifically
    target_wm = (word_maps[target_word_idx]
                 if target_word_idx < len(word_maps) else None)
    target_per_pair, target_coherence_in_gt = _pearson_in_region(target_wm)

    return {
        "word_scores_in_gt": word_scores_in_gt,
        "target_coherence_in_gt": target_coherence_in_gt,
        "target_per_pair_in_gt": target_per_pair,
        "gt_masks_tam": gt_masks_tam,
    }


def pearson_temporal_coherence(tam_result: dict) -> dict:
    """Compute Pearson-based temporal coherence for the first object noun.

    Finds the first noun in the generated text, extracts its per-frame TAM map
    (T, H, W), and computes pairwise Pearson correlation between consecutive
    frames.

    Also groups all tokens into words and merges their maps — used for the
    summary visualizations in visualizer.py.

    Returns
    -------
    dict:
        "coherence_score"   : float in [0, 1]
        "target_word"       : str — noun used as the attention target
        "target_word_idx"   : int — index into word_groups (or 0)
        "per_pair_corr"     : List[float] — per-consecutive-pair correlations
        "word_groups"       : List[dict]  — {"word", "token_indices"}
        "word_maps"         : List[ndarray|(T,H,W) or None]  — merged per-word maps
    """
    T = tam_result["vision_shape"][0]
    gen_tokens = tam_result.get("gen_tokens", [])
    tam_maps = tam_result.get("tam_maps", [])

    word_groups = _group_tokens_into_words(gen_tokens)
    word_maps = _merge_word_maps(word_groups, tam_maps)

    # Find first noun
    noun_idx, noun_word = _find_first_noun_in_groups(word_groups)
    if noun_idx is None:
        noun_idx, noun_word = 0, word_groups[0]["word"] if word_groups else "(first)"

    target_map = word_maps[noun_idx] if noun_idx < len(word_maps) else None

    if target_map is None or target_map.shape[0] != T:
        return {
            "coherence_score": None,
            "target_word": noun_word,
            "target_word_idx": noun_idx,
            "per_pair_corr": [],
            "word_groups": word_groups,
            "word_maps": word_maps,
        }

    # Per-frame maps for target noun (normalised to [0,1])
    float_maps = [
        (target_map[t].astype(np.float64) / 255.0
         if target_map.dtype == np.uint8
         else target_map[t].astype(np.float64))
        for t in range(T)
    ]

    per_pair: List[float] = []
    for i in range(len(float_maps) - 1):
        m1 = float_maps[i].flatten()
        m2 = float_maps[i + 1].flatten()
        m1c = m1 - m1.mean()
        m2c = m2 - m2.mean()
        n1 = np.linalg.norm(m1c)
        n2 = np.linalg.norm(m2c)
        if n1 < 1e-8 or n2 < 1e-8:
            per_pair.append(0.0)
        else:
            corr = float(np.dot(m1c, m2c) / (n1 * n2))
            per_pair.append((corr + 1.0) / 2.0)

    coherence = float(np.mean(per_pair)) if per_pair else 1.0

    return {
        "coherence_score": coherence,
        "target_word": noun_word,
        "target_word_idx": noun_idx,
        "per_pair_corr": per_pair,
        "word_groups": word_groups,
        "word_maps": word_maps,
    }
