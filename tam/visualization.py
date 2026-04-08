"""Visualization utilities for TAM: heatmap generation, text rendering, composition.

Supports two text rendering backends:
1. LaTeX (xelatex) — high quality, requires texlive-xetex
2. Matplotlib/PIL fallback — works everywhere
"""

import os
import cv2
import numpy as np
from pathlib import Path



# ---------- LaTeX-based text visualization ----------

def generate_latex(words, relevances, cmap="bwr", font=r'{18pt}{21pt}'):
    """Generate LaTeX source for color-coded token visualization.

    Relevance encoding:
        >= 0: earlier context tokens (jet colormap)
        -1: current explained token (black bg, white text)
        -2: future tokens (gray)
        -3: newline + "Candidates:" label
        -4: custom string on new line
    """
    latex_code = r'''
    \documentclass[arwidth=200mm]{standalone}
    \renewcommand{\normalsize}{\fontsize''' + font + r'''\selectfont}
    \usepackage[dvipsnames]{xcolor}

    \begin{document}
    \fbox{
    \parbox{\textwidth}{
    \setlength\fboxsep{0pt}
    '''

    for i in range(len(words)):
        word = words[i]
        relevance = relevances[i]

        if relevance >= 0:
            jet_colormap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
            b, g, r = jet_colormap[int(relevances[i] * 255)][0].tolist()
            if word[:2] == '$ ' and word[-1] == '$':
                latex_code += f' \\textbf{{\\textcolor[RGB]{{{r},{g},{b}}}{{\\strut {word}}}}}, '
            elif word.startswith('▁') or word.startswith('Ġ') or word.startswith(' '):
                word = word.replace('▁', ' ').replace('Ġ', ' ')
                latex_code += f' \\textbf{{\\textcolor[RGB]{{{r},{g},{b}}}{{\\strut {word}}}}}'
            else:
                latex_code += f'\\textbf{{\\textcolor[RGB]{{{r},{g},{b}}}{{\\strut {word}}}}}'

        elif relevance == -1:
            if word.startswith('▁') or word.startswith('Ġ') or word.startswith(' '):
                word = word.replace('▁', ' ').replace('Ġ', ' ')
                latex_code += f' \\textbf{{\\colorbox[RGB]{{{0},{0},{0}}}{{\\textcolor[RGB]{{{255},{255},{255}}}{{\\strut {word}}}}}}}'
            else:
                latex_code += f'\\textbf{{\\colorbox[RGB]{{{0},{0},{0}}}{{\\textcolor[RGB]{{{255},{255},{255}}}{{\\strut {word}}}}}}}'

        elif relevance == -2:
            b, g, r = 200, 200, 200
            if word.startswith('▁') or word.startswith('Ġ') or word.startswith(' '):
                word = word.replace('▁', ' ').replace('Ġ', ' ')
                latex_code += f' \\textbf{{\\textcolor[RGB]{{{r},{g},{b}}}{{\\strut {word}}}}}'
            else:
                latex_code += f'\\textbf{{\\textcolor[RGB]{{{r},{g},{b}}}{{\\strut {word}}}}}'

        elif relevance == -3:
            latex_code += '\\\\$Candidates:$'

        elif relevance == -4:
            latex_code += '\\\\' + word

    latex_code += r'}}\end{document}'
    return latex_code


def compile_latex_to_img(latex_code, path='word_colors.pdf', dpi=500):
    """Compile LaTeX to image via xelatex + pymupdf. Returns BGR numpy array or None."""
    import subprocess

    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)

    with open(path.with_suffix(".tex"), 'w') as f:
        f.write(latex_code)

    try:
        subprocess.run(
            ['xelatex', '--output-directory', str(path.parent), str(path.with_suffix(".tex"))],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60,
        )
    except Exception:
        return None

    try:
        import fitz
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        page = fitz.open(str(path.with_suffix(".pdf"))).load_page(0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        png_data = pix.tobytes("png")
        img_array = np.frombuffer(png_data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_ANYCOLOR)[:, :, :3]
    except Exception:
        img = None

    # Cleanup auxiliary files
    for suffix in ['.aux', '.log', '.tex', '.pdf']:
        try:
            os.remove(path.with_suffix(suffix))
        except OSError:
            pass

    return img


# ---------- Matplotlib fallback text visualization ----------

def _render_text_matplotlib(words, relevances, candidates, candi_scores, vis_token_idx, width=500):
    """Pure-Python fallback for text visualization using matplotlib."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib import cm
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(width / 100, 2), dpi=100)
    ax.axis('off')

    x, y = 0.01, 0.85
    max_x = 0.99
    line_height = 0.18

    for i in range(len(words)):
        word = words[i]
        rel = relevances[i] if i < len(relevances) else -2

        if rel >= 0:
            color = cm.jet(rel)
        elif rel == -1:
            color = 'white'
        elif rel == -2:
            color = 'lightgray'
        else:
            continue

        display_word = word.replace('▁', ' ').replace('Ġ', ' ')
        bbox = dict(facecolor='black', alpha=0.9) if rel == -1 else None

        text_obj = ax.text(x, y, display_word, fontsize=7, color=color,
                          fontweight='bold', transform=ax.transAxes, bbox=bbox)

        # Estimate text width
        x += len(display_word) * 0.012 + 0.005
        if x > max_x:
            x = 0.01
            y -= line_height

    # Add candidates
    if candidates:
        y -= line_height
        ax.text(0.01, y, "Candidates: ", fontsize=6, color='gray',
                fontweight='bold', transform=ax.transAxes)
        x = 0.15
        for j, cand in enumerate(candidates):
            score = candi_scores[j].item() if hasattr(candi_scores[j], 'item') else float(candi_scores[j])
            color = cm.jet(score)
            ax.text(x, y, f"{cand} ({score:.2f})", fontsize=6, color=color,
                    fontweight='bold', transform=ax.transAxes)
            x += 0.2

    fig.tight_layout(pad=0)
    fig.canvas.draw()

    # Convert to numpy
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    plt.close(fig)

    return img


def vis_text(words, relevances, candidates, candi_scores, vis_token_idx,
             path='heatmap.jpg', font=r'{18pt}{21pt}'):
    """Render token text visualization. Tries LaTeX first, falls back to matplotlib."""
    # Build score arrays
    add_scores = [-2] * (len(words[:-1]) - len(relevances))
    all_scores = relevances.tolist() + add_scores + [-3] + candi_scores.cpu().float().tolist()
    all_scores[vis_token_idx] = -1

    all_words = words[:-1] + [''] + ['$ ' + _ + '$' for _ in candidates]

    # Escape LaTeX special characters
    all_words_escaped = [
        _.replace('\\', '\\backslash').replace('\n', '\\newline')
        .replace('_', '\\_').replace('^', '\\^')
        .replace('&', '\\&').replace('%', '\\%').replace('Ċ', '\\newline')
        for _ in all_words
    ]

    # Try LaTeX rendering
    try:
        latex_code = generate_latex(all_words_escaped, all_scores, cmap='bwr', font=font)
        img = compile_latex_to_img(latex_code, path=path)
        if isinstance(img, np.ndarray):
            return img
    except Exception:
        pass

    # Fallback to matplotlib
    return _render_text_matplotlib(all_words, all_scores, candidates, candi_scores, vis_token_idx)


# ---------- Multimodal composition ----------

def multimodal_process(raw_img, vision_shape, img_scores, txt_scores, txts,
                       candidates, candi_scores, vis_token_idx, img_save_fn,
                       eval_only=False, vis_width=-1):
    """Process and compose multimodal activation maps.

    Normalizes image and text scores together, applies Rank Gaussian Filter,
    generates heatmaps, and composes final visualization.

    Args:
        raw_img: Raw image (numpy BGR) or list of images/frames.
        vision_shape: (h, w), list of (h, w), or (b, h, w) for video.
        img_scores: Activation scores for image tokens.
        txt_scores: Activation scores for text tokens.
        txts: Token text list for visualization.
        candidates: Top-k candidate token strings.
        candi_scores: Candidate scores tensor.
        vis_token_idx: Index of the explained token.
        img_save_fn: Save path (empty string for eval-only).
        eval_only: Skip visualization if True.
        vis_width: Target width for resizing (-1 for no resize).

    Returns:
        (out_img, img_map) tuple. out_img is None if eval_only.
    """
    from .tam_core import rank_gaussian_filter

    # Normalize multimodal tokens
    txt_scores = txt_scores[:-1]  # ignore self score
    all_scores = np.concatenate([img_scores, txt_scores], 0)
    score_range = all_scores.max() - all_scores.min()
    if score_range > 0:
        all_scores = (all_scores - all_scores.min()) / score_range
    img_scores = all_scores[:len(img_scores)]
    txt_scores = all_scores[len(img_scores):]

    eval_only = True if img_save_fn == "" else False

    # --- Multiple images ---
    if isinstance(vision_shape[0], tuple):
        resized_img, img_map = [], []
        start_idx = 0
        for n in range(len(vision_shape)):
            t_h, t_w = vision_shape[n]
            h, w, c = raw_img[n].shape
            if vis_width > 0:
                h = int(vis_width)
                w = int(float(w) / h * vis_width)

            end_idx = start_idx + int(t_h * t_w)
            img_map_ = rank_gaussian_filter(img_scores[start_idx:end_idx].reshape(t_h, t_w), 3)
            start_idx = end_idx
            img_map_ = (img_map_ * 255).astype('uint8')

            if not eval_only:
                img_map_ = cv2.applyColorMap(img_map_, cv2.COLORMAP_JET)
                img_map_ = cv2.resize(img_map_, (w, h))
                if vis_width > 0:
                    resized_img.append(cv2.resize(raw_img[n], (w, h)))
            img_map.append(img_map_)

        if eval_only:
            return None, img_map

        out_img = [(img_map[i] * 0.5 + resized_img[i] * 0.5).astype(np.uint8) for i in range(len(vision_shape))]
        out_img = np.concatenate(out_img, 1)

        try:
            txt_map = vis_text(txts, txt_scores, candidates, candi_scores, vis_token_idx,
                              path=img_save_fn, font=r'{5pt}{6pt}')
        except Exception:
            return out_img, img_map

        if not isinstance(txt_map, np.ndarray):
            return out_img, img_map

        txt_map = cv2.resize(txt_map, (out_img.shape[1],
                             int(float(txt_map.shape[0]) / float(txt_map.shape[1]) * out_img.shape[1])))
        out_img = np.concatenate([out_img, txt_map], 0)
        return out_img, img_map

    # --- Single image ---
    elif len(vision_shape) == 2:
        t_h, t_w = vision_shape
        h, w, c = raw_img.shape
        if vis_width > 0:
            h = int(float(h) / w * vis_width)
            w = int(vis_width)

        img_scores = rank_gaussian_filter(img_scores.reshape(t_h, t_w), 3)
        img_scores = (img_scores * 255).astype('uint8')

        if eval_only:
            return None, img_scores

        img_map = cv2.applyColorMap(img_scores, cv2.COLORMAP_JET)
        img_map = cv2.resize(img_map, (w, h))
        if vis_width > 0:
            raw_img = cv2.resize(raw_img, (w, h))
        out_img = (img_map * 0.5 + raw_img * 0.5).astype(np.uint8)

        try:
            txt_map = vis_text(txts, txt_scores, candidates, candi_scores, vis_token_idx, path=img_save_fn)
        except Exception:
            return out_img, img_scores

        if not isinstance(txt_map, np.ndarray):
            return out_img, img_scores

        txt_map = cv2.resize(txt_map, (w, int(float(txt_map.shape[0]) / float(txt_map.shape[1]) * w)))
        out_img = np.concatenate([out_img, txt_map], 0)
        return out_img, img_scores

    # --- Video ---
    else:
        b, t_h, t_w = vision_shape
        h, w, c = raw_img[0].shape
        if vis_width > 0:
            h = int(float(h) / w * vis_width)
            w = int(vis_width)

        img_scores = np.array([
            rank_gaussian_filter(_.reshape(t_h, t_w), 3)
            for _ in np.array_split(img_scores, b)
        ])
        img_scores = (img_scores * 255).astype('uint8')

        if eval_only:
            return None, img_scores

        img_map = [cv2.resize(cv2.applyColorMap(_, cv2.COLORMAP_JET), (w, h)) for _ in img_scores]
        if vis_width > 0:
            raw_img = [cv2.resize(_, (w, h)) for _ in raw_img]
        out_img = [(img_map[i] * 0.5 + raw_img[i] * 0.5).astype(np.uint8) for i in range(b)]
        out_img = np.concatenate(out_img, 1)

        try:
            txt_map = vis_text(txts, txt_scores, candidates, candi_scores, vis_token_idx,
                              path=img_save_fn, font=r'{5pt}{6pt}')
        except Exception:
            return out_img, img_scores

        if not isinstance(txt_map, np.ndarray):
            return out_img, img_scores

        txt_map = cv2.resize(txt_map, (int(w * b),
                             int(float(txt_map.shape[0]) / float(txt_map.shape[1]) * w * b)))
        out_img = np.concatenate([out_img, txt_map], 0)
        return out_img, img_scores
