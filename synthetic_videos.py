"""
Generate synthetic test videos for evaluating video object tracking in MLLMs.

Each video type is saved as a sequence of JPEG frames in a subdirectory under
test_videos/, along with a trajectory.json file containing ground-truth
positions and visibility for the primary tracked object.

Compatible with TAM's video input format (list of image paths).
"""

import json
import os
from PIL import Image, ImageDraw


BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_videos")
FRAME_SIZE = (448, 448)
BG_COLOR = (180, 180, 180)  # neutral gray
RADIUS = 30


def _save_video(name, frames, trajectory):
    """Save a list of PIL images and trajectory dict to disk."""
    out_dir = os.path.join(BASE_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    for idx, img in enumerate(frames):
        img.save(os.path.join(out_dir, f"{idx:04d}.jpg"), quality=95)

    with open(os.path.join(out_dir, "trajectory.json"), "w") as f:
        json.dump(trajectory, f, indent=2)

    print(f"  Saved {len(frames)} frames to {out_dir}/")


# ── 1. Simple translation ────────────────────────────────────────────────────

def generate_simple_translation():
    """Red circle moving linearly left-to-right across a gray background."""
    n_frames = 10
    frames = []
    trajectory = []

    y = FRAME_SIZE[1] // 2
    x_start = RADIUS + 20
    x_end = FRAME_SIZE[0] - RADIUS - 20

    for i in range(n_frames):
        t = i / (n_frames - 1)
        x = int(x_start + t * (x_end - x_start))

        img = Image.new("RGB", FRAME_SIZE, BG_COLOR)
        draw = ImageDraw.Draw(img)
        draw.ellipse(
            [x - RADIUS, y - RADIUS, x + RADIUS, y + RADIUS],
            fill=(220, 40, 40),
        )
        frames.append(img)
        trajectory.append({"frame_idx": i, "x": x, "y": y, "visible": True})

    _save_video("simple_translation", frames, trajectory)


# ── 2. Two objects crossing ───────────────────────────────────────────────────

def generate_two_objects_crossing():
    """Red and blue circles start on opposite sides, cross, and continue."""
    n_frames = 10
    frames = []
    trajectory = []  # track the RED circle

    y_red = FRAME_SIZE[1] // 2 - 10
    y_blue = FRAME_SIZE[1] // 2 + 10
    x_start = RADIUS + 20
    x_end = FRAME_SIZE[0] - RADIUS - 20

    for i in range(n_frames):
        t = i / (n_frames - 1)
        x_red = int(x_start + t * (x_end - x_start))
        x_blue = int(x_end - t * (x_end - x_start))

        img = Image.new("RGB", FRAME_SIZE, BG_COLOR)
        draw = ImageDraw.Draw(img)

        # Draw blue first so red is on top when they overlap
        draw.ellipse(
            [x_blue - RADIUS, y_blue - RADIUS, x_blue + RADIUS, y_blue + RADIUS],
            fill=(40, 40, 220),
        )
        draw.ellipse(
            [x_red - RADIUS, y_red - RADIUS, x_red + RADIUS, y_red + RADIUS],
            fill=(220, 40, 40),
        )

        frames.append(img)
        trajectory.append({"frame_idx": i, "x": x_red, "y": y_red, "visible": True})

    _save_video("two_objects_crossing", frames, trajectory)


# ── 3. Appearance change ─────────────────────────────────────────────────────

def generate_appearance_change():
    """Circle gradually transitions from red to blue while moving left-to-right."""
    n_frames = 10
    frames = []
    trajectory = []

    y = FRAME_SIZE[1] // 2
    x_start = RADIUS + 20
    x_end = FRAME_SIZE[0] - RADIUS - 20

    for i in range(n_frames):
        t = i / (n_frames - 1)
        x = int(x_start + t * (x_end - x_start))

        # Interpolate colour from red (220,40,40) to blue (40,40,220)
        r = int(220 + t * (40 - 220))
        g = 40
        b = int(40 + t * (220 - 40))

        img = Image.new("RGB", FRAME_SIZE, BG_COLOR)
        draw = ImageDraw.Draw(img)
        draw.ellipse(
            [x - RADIUS, y - RADIUS, x + RADIUS, y + RADIUS],
            fill=(r, g, b),
        )
        frames.append(img)
        trajectory.append({"frame_idx": i, "x": x, "y": y, "visible": True})

    _save_video("appearance_change", frames, trajectory)


# ── 4. Occlusion ──────────────────────────────────────────────────────────────

def generate_occlusion():
    """Red circle passes behind a green barrier in the centre of the frame."""
    n_frames = 10
    frames = []
    trajectory = []

    y = FRAME_SIZE[1] // 2
    x_start = RADIUS + 20
    x_end = FRAME_SIZE[0] - RADIUS - 20

    # Green barrier: wide enough to fully hide the circle for ~2-3 frames
    # With 10 frames spanning ~400px, each step is ~44px.
    # Barrier width = 3 steps worth + 2*RADIUS = ~130 + 60 = ~190px centred.
    barrier_half_w = 95
    barrier_cx = FRAME_SIZE[0] // 2
    barrier_left = barrier_cx - barrier_half_w
    barrier_right = barrier_cx + barrier_half_w

    for i in range(n_frames):
        t = i / (n_frames - 1)
        x = int(x_start + t * (x_end - x_start))

        img = Image.new("RGB", FRAME_SIZE, BG_COLOR)
        draw = ImageDraw.Draw(img)

        # Determine visibility: circle fully behind barrier?
        circle_left = x - RADIUS
        circle_right = x + RADIUS
        occluded = circle_left >= barrier_left and circle_right <= barrier_right

        # Draw circle first (behind barrier)
        draw.ellipse(
            [x - RADIUS, y - RADIUS, x + RADIUS, y + RADIUS],
            fill=(220, 40, 40),
        )

        # Draw barrier on top
        draw.rectangle(
            [barrier_left, 0, barrier_right, FRAME_SIZE[1]],
            fill=(40, 180, 40),
        )

        frames.append(img)
        trajectory.append({
            "frame_idx": i,
            "x": x,
            "y": y,
            "visible": not occluded,
        })

    _save_video("occlusion", frames, trajectory)


# ── 5. Re-entry ───────────────────────────────────────────────────────────────

def generate_reentry():
    """Red circle exits the right edge and re-enters from the left."""
    n_frames = 14
    frames = []
    trajectory = []

    y = FRAME_SIZE[1] // 2

    # Phase 1 (frames 0-5): move from left to beyond right edge
    # Phase 2 (frames 6-8): off-screen
    # Phase 3 (frames 9-13): re-enter from left and move right

    for i in range(n_frames):
        img = Image.new("RGB", FRAME_SIZE, BG_COLOR)
        draw = ImageDraw.Draw(img)

        if i <= 5:
            # Moving left to right, exiting at frame 5
            t = i / 5
            x = int(50 + t * (FRAME_SIZE[0] + RADIUS - 50))
        elif i <= 8:
            # Off-screen to the right
            x = FRAME_SIZE[0] + RADIUS + 50
        else:
            # Re-enter from left (frames 9-13)
            t = (i - 9) / 4
            x = int(-RADIUS + t * (FRAME_SIZE[0] // 2 + RADIUS))

        # Determine if visible (circle at least partially on-screen)
        on_screen = (-RADIUS < x < FRAME_SIZE[0] + RADIUS)

        if on_screen:
            draw.ellipse(
                [x - RADIUS, y - RADIUS, x + RADIUS, y + RADIUS],
                fill=(220, 40, 40),
            )

        frames.append(img)
        trajectory.append({
            "frame_idx": i,
            "x": x,
            "y": y,
            "visible": on_screen,
        })

    _save_video("reentry", frames, trajectory)


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_all():
    """Generate all synthetic test videos."""
    os.makedirs(BASE_DIR, exist_ok=True)
    print(f"Generating synthetic test videos in {BASE_DIR}/\n")

    print("1/5  simple_translation")
    generate_simple_translation()

    print("2/5  two_objects_crossing")
    generate_two_objects_crossing()

    print("3/5  appearance_change")
    generate_appearance_change()

    print("4/5  occlusion")
    generate_occlusion()

    print("5/5  reentry")
    generate_reentry()

    print("\nDone. All test videos generated.")


if __name__ == "__main__":
    generate_all()
