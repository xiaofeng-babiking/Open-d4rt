#!/usr/bin/env python3
"""Render GT vs predicted 3D tracks from a WorldTrack eval prediction artifact.

Consumes a `<video_name>_pred.npz` written by
`eval_track3d_in_worldtrack.py --save-predictions` and produces:
- `<video_name>_tracks_3d.png`: static overlay of full GT and aligned predicted trajectories.
- `<video_name>_tracks_3d.gif`: animated side-by-side GT | prediction view.

Predictions are aligned to GT with the stored `global_scale`, matching the
APD/EPE(global) metric protocol.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize GT vs predicted 3D tracks from a _pred.npz artifact.")
    parser.add_argument("--pred-npz", required=True, help="Path to <video_name>_pred.npz from the eval script.")
    parser.add_argument("--output-dir", default="", help="Output directory. Defaults to tmp/track_vis/<video_name>.")
    parser.add_argument("--max-tracks", type=int, default=64, help="Cap on rendered tracks (evenly subsampled).")
    parser.add_argument(
        "--color-mode",
        choices=("auto", "rgb", "track"),
        default="auto",
        help="Track colors: frame-0 query pixel RGB (semantics) or per-track rainbow. auto prefers rgb when stored.",
    )
    parser.add_argument("--fps", type=int, default=10, help="GIF frame rate.")
    parser.add_argument("--trail-frames", type=int, default=12, help="Trailing line length in frames for the GIF.")
    parser.add_argument("--elev", type=float, default=18.0, help="Camera elevation in degrees.")
    parser.add_argument("--rotate-deg", type=float, default=50.0, help="Total azimuth sweep across the GIF.")
    return parser.parse_args()


def _to_plot_frame(xyz_tq3: np.ndarray) -> np.ndarray:
    """Remap camera coords (x right, y down, z forward) to plot coords (x, depth, up)."""
    out = np.empty_like(xyz_tq3)
    out[..., 0] = xyz_tq3[..., 0]
    out[..., 1] = xyz_tq3[..., 2]
    out[..., 2] = -xyz_tq3[..., 1]
    return out


def _axis_bounds(points: np.ndarray) -> list[tuple[float, float]]:
    flat = points.reshape(-1, 3)
    flat = flat[np.isfinite(flat).all(axis=1)]
    bounds = []
    for axis in range(3):
        lo, hi = np.percentile(flat[:, axis], [1.0, 99.0])
        pad = 0.05 * max(hi - lo, 1e-6)
        bounds.append((float(lo - pad), float(hi + pad)))
    return bounds


def _setup_axes(ax: plt.Axes, title: str, bounds: list[tuple[float, float]], elev: float, azim: float) -> None:
    ax.set_title(title)
    ax.set_xlim(*bounds[0])
    ax.set_ylim(*bounds[1])
    ax.set_zlim(*bounds[2])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_zlabel("-y [m]")
    ax.view_init(elev=elev, azim=azim)


def main() -> int:
    args = parse_args()
    pred_npz_path = Path(args.pred_npz)
    pack = np.load(pred_npz_path, allow_pickle=False)
    video_name = pred_npz_path.stem.removesuffix("_pred")
    output_dir = Path(args.output_dir) if args.output_dir else Path("tmp/track_vis") / video_name
    output_dir.mkdir(parents=True, exist_ok=True)

    gt = np.asarray(pack["gt_tracks_xyz_world"], dtype=np.float64)
    pred = np.asarray(pack["pred_tracks_xyz_ref0"], dtype=np.float64) * float(pack["global_scale"])
    num_frames, num_tracks = gt.shape[0], gt.shape[1]

    dists = np.linalg.norm(pred - gt, axis=-1)
    epe = float(np.mean(dists[np.isfinite(dists)]))

    query_rgb = np.asarray(pack["query_rgb"], dtype=np.uint8) if "query_rgb" in pack.files else None
    if num_tracks > int(args.max_tracks):
        pick = np.linspace(0, num_tracks - 1, num=int(args.max_tracks), dtype=np.int64)
        gt, pred = gt[:, pick], pred[:, pick]
        if query_rgb is not None:
            query_rgb = query_rgb[pick]
        num_tracks = int(args.max_tracks)

    gt_plot = _to_plot_frame(gt)
    pred_plot = _to_plot_frame(pred)
    bounds = _axis_bounds(np.concatenate([gt_plot, pred_plot], axis=1))
    color_mode = args.color_mode
    if color_mode == "auto":
        color_mode = "rgb" if query_rgb is not None else "track"
    if color_mode == "rgb":
        if query_rgb is None:
            raise SystemExit(f"{pred_npz_path} has no query_rgb key; regenerate it or use --color-mode track.")
        colors = np.concatenate([query_rgb.astype(np.float64) / 255.0, np.ones((num_tracks, 1))], axis=1)
    else:
        colors = cm.hsv(np.linspace(0.0, 0.92, num_tracks))

    # Static overlay: full GT trajectories in gray, aligned predictions colored.
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    for q in range(num_tracks):
        ax.plot(*gt_plot[:, q].T, color="0.55", lw=0.8, alpha=0.7)
        ax.plot(*pred_plot[:, q].T, color=colors[q], lw=1.0, alpha=0.9)
    _setup_axes(ax, f"{video_name}: GT (gray) vs pred (color), EPE(global)={epe:.4f} m", bounds, args.elev, -60.0)
    png_path = output_dir / f"{video_name}_tracks_3d.png"
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Animated side-by-side view with trailing lines.
    frames = []
    trail = max(1, int(args.trail_frames))
    for t in range(num_frames):
        azim = -60.0 + float(args.rotate_deg) * t / max(num_frames - 1, 1)
        fig = plt.figure(figsize=(12, 6))
        for col, (label, tracks) in enumerate((("GT", gt_plot), (f"OpenD4RT pred (EPE={epe:.3f} m)", pred_plot))):
            ax = fig.add_subplot(1, 2, col + 1, projection="3d")
            start = max(0, t - trail)
            for q in range(num_tracks):
                ax.plot(*tracks[start : t + 1, q].T, color=colors[q], lw=1.2, alpha=0.8)
            ax.scatter(*tracks[t].T, c=colors, s=14, depthshade=False)
            _setup_axes(ax, f"{label} — frame {t + 1}/{num_frames}", bounds, args.elev, azim)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(frame)
        plt.close(fig)

    gif_path = output_dir / f"{video_name}_tracks_3d.gif"
    imageio.mimsave(gif_path, frames, fps=int(args.fps), loop=0)
    print(f"Saved {png_path}")
    print(f"Saved {gif_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
