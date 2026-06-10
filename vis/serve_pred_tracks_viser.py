#!/usr/bin/env python3
"""Interactive Viser viewer for WorldTrack eval prediction artifacts.

Serves a `<video_name>_pred.npz` written by
`eval_track3d_in_worldtrack.py --save-predictions`: GT and predicted 3D tracks
with playback, per-track colors (GT lightened), motion trails, and metric-style
global scale alignment. No model, dataset, or GPU required.
"""

from __future__ import annotations

import argparse
import colorsys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve GT vs predicted 3D tracks from a _pred.npz artifact.")
    parser.add_argument("--pred-npz", required=True, help="Path to <video_name>_pred.npz from the eval script.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--max-tracks", type=int, default=0, help="Cap on rendered tracks (evenly subsampled). <=0 keeps all.")
    return parser.parse_args()


def _track_colors(num_tracks: int) -> np.ndarray:
    colors = np.zeros((max(num_tracks, 1), 3), dtype=np.uint8)
    for idx in range(num_tracks):
        rgb = colorsys.hsv_to_rgb((idx / max(num_tracks, 1)) * 0.92, 0.85, 1.0)
        colors[idx] = np.asarray([round(c * 255.0) for c in rgb], dtype=np.uint8)
    return colors


def _lighten_colors(colors: np.ndarray, amount: float) -> np.ndarray:
    out = colors.astype(np.float64)
    out = out + (255.0 - out) * float(np.clip(amount, 0.0, 1.0))
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def main() -> int:
    args = parse_args()
    try:
        import viser
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Missing dependency: `viser` is not installed in current env.") from exc

    pred_npz_path = Path(args.pred_npz).resolve()
    pack = np.load(pred_npz_path, allow_pickle=False)
    video_name = pred_npz_path.stem.removesuffix("_pred")

    gt_qt3 = np.asarray(pack["gt_tracks_xyz_world"], dtype=np.float32).transpose(1, 0, 2)
    pred_raw_qt3 = np.asarray(pack["pred_tracks_xyz_ref0"], dtype=np.float32).transpose(1, 0, 2)
    pred_vis_qt = np.asarray(pack["pred_visibility"], dtype=bool).T
    global_scale = float(pack["global_scale"])
    query_rgb = np.asarray(pack["query_rgb"], dtype=np.uint8) if "query_rgb" in pack.files else None
    num_tracks, num_frames = gt_qt3.shape[0], gt_qt3.shape[1]

    # Optional dense scene layer, stored frame-major [T, G, 3] / [T, G].
    dense_xyz_tg3 = np.asarray(pack["dense_tracks_xyz_ref0"], dtype=np.float32) if "dense_tracks_xyz_ref0" in pack.files else None
    dense_vis_tg = np.asarray(pack["dense_visibility"], dtype=bool) if "dense_visibility" in pack.files else None
    dense_rgb_g = np.asarray(pack["dense_rgb"], dtype=np.uint8) if "dense_rgb" in pack.files else None

    if int(args.max_tracks) > 0 and num_tracks > int(args.max_tracks):
        pick = np.linspace(0, num_tracks - 1, num=int(args.max_tracks), dtype=np.int64)
        gt_qt3, pred_raw_qt3, pred_vis_qt = gt_qt3[pick], pred_raw_qt3[pick], pred_vis_qt[pick]
        if query_rgb is not None:
            query_rgb = query_rgb[pick]
        num_tracks = int(args.max_tracks)

    dists = np.linalg.norm(pred_raw_qt3 * global_scale - gt_qt3, axis=-1)
    epe_global = float(np.mean(dists[np.isfinite(dists)]))

    track_id_colors = _track_colors(num_tracks)

    gt_flat = gt_qt3.reshape(-1, 3)
    gt_flat = gt_flat[np.isfinite(gt_flat).all(axis=1)]
    center = np.median(gt_flat, axis=0)
    radius = float(max(np.percentile(np.linalg.norm(gt_flat - center, axis=1), 95.0), 0.5))

    server = viser.ViserServer(host=args.host, port=int(args.port))
    server.scene.set_up_direction("-y")
    dense_count = int(dense_xyz_tg3.shape[1]) if dense_xyz_tg3 is not None else 0
    server.gui.add_markdown(
        f"**{video_name}** — {num_tracks} tracks × {num_frames} frames"
        + (f" + {dense_count} dense points" if dense_count else "")
        + f"  \nEPE(global) = {epe_global:.4f} m, global scale = {global_scale:.4f}"
    )

    with server.gui.add_folder("Timeline", expand_by_default=True):
        frame_slider = server.gui.add_slider("frame_idx", min=0, max=max(num_frames - 1, 0), step=1, initial_value=0)
        play_box = server.gui.add_checkbox("play", initial_value=True)
        loop_box = server.gui.add_checkbox("loop", initial_value=True)
        fps_slider = server.gui.add_slider("fps", min=1, max=30, step=1, initial_value=10)

    with server.gui.add_folder("Display", expand_by_default=True):
        show_gt = server.gui.add_checkbox("show_gt_tracks", initial_value=True)
        show_pred = server.gui.add_checkbox("show_pred_tracks", initial_value=True)
        color_mode = server.gui.add_dropdown(
            "color_mode",
            options=("rgb", "track_id") if query_rgb is not None else ("track_id",),
            initial_value="rgb" if query_rgb is not None else "track_id",
        )
        apply_scale = server.gui.add_checkbox("apply_global_scale", initial_value=True)
        hide_invisible = server.gui.add_checkbox("hide_pred_invisible", initial_value=False)
        history_slider = server.gui.add_slider("track_history (0=full)", min=0, max=num_frames, step=1, initial_value=12)
        head_size = server.gui.add_slider("point_size_scale", min=0.2, max=3.0, step=0.1, initial_value=1.0)
        line_width = server.gui.add_slider("line_width", min=1.0, max=8.0, step=0.5, initial_value=3.0)

    dense_controls: tuple[Any, ...] = ()
    if dense_xyz_tg3 is not None:
        with server.gui.add_folder("Dense scene", expand_by_default=True):
            show_dense = server.gui.add_checkbox("show_dense_points", initial_value=True)
            dense_hide_invisible = server.gui.add_checkbox("dense_hide_invisible", initial_value=True)
            dense_size = server.gui.add_slider("dense_point_size_scale", min=0.2, max=3.0, step=0.1, initial_value=1.0)
        dense_controls = (show_dense, dense_hide_invisible, dense_size)

    render_handles: list[Any] = []
    render_lock = threading.Lock()

    def _render_track_layer(
        *,
        name_prefix: str,
        xyz_qt3: np.ndarray,
        keep_qt: np.ndarray,
        colors: np.ndarray,
        frame_idx: int,
    ) -> None:
        hist = int(history_slider.value)
        t0 = 0 if hist <= 0 else max(0, frame_idx - hist + 1)
        segs, seg_cols, head_pts, head_cols = [], [], [], []
        for qi in range(xyz_qt3.shape[0]):
            pts = [xyz_qt3[qi, ti] for ti in range(t0, frame_idx + 1) if keep_qt[qi, ti]]
            if len(pts) >= 2:
                pts_arr = np.asarray(pts, dtype=np.float32)
                seg = np.stack([pts_arr[:-1], pts_arr[1:]], axis=1)
                segs.append(seg)
                col = np.repeat(colors[qi][None, None, :], repeats=seg.shape[0], axis=0)
                seg_cols.append(np.repeat(col, repeats=2, axis=1))
            if pts:
                head_pts.append(pts[-1])
                head_cols.append(colors[qi])
        if segs:
            render_handles.append(
                server.scene.add_line_segments(
                    f"/tracks/{name_prefix}/lines",
                    points=np.concatenate(segs, axis=0).astype(np.float32),
                    colors=np.concatenate(seg_cols, axis=0).astype(np.uint8),
                    line_width=float(line_width.value),
                )
            )
        if head_pts:
            render_handles.append(
                server.scene.add_point_cloud(
                    f"/tracks/{name_prefix}/heads",
                    points=np.asarray(head_pts, dtype=np.float32),
                    colors=np.asarray(head_cols, dtype=np.uint8),
                    point_size=0.012 * radius * float(head_size.value),
                    point_shape="circle",
                )
            )

    def render() -> None:
        with render_lock:
            for handle in render_handles:
                try:
                    handle.remove()
                except Exception:
                    pass
            render_handles.clear()
            frame_idx = int(frame_slider.value)
            pred_qt3 = pred_raw_qt3 * global_scale if bool(apply_scale.value) else pred_raw_qt3
            pred_colors = query_rgb if (str(color_mode.value) == "rgb" and query_rgb is not None) else track_id_colors
            gt_colors = _lighten_colors(pred_colors, amount=0.55)
            if bool(show_gt.value):
                keep = np.isfinite(gt_qt3).all(axis=-1)
                _render_track_layer(name_prefix="gt", xyz_qt3=gt_qt3, keep_qt=keep, colors=gt_colors, frame_idx=frame_idx)
            if bool(show_pred.value):
                keep = np.isfinite(pred_qt3).all(axis=-1)
                if bool(hide_invisible.value):
                    keep &= pred_vis_qt
                _render_track_layer(name_prefix="pred", xyz_qt3=pred_qt3, keep_qt=keep, colors=pred_colors, frame_idx=frame_idx)
            if dense_xyz_tg3 is not None and bool(show_dense.value):
                points = dense_xyz_tg3[frame_idx] * (global_scale if bool(apply_scale.value) else 1.0)
                keep = np.isfinite(points).all(axis=-1)
                if dense_vis_tg is not None and bool(dense_hide_invisible.value):
                    keep &= dense_vis_tg[frame_idx]
                if np.any(keep):
                    render_handles.append(
                        server.scene.add_point_cloud(
                            "/dense/points",
                            points=points[keep].astype(np.float32),
                            colors=(dense_rgb_g[keep] if dense_rgb_g is not None else np.full((int(keep.sum()), 3), 180, dtype=np.uint8)),
                            point_size=0.006 * radius * float(dense_size.value),
                            point_shape="circle",
                        )
                    )

    for control in (frame_slider, show_gt, show_pred, color_mode, apply_scale, hide_invisible, history_slider, head_size, line_width, *dense_controls):
        control.on_update(lambda _event: render())

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.up_direction = (0.0, -1.0, 0.0)
        client.camera.position = tuple(float(v) for v in (center + np.asarray([0.0, -0.6, -2.8]) * radius).tolist())
        client.camera.look_at = tuple(float(v) for v in center.tolist())

    def _playback_loop() -> None:
        while True:
            if bool(play_box.value) and num_frames > 1:
                nxt = int(frame_slider.value) + 1
                if nxt >= num_frames:
                    if bool(loop_box.value):
                        frame_slider.value = 0
                    else:
                        play_box.value = False
                else:
                    frame_slider.value = nxt
            time.sleep(1.0 / max(float(fps_slider.value), 1.0))

    render()
    threading.Thread(target=_playback_loop, daemon=True).start()
    print(f"Serving {video_name} on http://{args.host}:{int(args.port)} — Ctrl+C to stop.")
    while True:
        time.sleep(3600.0)


if __name__ == "__main__":
    raise SystemExit(main())
