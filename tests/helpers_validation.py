"""Shared assertions for VGGT-processed dataset adapter samples."""

from __future__ import annotations

import cv2
import numpy as np
import torch

from helpers_synthetic import SyntheticFrame, frame_signature_red

SAMPLE_KEYS = {
    "video",
    "aspect_ratio",
    "depth_m",
    "depth_valid",
    "query",
    "query_stats",
    "target",
    "mask",
    "camera",
    "augment_info",
    "meta",
}
TARGET_KEYS = {"xyz_3d", "uv_2d", "visibility", "displacement", "normal"}


def assert_sample_schema(sample: dict, clip_frames: int, out_hw: tuple[int, int]) -> None:
    h, w = out_hw
    assert SAMPLE_KEYS.issubset(sample.keys()), f"missing keys: {SAMPLE_KEYS - set(sample.keys())}"

    video = sample["video"]
    assert isinstance(video, torch.Tensor) and video.dtype == torch.float32
    assert tuple(video.shape) == (clip_frames, 3, h, w)
    assert torch.isfinite(video).all()
    assert float(video.min()) >= 0.0 and float(video.max()) <= 1.0

    assert tuple(sample["depth_m"].shape) == (clip_frames, h, w)
    assert sample["depth_m"].dtype == torch.float32
    assert tuple(sample["depth_valid"].shape) == (clip_frames, h, w)
    assert sample["depth_valid"].dtype == torch.bool

    assert tuple(sample["aspect_ratio"].shape) == (1,)

    query = sample["query"]
    m = query["u"].numel()
    assert m > 0
    for key in ("u", "v"):
        assert query[key].dtype == torch.float32
        assert float(query[key].min()) >= 0.0 and float(query[key].max()) <= 1.0
    for key in ("t_src", "t_tgt", "t_cam"):
        assert query[key].dtype == torch.long
        assert int(query[key].min()) >= 0 and int(query[key].max()) < clip_frames
        assert query[key].numel() == m

    target = sample["target"]
    mask = sample["mask"]
    assert TARGET_KEYS.issubset(target.keys())
    assert TARGET_KEYS.issubset(mask.keys())
    assert tuple(target["xyz_3d"].shape) == (m, 3)
    assert tuple(target["uv_2d"].shape) == (m, 2)
    for key in TARGET_KEYS:
        assert mask[key].dtype == torch.bool
        assert torch.isfinite(target[key][mask[key]]).all()

    cam = sample["camera"]
    assert tuple(cam["K"].shape) == (clip_frames, 3, 3)
    assert tuple(cam["T_wc"].shape) == (clip_frames, 4, 4)
    assert tuple(cam["camera_valid"].shape) == (clip_frames,)


def expected_resized_k(k: np.ndarray, src_hw: tuple[int, int], out_hw: tuple[int, int]) -> np.ndarray:
    sx = out_hw[1] / float(src_hw[1])
    sy = out_hw[0] / float(src_hw[0])
    out = np.asarray(k, dtype=np.float64).copy()
    out[0, 0] *= sx
    out[0, 2] *= sx
    out[1, 1] *= sy
    out[1, 2] *= sy
    return out


def reprojection_median_rel_error(
    depth: np.ndarray,
    depth_valid: np.ndarray,
    k_seq: np.ndarray,
    t_wc_seq: np.ndarray,
    i: int = 0,
    j: int = -1,
) -> float:
    """Static-scene cross-frame consistency of (depth, K, T_wc) inside a sample."""
    j = j % depth.shape[0]
    h, w = depth.shape[1:]
    ys, xs = np.nonzero(depth_valid[i])
    z = depth[i, ys, xs].astype(np.float64)
    k_i, k_j = k_seq[i].astype(np.float64), k_seq[j].astype(np.float64)
    pts = np.stack(
        [
            (xs + 0.5 - k_i[0, 2]) / k_i[0, 0] * z,
            (ys + 0.5 - k_i[1, 2]) / k_i[1, 1] * z,
            z,
            np.ones_like(z),
        ],
        axis=0,
    )
    pts_j = np.linalg.inv(t_wc_seq[j].astype(np.float64)) @ t_wc_seq[i].astype(np.float64) @ pts
    zj = pts_j[2]
    uj = k_j[0, 0] * pts_j[0] / zj + k_j[0, 2] - 0.5
    vj = k_j[1, 1] * pts_j[1] / zj + k_j[1, 2] - 0.5
    ok = (zj > 1e-3) & (uj >= 0) & (uj <= w - 1) & (vj >= 0) & (vj <= h - 1)
    assert ok.sum() >= 50, "too few reprojected points to assess consistency"
    dj = depth[j, np.round(vj[ok]).astype(int), np.round(uj[ok]).astype(int)]
    valid_j = depth_valid[j, np.round(vj[ok]).astype(int), np.round(uj[ok]).astype(int)]
    rel = np.abs(zj[ok][valid_j] - dj[valid_j]) / np.maximum(dj[valid_j], 1e-6)
    return float(np.median(rel))


def assert_sample_matches_frames(
    sample: dict,
    frames: list[SyntheticFrame],
    out_hw: tuple[int, int],
    depth_rel_tol: float = 0.01,
    rgb_tol: float = 5.0,
) -> None:
    clip_frames = len(frames)
    h, w = out_hw
    src_h, src_w = frames[0].depth.shape

    k_seq = sample["camera"]["K"].numpy()
    t_wc_seq = sample["camera"]["T_wc"].numpy()
    depth = sample["depth_m"].numpy()
    depth_valid = sample["depth_valid"].numpy()
    video = sample["video"].numpy()

    assert bool(sample["camera"]["camera_valid"].all())
    assert np.isclose(float(sample["aspect_ratio"][0]), src_w / src_h, atol=1e-5)

    for t, fr in enumerate(frames):
        np.testing.assert_allclose(
            k_seq[t], expected_resized_k(fr.k, (src_h, src_w), out_hw), rtol=1e-4, atol=1e-3
        )
        np.testing.assert_allclose(t_wc_seq[t], fr.t_wc.astype(np.float32), rtol=1e-5, atol=1e-5)

        expected_depth = cv2.resize(fr.depth, (w, h), interpolation=cv2.INTER_NEAREST)
        valid = depth_valid[t]
        assert valid.mean() > 0.5, f"frame {t}: too few valid depth pixels"
        rel = np.abs(depth[t][valid] - expected_depth[valid]) / np.maximum(expected_depth[valid], 1e-6)
        assert rel.max() < depth_rel_tol, f"frame {t}: max depth rel err {rel.max():.4f}"

        red_mean = float(video[t, 0].mean()) * 255.0
        expected_red = float(frame_signature_red(t))
        assert abs(red_mean - expected_red) < rgb_tol, (
            f"frame {t}: red signature {red_mean:.1f} != {expected_red:.1f} (frame order/pairing bug?)"
        )

    err = reprojection_median_rel_error(depth, depth_valid, k_seq, t_wc_seq, i=0, j=clip_frames - 1)
    assert err < 0.01, f"cross-frame reprojection inconsistency: median rel err {err:.4f}"


def assert_query_targets_consistent(sample: dict, depth_rel_tol: float = 0.02) -> None:
    query = sample["query"]
    target = sample["target"]
    mask = sample["mask"]
    depth = sample["depth_m"].numpy()
    depth_valid = sample["depth_valid"].numpy()
    t_hw = depth.shape

    u = query["u"].numpy()
    v = query["v"].numpy()
    t_src = query["t_src"].numpy()
    t_tgt = query["t_tgt"].numpy()
    t_cam = query["t_cam"].numpy()
    xyz = target["xyz_3d"].numpy()
    m_xyz = mask["xyz_3d"].numpy()

    assert m_xyz.any(), "no valid xyz queries produced"

    same = m_xyz & (t_src == t_tgt) & (t_tgt == t_cam)
    assert same.any(), "no same-frame queries to validate"
    px = np.clip(np.round(u[same] * t_hw[2] - 0.5).astype(int), 0, t_hw[2] - 1)
    py = np.clip(np.round(v[same] * t_hw[1] - 0.5).astype(int), 0, t_hw[1] - 1)
    d = depth[t_src[same], py, px]
    ok = depth_valid[t_src[same], py, px]
    assert ok.mean() > 0.9
    rel = np.abs(xyz[same][:, 2][ok] - d[ok]) / np.maximum(d[ok], 1e-6)
    assert np.median(rel) < depth_rel_tol, f"same-frame z vs depth: median rel {np.median(rel):.4f}"

    m_disp = mask["displacement"].numpy()
    if m_disp.any():
        disp = target["displacement"].numpy()[m_disp]
        assert np.abs(disp).max() < 1e-5, "static-scene displacement targets must be zero"
