"""Adapters for VGGT/DUSt3R-style preprocessed datasets.

These datasets (as found under e.g. /jfs/Data_4DFF/train_data) store per-frame
``rgb`` + metric ``depth`` + ``cam`` files, where every camera file carries
pinhole ``intrinsics`` and a **cam2world** ``pose`` (empirically verified via
cross-frame depth reprojection on every supported dataset).

All supported variants are static-scene datasets, so query supervision is
constructed with the shared :func:`build_queries_from_depth` reprojection
builder, exactly like the existing raw depth datasets (mvs_synth, scannet,
tartanair, ...).
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .bad_sample_registry import (
    BadSampleRegistry,
    RetryableSampleError,
    failed_paths_from_exception,
    is_retryable_data_error,
)
from .depth_query_builder import build_queries_from_depth
from .raw_augment import (
    RawAugmentConfig,
    apply_photometric_augment,
    apply_spatial_crop_images_only,
    build_augment_info,
    count_valid_depth_frames,
    sample_frame_indices_with_stride,
    sanitize_depth_map,
)
from .seeding import SeededDatasetMixin, stable_split_bucket

KNOWN_VARIANTS = ("tartanair", "mvs_synth", "scannet", "blendermvs", "co3d", "vkitti")

_VKITTI_DEPTH_SENTINEL = 65535  # raw uint16 value encoding "no depth" (655.35 m)


@dataclass
class VggtProcessedConfig:
    root: Path | None
    variant: str
    split: str
    clip_frames: int
    image_size: tuple[int, int]  # (H, W)
    queries_per_clip: int
    hard_query_ratio: float
    prob_t_tgt_equals_t_cam: float
    training: bool
    roots: tuple[Path, ...] | None = None  # multi-root variants (blendermvs)
    t_src_tgt_delta_choices: tuple[int | None, ...] | None = None
    t_src_tgt_delta_probs: tuple[float, ...] | None = None
    split_modulo: int = 20
    max_scenes: int | None = None
    max_depth_m: float = 1e5
    depth_clip_percentile: float = 0.0
    min_depth_valid_ratio: float = 0.0
    min_valid_frames_ratio: float = 0.0
    vkitti_variants: tuple[str, ...] = ("clone",)
    vkitti_cameras: tuple[str, ...] = ("Camera_0",)
    use_co3d_masks: bool = True
    static_consistency_filter: bool = False
    static_consistency_rel_threshold: float = 0.05
    augment: RawAugmentConfig | None = None
    bad_sample_registry_path: Path = field(default_factory=lambda: Path("data/meta/bad_sample.json"))
    max_sample_retries: int = 64
    # Directory for persisted scene indices. Full discovery walks every scene
    # dir under the root (≈13 min for the scannet root on /jfs), so training
    # configs should set this; delete the cache file to force a rescan.
    scene_index_cache: Path | None = None


@dataclass
class _Frame:
    frame_id: int
    rgb_path: Path
    depth_path: Path
    cam_path: Path
    mask_path: Path | None = None


@dataclass
class _Scene:
    scene_id: str
    frames: list[_Frame]


class _FilteredDepthSampleError(RetryableSampleError):
    """Sample-level quality filter; retry without persisting a bad-sample entry."""


def _scene_cache_file(cfg: VggtProcessedConfig) -> Path | None:
    if cfg.scene_index_cache is None:
        return None
    roots_token = "|".join(str(r) for r in (cfg.roots if cfg.roots else (cfg.root,)))
    digest = hashlib.sha1(roots_token.encode("utf-8")).hexdigest()[:12]
    return Path(cfg.scene_index_cache) / f"{cfg.variant}_{digest}.json"


def _scenes_to_payload(scenes: list[_Scene]) -> list[dict[str, Any]]:
    return [
        {
            "scene_id": scene.scene_id,
            "frames": [
                {
                    "frame_id": int(fr.frame_id),
                    "rgb": str(fr.rgb_path),
                    "depth": str(fr.depth_path),
                    "cam": str(fr.cam_path),
                    "mask": str(fr.mask_path) if fr.mask_path is not None else None,
                }
                for fr in scene.frames
            ],
        }
        for scene in scenes
    ]


def _scenes_from_payload(payload: list[dict[str, Any]]) -> list[_Scene]:
    scenes: list[_Scene] = []
    for entry in payload:
        frames = [
            _Frame(
                frame_id=int(fr["frame_id"]),
                rgb_path=Path(fr["rgb"]),
                depth_path=Path(fr["depth"]),
                cam_path=Path(fr["cam"]),
                mask_path=Path(fr["mask"]) if fr.get("mask") else None,
            )
            for fr in entry["frames"]
        ]
        scenes.append(_Scene(scene_id=str(entry["scene_id"]), frames=frames))
    return scenes


# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------


def _read_rgb(path: Path, width: int, height: int) -> tuple[np.ndarray, int, int]:
    """Read RGB, return (resized uint8 [H,W,3], src_w, src_h)."""
    try:
        img = Image.open(path).convert("RGB")
        src_w, src_h = img.size
        img = img.resize((width, height), resample=Image.Resampling.BILINEAR)
        return np.asarray(img, dtype=np.uint8), src_w, src_h
    except Exception as exc:
        raise RetryableSampleError(f"Failed to read RGB image: {path}: {exc}", failed_paths=[str(path)]) from exc


def _read_depth_npy(path: Path) -> np.ndarray:
    try:
        depth = np.load(path)
    except Exception as exc:
        raise RetryableSampleError(f"Failed to read depth npy: {path}: {exc}", failed_paths=[str(path)]) from exc
    if depth.ndim != 2:
        raise RetryableSampleError(f"Invalid depth npy shape for {path}: {depth.shape}", failed_paths=[str(path)])
    return depth.astype(np.float32, copy=False)


def _read_depth_png(path: Path, divisor: float, invalid_raw_values: tuple[int, ...] = ()) -> np.ndarray:
    try:
        raw = np.asarray(Image.open(path))
    except Exception as exc:
        raise RetryableSampleError(f"Failed to read depth png: {path}: {exc}", failed_paths=[str(path)]) from exc
    if raw.ndim != 2:
        raise RetryableSampleError(f"Invalid depth png shape for {path}: {raw.shape}", failed_paths=[str(path)])
    depth = raw.astype(np.float32) / float(divisor)
    for sentinel in invalid_raw_values:
        depth[raw == sentinel] = 0.0
    return depth


def _read_mask(path: Path, width: int, height: int) -> np.ndarray:
    try:
        mask = np.asarray(Image.open(path).convert("L"))
    except Exception as exc:
        raise RetryableSampleError(f"Failed to read mask: {path}: {exc}", failed_paths=[str(path)]) from exc
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask > 127


def _read_depth_exr(path: Path) -> np.ndarray:
    try:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    except Exception as exc:
        raise RetryableSampleError(f"Failed to read EXR depth: {path}: {exc}", failed_paths=[str(path)]) from exc
    if depth is None:
        raise RetryableSampleError(f"Failed to read EXR depth: {path}", failed_paths=[str(path)])
    if depth.ndim == 3:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise RetryableSampleError(f"Invalid EXR depth shape for {path}: {depth.shape}", failed_paths=[str(path)])
    return depth.astype(np.float32, copy=False)


def read_safetensor_arrays(path: Path) -> dict[str, np.ndarray]:
    """Minimal safetensors reader (header = length-prefixed JSON + raw buffers)."""
    dtype_map = {"F16": np.float16, "F32": np.float32, "F64": np.float64}
    try:
        with open(path, "rb") as f:
            (header_len,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(header_len))
            payload = f.read()
        out: dict[str, np.ndarray] = {}
        for key, spec in header.items():
            if key == "__metadata__":
                continue
            start, end = spec["data_offsets"]
            arr = np.frombuffer(payload[start:end], dtype=dtype_map[spec["dtype"]])
            out[key] = arr.reshape(spec["shape"]).copy()
        return out
    except Exception as exc:
        raise RetryableSampleError(f"Failed to read safetensor: {path}: {exc}", failed_paths=[str(path)]) from exc


def _cam_from_npz(path: Path, k_key: str, pose_key: str) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    try:
        with np.load(path) as data:
            k = np.asarray(data[k_key], dtype=np.float32)
            t_wc = np.asarray(data[pose_key], dtype=np.float32)
            extras = {}
            if "maximum_depth" in data.files:
                extras["maximum_depth"] = float(data["maximum_depth"])
    except Exception as exc:
        raise RetryableSampleError(f"Failed to read cam npz: {path}: {exc}", failed_paths=[str(path)]) from exc
    return _validate_cam(k, t_wc, path), t_wc.astype(np.float32), extras


def _validate_cam(k: np.ndarray, t_wc: np.ndarray, path: Path) -> np.ndarray:
    if k.shape != (3, 3) or t_wc.shape != (4, 4):
        raise RetryableSampleError(
            f"Invalid camera shapes in {path}: K={k.shape}, pose={t_wc.shape}", failed_paths=[str(path)]
        )
    if not (np.isfinite(k).all() and np.isfinite(t_wc).all()):
        raise RetryableSampleError(f"Non-finite camera values in {path}", failed_paths=[str(path)])
    det = float(np.linalg.det(t_wc[:3, :3].astype(np.float64)))
    if abs(det - 1.0) > 1e-2:
        raise RetryableSampleError(
            f"Camera pose rotation determinant is invalid in {path}: det={det:.6f}", failed_paths=[str(path)]
        )
    return k.astype(np.float32)


def _cam_from_rt_payload(
    payload: dict[str, np.ndarray], path: Path
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    try:
        k = np.asarray(payload["intrinsics"], dtype=np.float32)
        t_wc = np.eye(4, dtype=np.float32)
        t_wc[:3, :3] = np.asarray(payload["R_cam2world"], dtype=np.float32)
        t_wc[:3, 3] = np.asarray(payload["t_cam2world"], dtype=np.float32).reshape(3)
    except Exception as exc:
        raise RetryableSampleError(f"Failed to parse R/t camera payload: {path}: {exc}", failed_paths=[str(path)]) from exc
    return _validate_cam(k, t_wc, path), t_wc, {}


# ---------------------------------------------------------------------------
# Per-variant frame decoding
# ---------------------------------------------------------------------------


def _decode_cam_tartanair(frame: _Frame) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    return _cam_from_npz(frame.cam_path, "camera_intrinsics", "camera_pose")


def _decode_depth_tartanair(frame: _Frame, extras: dict[str, float]) -> np.ndarray:
    return _read_depth_npy(frame.depth_path)


def _decode_cam_mvs_synth(frame: _Frame) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    return _cam_from_npz(frame.cam_path, "intrinsics", "pose")


def _decode_depth_scannet(frame: _Frame, extras: dict[str, float]) -> np.ndarray:
    return _read_depth_png(frame.depth_path, divisor=1000.0)


def _decode_cam_blendermvs(frame: _Frame) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if frame.cam_path.suffix == ".safetensor":
        payload = read_safetensor_arrays(frame.cam_path)
    else:
        try:
            with np.load(frame.cam_path) as data:
                payload = {key: np.asarray(data[key]) for key in data.files}
        except Exception as exc:
            raise RetryableSampleError(
                f"Failed to read cam npz: {frame.cam_path}: {exc}", failed_paths=[str(frame.cam_path)]
            ) from exc
    return _cam_from_rt_payload(payload, frame.cam_path)


def _decode_depth_blendermvs(frame: _Frame, extras: dict[str, float]) -> np.ndarray:
    return _read_depth_exr(frame.depth_path)


def _decode_cam_co3d(frame: _Frame) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    return _cam_from_npz(frame.cam_path, "camera_intrinsics", "camera_pose")


def _decode_depth_co3d(frame: _Frame, extras: dict[str, float]) -> np.ndarray:
    maximum_depth = extras.get("maximum_depth")
    if maximum_depth is None or not np.isfinite(maximum_depth) or maximum_depth <= 0:
        raise RetryableSampleError(
            f"co3d frame missing usable maximum_depth for {frame.depth_path}",
            failed_paths=[str(frame.cam_path)],
        )
    return _read_depth_png(frame.depth_path, divisor=65535.0 / float(maximum_depth))


def _decode_cam_vkitti(frame: _Frame) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    return _cam_from_npz(frame.cam_path, "camera_intrinsics", "camera_pose")


def _decode_depth_vkitti(frame: _Frame, extras: dict[str, float]) -> np.ndarray:
    return _read_depth_png(frame.depth_path, divisor=100.0, invalid_raw_values=(_VKITTI_DEPTH_SENTINEL,))


@dataclass(frozen=True)
class _VariantSpec:
    discover: Callable[[VggtProcessedConfig], list[_Scene]]
    decode_cam: Callable[[_Frame], tuple[np.ndarray, np.ndarray, dict[str, float]]]
    decode_depth: Callable[[_Frame, dict[str, float]], np.ndarray]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _index_by_id(paths: list[Path], id_parser: Callable[[str], int | None]) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for p in paths:
        frame_id = id_parser(p.name)
        if frame_id is not None:
            out[frame_id] = p
    return out


def _suffix_id_parser(suffix: str) -> Callable[[str], int | None]:
    def parse(name: str) -> int | None:
        stem = name.split(".")[0]
        if not stem.endswith(suffix):
            return None
        token = stem[: -len(suffix)]
        return int(token) if token.isdigit() else None

    return parse


def _frames_from_suffix_dir(leaf_dir: Path) -> list[_Frame]:
    """Frames named ``{id}_rgb.*`` / ``{id}_depth.*`` / ``{id}_cam.npz`` in one dir."""
    rgb = _index_by_id(
        sorted(list(leaf_dir.glob("*_rgb.png")) + list(leaf_dir.glob("*_rgb.jpg"))), _suffix_id_parser("_rgb")
    )
    depth = _index_by_id(
        sorted(list(leaf_dir.glob("*_depth.npy")) + list(leaf_dir.glob("*_depth.png"))), _suffix_id_parser("_depth")
    )
    cam = _index_by_id(sorted(leaf_dir.glob("*_cam.npz")), _suffix_id_parser("_cam"))
    common = sorted(set(rgb) & set(depth) & set(cam))
    return [_Frame(frame_id=i, rgb_path=rgb[i], depth_path=depth[i], cam_path=cam[i]) for i in common]


def _numeric_id_parser(name: str) -> int | None:
    stem = name.split(".")[0]
    return int(stem) if stem.isdigit() else None


def _co3d_id_parser(name: str) -> int | None:
    stem = name.split(".")[0]
    if not stem.startswith("frame"):
        return None
    token = stem[len("frame"):]
    return int(token) if token.isdigit() else None


def _frames_from_subdirs(
    scene_dir: Path,
    rgb_dir: str,
    depth_dir: str,
    cam_dir: str,
    rgb_exts: tuple[str, ...],
    depth_exts: tuple[str, ...],
) -> list[_Frame]:
    rgb_root = scene_dir / rgb_dir
    depth_root = scene_dir / depth_dir
    cam_root = scene_dir / cam_dir
    if not (rgb_root.is_dir() and depth_root.is_dir() and cam_root.is_dir()):
        return []
    rgb = _index_by_id(
        sorted(p for ext in rgb_exts for p in rgb_root.glob(f"*{ext}")), _numeric_id_parser
    )
    depth = _index_by_id(
        sorted(p for ext in depth_exts for p in depth_root.glob(f"*{ext}")), _numeric_id_parser
    )
    cam = _index_by_id(sorted(cam_root.glob("*.npz")), _numeric_id_parser)
    common = sorted(set(rgb) & set(depth) & set(cam))
    return [_Frame(frame_id=i, rgb_path=rgb[i], depth_path=depth[i], cam_path=cam[i]) for i in common]


def _discover_tartanair(cfg: VggtProcessedConfig) -> list[_Scene]:
    root = Path(cfg.root or "")
    scenes: list[_Scene] = []
    for env_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for diff_dir in sorted(p for p in env_dir.iterdir() if p.is_dir()):
            for traj_dir in sorted(p for p in diff_dir.iterdir() if p.is_dir()):
                frames = _frames_from_suffix_dir(traj_dir)
                if frames:
                    scene_id = f"{env_dir.name}/{diff_dir.name}/{traj_dir.name}"
                    scenes.append(_Scene(scene_id=scene_id, frames=frames))
    return scenes


def _discover_mvs_synth(cfg: VggtProcessedConfig) -> list[_Scene]:
    root = Path(cfg.root or "")
    scenes: list[_Scene] = []
    for clip_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        frames = _frames_from_subdirs(
            clip_dir, "rgb", "depth", "cam", rgb_exts=(".jpg", ".png"), depth_exts=(".npy",)
        )
        if frames:
            scenes.append(_Scene(scene_id=clip_dir.name, frames=frames))
    return scenes


def _discover_scannet(cfg: VggtProcessedConfig) -> list[_Scene]:
    root = Path(cfg.root or "")
    scenes: list[_Scene] = []
    for scene_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        frames = _frames_from_subdirs(
            scene_dir, "color", "depth", "cam", rgb_exts=(".jpg", ".png"), depth_exts=(".png",)
        )
        if frames:
            scenes.append(_Scene(scene_id=scene_dir.name, frames=frames))
    return scenes


def _discover_blendermvs(cfg: VggtProcessedConfig) -> list[_Scene]:
    roots = cfg.roots if cfg.roots else (cfg.root,)
    scenes: list[_Scene] = []
    for root in roots:
        root = Path(root or "")
        for scene_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            rgb = _index_by_id(sorted(scene_dir.glob("*.jpg")), _numeric_id_parser)
            depth = _index_by_id(sorted(scene_dir.glob("*.exr")), _numeric_id_parser)
            cam = _index_by_id(
                sorted(list(scene_dir.glob("*.npz")) + list(scene_dir.glob("*.safetensor"))),
                _numeric_id_parser,
            )
            common = sorted(set(rgb) & set(depth) & set(cam))
            frames = [
                _Frame(frame_id=i, rgb_path=rgb[i], depth_path=depth[i], cam_path=cam[i]) for i in common
            ]
            if frames:
                scenes.append(_Scene(scene_id=f"{root.name}/{scene_dir.name}", frames=frames))
    return scenes


def _discover_co3d(cfg: VggtProcessedConfig) -> list[_Scene]:
    root = Path(cfg.root or "")
    scenes: list[_Scene] = []
    for category_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for seq_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
            images_dir = seq_dir / "images"
            depths_dir = seq_dir / "depths"
            masks_dir = seq_dir / "masks"
            if not (images_dir.is_dir() and depths_dir.is_dir()):
                continue
            # One directory listing per kind: per-frame exists() probes are ruinously
            # slow on network filesystems (hundreds of stat round-trips per sequence).
            rgb = _index_by_id(sorted(images_dir.glob("frame*.jpg")), _co3d_id_parser)
            cam = _index_by_id(sorted(images_dir.glob("frame*.npz")), _co3d_id_parser)
            depth = _index_by_id(sorted(depths_dir.glob("frame*.geometric.png")), _co3d_id_parser)
            masks = (
                _index_by_id(sorted(masks_dir.glob("frame*.png")), _co3d_id_parser)
                if cfg.use_co3d_masks and masks_dir.is_dir()
                else {}
            )
            frames: list[_Frame] = []
            for i in sorted(set(rgb) & set(cam) & set(depth)):
                frames.append(
                    _Frame(
                        frame_id=i,
                        rgb_path=rgb[i],
                        depth_path=depth[i],
                        cam_path=cam[i],
                        mask_path=masks.get(i),
                    )
                )
            if frames:
                scenes.append(_Scene(scene_id=f"{category_dir.name}/{seq_dir.name}", frames=frames))
    return scenes


def _discover_vkitti(cfg: VggtProcessedConfig) -> list[_Scene]:
    root = Path(cfg.root or "")
    scenes: list[_Scene] = []
    for scene_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for variant_name in cfg.vkitti_variants:
            variant_dir = scene_dir / variant_name
            if not variant_dir.is_dir():
                continue
            for camera_name in cfg.vkitti_cameras:
                camera_dir = variant_dir / camera_name
                if not camera_dir.is_dir():
                    continue
                frames = _frames_from_suffix_dir(camera_dir)
                if frames:
                    scene_id = f"{scene_dir.name}/{variant_name}/{camera_name}"
                    scenes.append(_Scene(scene_id=scene_id, frames=frames))
    return scenes


_VARIANTS: dict[str, _VariantSpec] = {
    "tartanair": _VariantSpec(
        discover=_discover_tartanair,
        decode_cam=_decode_cam_tartanair,
        decode_depth=_decode_depth_tartanair,
    ),
    "mvs_synth": _VariantSpec(
        discover=_discover_mvs_synth,
        decode_cam=_decode_cam_mvs_synth,
        decode_depth=_decode_depth_tartanair,
    ),
    "scannet": _VariantSpec(
        discover=_discover_scannet,
        decode_cam=_decode_cam_mvs_synth,
        decode_depth=_decode_depth_scannet,
    ),
    "blendermvs": _VariantSpec(
        discover=_discover_blendermvs,
        decode_cam=_decode_cam_blendermvs,
        decode_depth=_decode_depth_blendermvs,
    ),
    "co3d": _VariantSpec(
        discover=_discover_co3d,
        decode_cam=_decode_cam_co3d,
        decode_depth=_decode_depth_co3d,
    ),
    "vkitti": _VariantSpec(
        discover=_discover_vkitti,
        decode_cam=_decode_cam_vkitti,
        decode_depth=_decode_depth_vkitti,
    ),
}


def mask_temporally_inconsistent_depth(
    depth: np.ndarray,
    depth_valid: np.ndarray,
    k_seq: np.ndarray,
    t_wc_seq: np.ndarray,
    camera_valid: np.ndarray,
    rel_threshold: float,
) -> np.ndarray:
    """Invalidate pixels whose depth violates static-scene consistency vs adjacent frames.

    For each frame ``i`` and neighbor ``j``, every valid pixel of ``i`` is
    unprojected, reprojected into ``j`` assuming a static world, and compared
    against ``j``'s observed depth. Pixels whose relative depth mismatch exceeds
    ``rel_threshold`` in any checkable direction (moving objects, but also
    occlusions — conservative) are marked invalid. Pixels that project out of
    bounds or onto invalid depth stay untouched.
    """
    t, h, w = depth.shape
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    bad = np.zeros_like(depth_valid)
    for i in range(t):
        if not camera_valid[i]:
            continue
        for j in (i - 1, i + 1):
            if j < 0 or j >= t or not camera_valid[j]:
                continue
            z = depth[i].astype(np.float64)
            k_i = k_seq[i].astype(np.float64)
            k_j = k_seq[j].astype(np.float64)
            pts = np.stack(
                [
                    (xs + 0.5 - k_i[0, 2]) / k_i[0, 0] * z,
                    (ys + 0.5 - k_i[1, 2]) / k_i[1, 1] * z,
                    z,
                    np.ones_like(z),
                ],
                axis=0,
            ).reshape(4, -1)
            rel_t = np.linalg.inv(t_wc_seq[j].astype(np.float64)) @ t_wc_seq[i].astype(np.float64)
            pts_j = rel_t @ pts
            zj = pts_j[2].reshape(h, w)
            safe_z = np.maximum(pts_j[2], 1e-9)
            # Invalid source depth may be NaN; route it far out of bounds before the cast.
            uj_f = np.nan_to_num(k_j[0, 0] * pts_j[0] / safe_z + k_j[0, 2] - 0.5, nan=-1e9, posinf=-1e9, neginf=-1e9)
            vj_f = np.nan_to_num(k_j[1, 1] * pts_j[1] / safe_z + k_j[1, 2] - 0.5, nan=-1e9, posinf=-1e9, neginf=-1e9)
            uj = np.round(np.clip(uj_f, -1e9, 1e9)).astype(np.int64).reshape(h, w)
            vj = np.round(np.clip(vj_f, -1e9, 1e9)).astype(np.int64).reshape(h, w)
            inb = depth_valid[i] & (zj > 1e-3) & (uj >= 0) & (uj < w) & (vj >= 0) & (vj < h)
            uc = np.clip(uj, 0, w - 1)
            vc = np.clip(vj, 0, h - 1)
            dj = depth[j, vc, uc]
            checked = inb & depth_valid[j, vc, uc] & (dj > 1e-3)
            rel = np.abs(zj - dj) / np.maximum(dj, 1e-6)
            bad[i] |= checked & (rel > float(rel_threshold))
    return depth_valid & ~bad


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class VggtProcessedDataset(SeededDatasetMixin, Dataset):
    """Loads VGGT/DUSt3R-processed clips and emits D4RT-compatible batches."""

    def __init__(self, config: VggtProcessedConfig) -> None:
        if config.variant not in _VARIANTS:
            raise ValueError(
                f"Unknown VGGT-processed variant: {config.variant!r}; supported: {sorted(_VARIANTS)}"
            )
        self.cfg = config
        self.spec = _VARIANTS[config.variant]
        self.h, self.w = config.image_size
        self._init_dataset_seeding(namespace=f"vggt_{config.variant}", default_seed=20260610)
        self.augment = config.augment or RawAugmentConfig()
        if not config.training:
            self.augment = RawAugmentConfig()
        self.bad_registry = BadSampleRegistry(path=config.bad_sample_registry_path)
        self.max_sample_retries = max(1, int(config.max_sample_retries))

        roots = config.roots if config.roots else (config.root,)
        for root in roots:
            if root is None or not Path(root).exists():
                raise FileNotFoundError(f"VGGT-processed root not found: {root}")

        cache_file = _scene_cache_file(config)
        scenes: list[_Scene] | None = None
        if cache_file is not None and cache_file.exists():
            try:
                scenes = _scenes_from_payload(json.loads(cache_file.read_text(encoding="utf-8")))
            except Exception:
                scenes = None  # unreadable cache: fall through to a fresh scan
        if scenes is None:
            scenes = self.spec.discover(config)
            if cache_file is not None:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(_scenes_to_payload(scenes)), encoding="utf-8")

        self.scenes: list[_Scene] = []
        for scene in scenes:
            if not self._in_split(scene.scene_id):
                continue
            if len(scene.frames) < int(config.clip_frames):
                continue
            self.scenes.append(scene)
            if config.max_scenes is not None and len(self.scenes) >= int(config.max_scenes):
                break

        if not self.scenes:
            raise ValueError(
                f"No valid {config.variant} VGGT-processed scenes for split={config.split} under {roots}"
            )

    def _in_split(self, scene_id: str) -> bool:
        split = str(self.cfg.split).strip().lower()
        if split == "all":
            return True
        modulo = max(3, int(self.cfg.split_modulo))
        bucket = stable_split_bucket(scene_id, modulo=modulo)
        val_bucket = modulo - 2
        test_bucket = modulo - 1
        if split == "val":
            return bucket == val_bucket
        if split == "test":
            return bucket == test_bucket
        return bucket < val_bucket

    def __len__(self) -> int:
        base = len(self.scenes) * 30 if self.cfg.training else len(self.scenes)
        return max(base, len(self.scenes))

    def _scene(self, index: int) -> _Scene:
        if self.cfg.training:
            sid = int(self.rng.integers(0, len(self.scenes)))
            return self.scenes[sid]
        return self.scenes[index % len(self.scenes)]

    def _frame_indices(self, scene_len: int, index: int) -> list[int]:
        return sample_frame_indices_with_stride(
            rng=self.rng,
            scene_len=scene_len,
            clip_frames=int(self.cfg.clip_frames),
            cfg=self.augment,
            training=bool(self.cfg.training),
            index=index,
        )

    def _sample_key(self, scene: _Scene, idxs: list[int]) -> str:
        frame_token = ",".join(str(int(scene.frames[i].frame_id)) for i in idxs)
        return f"{self.cfg.variant}_vggt::{scene.scene_id}::frames={frame_token}"

    def _sample_paths(self, scene: _Scene, idxs: list[int]) -> list[str]:
        out: list[str] = []
        for i in idxs:
            frame = scene.frames[i]
            out.extend([str(frame.rgb_path), str(frame.depth_path), str(frame.cam_path)])
            if frame.mask_path is not None:
                out.append(str(frame.mask_path))
        return out

    def _build_sample(self, scene: _Scene, idxs: list[int], clip_start: int) -> dict[str, Any]:
        video_list: list[np.ndarray] = []
        depth_list: list[np.ndarray] = []
        depth_valid_list: list[np.ndarray] = []
        k_seq: list[np.ndarray] = []
        t_wc_seq: list[np.ndarray] = []
        camera_valid: list[bool] = []
        first_src_hw: tuple[int, int] | None = None

        for i in idxs:
            frame = scene.frames[i]
            rgb, src_w, src_h = _read_rgb(frame.rgb_path, width=self.w, height=self.h)
            if first_src_hw is None:
                first_src_hw = (src_h, src_w)
            k_src, t_wc, extras = self.spec.decode_cam(frame)
            depth_m = self.spec.decode_depth(frame, extras)
            if depth_m.shape != (self.h, self.w):
                depth_m = cv2.resize(depth_m, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
            depth_m, valid = sanitize_depth_map(
                depth_m,
                max_depth_m=float(self.cfg.max_depth_m),
                depth_clip_percentile=float(self.cfg.depth_clip_percentile),
                min_valid_ratio=float(self.cfg.min_depth_valid_ratio),
            )
            if frame.mask_path is not None:
                valid = valid & _read_mask(frame.mask_path, width=self.w, height=self.h)

            k = k_src.astype(np.float32).copy()
            sx = float(self.w) / max(float(src_w), 1.0)
            sy = float(self.h) / max(float(src_h), 1.0)
            k[0, 0] *= sx
            k[0, 2] *= sx
            k[1, 1] *= sy
            k[1, 2] *= sy

            video_list.append(rgb)
            depth_list.append(depth_m)
            depth_valid_list.append(valid.astype(np.bool_))
            k_seq.append(k)
            t_wc_seq.append(t_wc.astype(np.float32))
            camera_valid.append(bool(np.isfinite(k).all() and np.isfinite(t_wc).all()))

        video = np.stack(video_list, axis=0).astype(np.float32) / 255.0
        video = np.transpose(video, (0, 3, 1, 2))
        depth = np.stack(depth_list, axis=0).astype(np.float32)
        depth_valid = np.stack(depth_valid_list, axis=0).astype(np.bool_)
        k_arr = np.stack(k_seq, axis=0).astype(np.float32)
        t_wc_arr = np.stack(t_wc_seq, axis=0).astype(np.float32)
        cam_valid = np.asarray(camera_valid, dtype=np.bool_)

        src_h, src_w = first_src_hw if first_src_hw else (self.h, self.w)
        aspect_ratio = np.array([src_w / max(float(src_h), 1.0)], dtype=np.float32)
        _crop_info: dict[str, Any] = {}
        if self.cfg.training:
            video = apply_photometric_augment(video_t_chw=video, rng=self.rng, cfg=self.augment)
            (video, depth, depth_valid, k_arr, aspect_ratio) = apply_spatial_crop_images_only(
                video_t_chw=video,
                depth_t_hw=depth,
                depth_valid_t_hw=depth_valid,
                k_t_33=k_arr,
                camera_valid_t=cam_valid,
                rng=self.rng,
                cfg=self.augment,
                native_aspect_ratio=aspect_ratio,
                out_info=_crop_info,
            )

        if self.cfg.static_consistency_filter:
            depth_valid = mask_temporally_inconsistent_depth(
                depth=depth,
                depth_valid=depth_valid,
                k_seq=k_arr,
                t_wc_seq=t_wc_arr,
                camera_valid=cam_valid,
                rel_threshold=float(self.cfg.static_consistency_rel_threshold),
            )

        min_valid_frames_ratio = max(0.0, min(1.0, float(self.cfg.min_valid_frames_ratio)))
        if min_valid_frames_ratio > 0.0:
            min_frames = max(1, int(np.ceil(float(len(idxs)) * min_valid_frames_ratio)))
            valid_frames = count_valid_depth_frames(
                depth_valid,
                min_valid_ratio=float(self.cfg.min_depth_valid_ratio),
            )
            if valid_frames < min_frames:
                raise _FilteredDepthSampleError(
                    f"{self.cfg.variant} VGGT clip has too few valid depth frames: "
                    f"{valid_frames}/{len(idxs)} < {min_frames}"
                )

        query, target, mask, query_stats = build_queries_from_depth(
            rng=self.rng,
            depth=depth,
            depth_valid=depth_valid,
            k_seq=k_arr,
            t_wc_seq=t_wc_arr,
            camera_valid=cam_valid,
            queries_per_clip=int(self.cfg.queries_per_clip),
            hard_query_ratio=float(self.cfg.hard_query_ratio),
            prob_t_tgt_equals_t_cam=float(self.cfg.prob_t_tgt_equals_t_cam),
            t_src_tgt_delta_choices=self.cfg.t_src_tgt_delta_choices,
            t_src_tgt_delta_probs=self.cfg.t_src_tgt_delta_probs,
        )

        return {
            "video": torch.from_numpy(video).float(),
            "aspect_ratio": torch.from_numpy(aspect_ratio.astype(np.float32)),
            "depth_m": torch.from_numpy(depth).float(),
            "depth_valid": torch.from_numpy(depth_valid).bool(),
            "query": {
                k: torch.from_numpy(v).to(torch.long if k.startswith("t_") else torch.float32)
                for k, v in query.items()
            },
            "query_stats": {k: torch.from_numpy(v).bool() for k, v in query_stats.items()},
            "target": {k: torch.from_numpy(v).float() for k, v in target.items()},
            "mask": {k: torch.from_numpy(v).bool() for k, v in mask.items()},
            "camera": {
                "K": torch.from_numpy(k_arr).float(),
                "T_wc": torch.from_numpy(t_wc_arr).float(),
                "camera_valid": torch.from_numpy(cam_valid).bool(),
            },
            "augment_info": {
                k: torch.from_numpy(v) for k, v in build_augment_info(_crop_info, image_hw=(self.h, self.w)).items()
            },
            "meta": {
                "dataset": f"{self.cfg.variant}_vggt",
                "scene_id": scene.scene_id,
                "clip_start": int(clip_start),
                "source_mode": "vggt_processed_depth_reproject",
                "native_hw": (src_h, src_w),
            },
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        last_error: Exception | None = None
        total = max(1, len(self))

        for attempt in range(self.max_sample_retries):
            query_index, _ = self._prepare_sample_rng(index=index, total=total, attempt=attempt)
            scene = self._scene(query_index)
            idxs = self._frame_indices(len(scene.frames), query_index)
            clip_start = int(idxs[0]) if idxs else 0
            sample_key = self._sample_key(scene, idxs)
            sample_paths = self._sample_paths(scene, idxs)

            if self.bad_registry.is_bad_sample(sample_key):
                continue
            if self.bad_registry.has_any_bad_path(sample_paths):
                continue

            try:
                sample = self._build_sample(scene=scene, idxs=idxs, clip_start=clip_start)
            except _FilteredDepthSampleError as exc:
                last_error = exc
                continue
            except Exception as exc:
                if not is_retryable_data_error(exc):
                    raise
                last_error = exc
                self.bad_registry.mark_bad(
                    dataset=f"{self.cfg.variant}_vggt",
                    sample_key=sample_key,
                    sample_paths=sample_paths,
                    failed_paths=failed_paths_from_exception(exc),
                    error=f"{type(exc).__name__}: {exc}",
                )
                continue

            sample["meta"]["sample_key"] = sample_key
            return sample

        raise RuntimeError(
            f"VggtProcessedDataset({self.cfg.variant}) failed to produce a valid sample after "
            f"{self.max_sample_retries} retries. last_error={last_error}"
        )
