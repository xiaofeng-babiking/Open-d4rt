"""Synthetic scene fixtures for VGGT-processed dataset adapter tests.

Generates frames of a static tilted plane with analytically exact depth,
known intrinsics (off-center principal point) and known cam2world poses,
then writes them in the exact on-disk layout of each VGGT-processed dataset
found under /jfs/Data_4DFF/train_data.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
from PIL import Image


@dataclass
class SyntheticFrame:
    rgb: np.ndarray  # [H, W, 3] uint8
    depth: np.ndarray  # [H, W] float32 meters
    k: np.ndarray  # [3, 3] float64
    t_wc: np.ndarray  # [4, 4] float64 cam2world


def _rot_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def frame_signature_red(t: int) -> int:
    """Per-frame red-channel signature used to verify frame ordering."""
    return 40 + 15 * t


def make_plane_frames(
    num_frames: int = 4,
    src_h: int = 48,
    src_w: int = 64,
) -> list[SyntheticFrame]:
    fx = 0.9 * src_w
    fy = 0.95 * src_w
    cx = src_w / 2.0 - 3.0
    cy = src_h / 2.0 + 2.0
    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    plane_n = np.array([0.06, -0.04, 1.0], dtype=np.float64)
    plane_p0 = np.array([0.0, 0.0, 5.0], dtype=np.float64)

    xs = (np.arange(src_w, dtype=np.float64) + 0.5)[None, :].repeat(src_h, axis=0)
    ys = (np.arange(src_h, dtype=np.float64) + 0.5)[:, None].repeat(src_w, axis=1)
    rays = np.stack([(xs - cx) / fx, (ys - cy) / fy, np.ones_like(xs)], axis=0)  # [3,H,W]

    frames: list[SyntheticFrame] = []
    for t in range(num_frames):
        r_wc = _rot_y(0.015 * t)
        cam_center = np.array([0.18 * t, 0.06 * t, -0.04 * t], dtype=np.float64)
        t_wc = np.eye(4, dtype=np.float64)
        t_wc[:3, :3] = r_wc
        t_wc[:3, 3] = cam_center

        dir_w = np.einsum("ij,jhw->ihw", r_wc, rays)
        denom = np.einsum("i,ihw->hw", plane_n, dir_w)
        s = float(plane_n @ (plane_p0 - cam_center)) / denom
        depth = s.astype(np.float32)  # rays have z == 1, so ray parameter == z_cam

        rgb = np.zeros((src_h, src_w, 3), dtype=np.uint8)
        rgb[..., 0] = frame_signature_red(t)
        rgb[..., 1] = (xs / src_w * 200.0).astype(np.uint8)
        rgb[..., 2] = (ys / src_h * 200.0).astype(np.uint8)

        frames.append(SyntheticFrame(rgb=rgb, depth=depth, k=k, t_wc=t_wc))
    return frames


def write_safetensor(path: Path, tensors: dict[str, np.ndarray]) -> None:
    dtype_names = {"float32": "F32", "float64": "F64"}
    header: dict[str, dict] = {}
    bufs: list[bytes] = []
    offset = 0
    for key, arr in tensors.items():
        arr = np.ascontiguousarray(arr)
        blob = arr.tobytes()
        header[key] = {
            "dtype": dtype_names[arr.dtype.name],
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(blob)],
        }
        bufs.append(blob)
        offset += len(blob)
    header_json = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        f.write(b"".join(bufs))


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    Image.fromarray(rgb).save(path)


def write_tartanair_scene(
    root: Path,
    frames: list[SyntheticFrame],
    env: str = "synthfactory",
    difficulty: str = "Easy",
    traj: str = "P000",
) -> Path:
    scene_dir = root / env / difficulty / traj
    scene_dir.mkdir(parents=True, exist_ok=True)
    for i, fr in enumerate(frames):
        _save_rgb(scene_dir / f"{i:06d}_rgb.png", fr.rgb)
        np.save(scene_dir / f"{i:06d}_depth.npy", fr.depth)
        np.savez(
            scene_dir / f"{i:06d}_cam.npz",
            camera_intrinsics=fr.k.astype(np.float32),
            camera_pose=fr.t_wc.astype(np.float32),
        )
    return scene_dir


def write_mvs_synth_scene(root: Path, frames: list[SyntheticFrame], clip: str = "0000") -> Path:
    scene_dir = root / clip
    for sub in ("rgb", "depth", "cam"):
        (scene_dir / sub).mkdir(parents=True, exist_ok=True)
    for i, fr in enumerate(frames):
        _save_rgb(scene_dir / "rgb" / f"{i:04d}.jpg", fr.rgb)
        np.save(scene_dir / "depth" / f"{i:04d}.npy", fr.depth)
        np.savez(
            scene_dir / "cam" / f"{i:04d}.npz",
            intrinsics=fr.k.astype(np.float32),
            pose=fr.t_wc.astype(np.float64),
        )
    return scene_dir


def write_scannet_scene(root: Path, frames: list[SyntheticFrame], scene: str = "scene0000_00") -> Path:
    scene_dir = root / scene
    for sub in ("color", "depth", "cam"):
        (scene_dir / sub).mkdir(parents=True, exist_ok=True)
    for i, fr in enumerate(frames):
        _save_rgb(scene_dir / "color" / f"{i:05d}.jpg", fr.rgb)
        depth_mm = np.clip(fr.depth * 1000.0, 0, 65535).astype(np.uint16)
        Image.fromarray(depth_mm).save(scene_dir / "depth" / f"{i:05d}.png")
        np.savez(
            scene_dir / "cam" / f"{i:05d}.npz",
            intrinsics=fr.k.astype(np.float32),
            pose=fr.t_wc.astype(np.float32),
        )
    return scene_dir


def write_blendermvs_scene(
    root: Path,
    frames: list[SyntheticFrame],
    scene: str = "000000000000000000000000",
    cam_format: str = "npz",
) -> Path:
    scene_dir = root / scene
    scene_dir.mkdir(parents=True, exist_ok=True)
    for i, fr in enumerate(frames):
        _save_rgb(scene_dir / f"{i:08d}.jpg", fr.rgb)
        ok = cv2.imwrite(str(scene_dir / f"{i:08d}.exr"), fr.depth.astype(np.float32))
        if not ok:
            raise RuntimeError("cv2 failed to write EXR (OpenEXR support missing?)")
        cam = {
            "intrinsics": fr.k.astype(np.float32),
            "R_cam2world": fr.t_wc[:3, :3].astype(np.float64),
            "t_cam2world": fr.t_wc[:3, 3].astype(np.float32),
        }
        if cam_format == "npz":
            np.savez(scene_dir / f"{i:08d}.npz", **cam)
        elif cam_format == "safetensor":
            write_safetensor(scene_dir / f"{i:08d}.safetensor", cam)
        else:
            raise ValueError(cam_format)
    return scene_dir


def write_co3d_sequence(
    root: Path,
    frames: list[SyntheticFrame],
    category: str = "apple",
    seq: str = "110_13072_25709",
    mask_pattern: str = "full",
) -> Path:
    seq_dir = root / category / seq
    for sub in ("images", "depths", "masks"):
        (seq_dir / sub).mkdir(parents=True, exist_ok=True)
    for i, fr in enumerate(frames):
        name = f"frame{i + 1:06d}"
        _save_rgb(seq_dir / "images" / f"{name}.jpg", fr.rgb)
        max_depth = float(np.nanmax(fr.depth)) * 1.25
        depth_u16 = np.clip(fr.depth / max_depth * 65535.0, 0, 65535).astype(np.uint16)
        Image.fromarray(depth_u16).save(seq_dir / "depths" / f"{name}.jpg.geometric.png")
        mask = np.full(fr.depth.shape, 255, dtype=np.uint8)
        if mask_pattern == "left_zero":
            mask[:, : fr.depth.shape[1] // 2] = 0
        elif mask_pattern != "full":
            raise ValueError(mask_pattern)
        Image.fromarray(mask).save(seq_dir / "masks" / f"{name}.png")
        np.savez(
            seq_dir / "images" / f"{name}.npz",
            camera_intrinsics=fr.k.astype(np.float64),
            camera_pose=fr.t_wc.astype(np.float32),
            maximum_depth=np.float32(max_depth),
        )
    return seq_dir


def write_vkitti_scene(
    root: Path,
    frames: list[SyntheticFrame],
    scene: str = "Scene01",
    variant: str = "clone",
    camera: str = "Camera_0",
) -> Path:
    cam_dir = root / scene / variant / camera
    cam_dir.mkdir(parents=True, exist_ok=True)
    for i, fr in enumerate(frames):
        _save_rgb(cam_dir / f"{i:05d}_rgb.jpg", fr.rgb)
        depth_cm = np.clip(fr.depth * 100.0, 0, 65535).astype(np.uint16)
        Image.fromarray(depth_cm).save(cam_dir / f"{i:05d}_depth.png")
        np.savez(
            cam_dir / f"{i:05d}_cam.npz",
            camera_intrinsics=fr.k.astype(np.float32),
            camera_pose=fr.t_wc.astype(np.float32),
        )
    return cam_dir
