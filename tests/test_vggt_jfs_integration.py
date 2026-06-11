"""Integration tests: VGGT-processed adapters against the real /jfs data.

Each test symlinks a small subset of real scenes into a tmp root so dataset
discovery stays bounded, then validates sample schema and cross-frame
geometric consistency (depth + intrinsics + cam2world pose must reproject
onto each other within tight tolerance on these static scenes).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from helpers_validation import assert_sample_schema, reprojection_median_rel_error

from src.data.vggt_processed_dataset import VggtProcessedConfig, VggtProcessedDataset

JFS_ROOT = Path("/jfs/Data_4DFF/train_data")

pytestmark = pytest.mark.skipif(not JFS_ROOT.exists(), reason="/jfs training data not mounted")

CLIP_FRAMES = 16
OUT_HW = (64, 64)


def jfs_config(root: Path | None, variant: str, tmp_path: Path, **overrides) -> VggtProcessedConfig:
    kwargs = dict(
        root=root,
        variant=variant,
        split="all",
        clip_frames=CLIP_FRAMES,
        image_size=OUT_HW,
        queries_per_clip=512,
        hard_query_ratio=0.2,
        prob_t_tgt_equals_t_cam=0.4,
        training=False,
        bad_sample_registry_path=tmp_path / "bad_samples.json",
    )
    kwargs.update(overrides)
    return VggtProcessedConfig(**kwargs)


def _symlink_children(src: Path, dst: Path, names: list[str]) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = src / name
        assert target.exists(), f"expected real data at {target}"
        os.symlink(target, dst / name)
    return dst


def _check_sample(sample, *, min_valid_ratio: float = 0.2, adjacent_tol: float = 0.05) -> None:
    assert_sample_schema(sample, CLIP_FRAMES, OUT_HW)
    depth = sample["depth_m"].numpy()
    valid = sample["depth_valid"].numpy()
    k_seq = sample["camera"]["K"].numpy()
    t_wc = sample["camera"]["T_wc"].numpy()

    assert valid.mean() > min_valid_ratio, f"valid depth ratio too low: {valid.mean():.3f}"
    med = float(__import__("numpy").median(depth[valid]))
    assert 0.1 < med < 2000.0, f"implausible median depth: {med}"

    err = reprojection_median_rel_error(depth, valid, k_seq, t_wc, i=0, j=1)
    assert err < adjacent_tol, f"adjacent-frame reprojection median rel err {err:.4f}"


class TestJfsTartanair:
    def test_real_sample(self, tmp_path):
        root = _symlink_children(JFS_ROOT / "tartanair" / "train", tmp_path / "root", ["abandonedfactory"])
        ds = VggtProcessedDataset(jfs_config(root, "tartanair", tmp_path))
        assert len(ds.scenes) >= 2  # Easy + Hard trajectories
        _check_sample(ds[0])


class TestJfsMvsSynth:
    def test_real_sample(self, tmp_path):
        root = _symlink_children(JFS_ROOT / "mvs_synth" / "train", tmp_path / "root", ["0000", "0001"])
        ds = VggtProcessedDataset(
            jfs_config(root, "mvs_synth", tmp_path, max_depth_m=1000.0, depth_clip_percentile=98.0)
        )
        assert len(ds.scenes) == 2
        _check_sample(ds[0])


class TestJfsScannet:
    def test_real_sample(self, tmp_path):
        root = _symlink_children(
            JFS_ROOT / "scannet" / "scans_train", tmp_path / "root", ["scene0000_00", "scene0001_00"]
        )
        ds = VggtProcessedDataset(jfs_config(root, "scannet", tmp_path))
        assert len(ds.scenes) == 2
        _check_sample(ds[0])


class TestJfsBlendermvs:
    def test_real_sample_both_formats(self, tmp_path):
        root_new = _symlink_children(
            JFS_ROOT / "blendedmvs", tmp_path / "root_new", ["000000000000000000000000"]
        )
        root_prev = _symlink_children(
            JFS_ROOT / "blendedmvs_previous" / "train", tmp_path / "root_prev", ["57f8d9bbe73f6760f10e916a"]
        )
        ds = VggtProcessedDataset(
            jfs_config(
                None,
                "blendermvs",
                tmp_path,
                roots=(root_new, root_prev),
                max_depth_m=1000.0,
                depth_clip_percentile=98.0,
            )
        )
        assert len(ds.scenes) == 2
        for index in range(2):  # one scene per processed format (safetensor + npz)
            _check_sample(ds[index])


class TestJfsCo3d:
    def test_real_sample(self, tmp_path):
        category_root = tmp_path / "root" / "apple"
        _symlink_children(
            JFS_ROOT / "co3d" / "apple", category_root, ["110_13072_25709", "151_16773_32218"]
        )
        ds = VggtProcessedDataset(jfs_config(tmp_path / "root", "co3d", tmp_path))
        assert len(ds.scenes) == 2
        # Foreground masks gate most of the background out, so the valid ratio is low.
        _check_sample(ds[0], min_valid_ratio=0.02, adjacent_tol=0.08)
        sample = ds[0]
        assert sample["depth_valid"].numpy().mean() < 0.95, "co3d masks should gate background depth"


class TestJfsVkitti:
    def test_real_sample_with_static_filter(self, tmp_path):
        root = _symlink_children(JFS_ROOT / "vkitti" / "train", tmp_path / "root", ["Scene01"])
        ds = VggtProcessedDataset(
            jfs_config(
                root,
                "vkitti",
                tmp_path,
                max_depth_m=655.0,
                static_consistency_filter=True,
            )
        )
        assert len(ds.scenes) == 1
        _check_sample(ds[0])
