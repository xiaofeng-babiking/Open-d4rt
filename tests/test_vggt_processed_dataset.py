"""Tests for the VGGT-processed dataset adapters (synthetic fixtures)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from helpers_synthetic import (
    SyntheticFrame,
    make_plane_frames,
    write_blendermvs_scene,
    write_co3d_sequence,
    write_mvs_synth_scene,
    write_scannet_scene,
    write_tartanair_scene,
    write_vkitti_scene,
)
from helpers_validation import (
    assert_query_targets_consistent,
    assert_sample_matches_frames,
    assert_sample_schema,
)

from src.data.vggt_processed_dataset import VggtProcessedConfig, VggtProcessedDataset

CLIP_FRAMES = 4
OUT_HW = (32, 32)


def make_config(root: Path | None, variant: str, tmp_path: Path, **overrides) -> VggtProcessedConfig:
    kwargs = dict(
        root=root,
        variant=variant,
        split="all",
        clip_frames=CLIP_FRAMES,
        image_size=OUT_HW,
        queries_per_clip=256,
        hard_query_ratio=0.2,
        prob_t_tgt_equals_t_cam=0.4,
        training=False,
        bad_sample_registry_path=tmp_path / "bad_samples.json",
    )
    kwargs.update(overrides)
    return VggtProcessedConfig(**kwargs)


@dataclass
class VariantCase:
    name: str
    variant: str
    write_scene: Callable[[Path, list[SyntheticFrame]], Path]
    depth_rel_tol: float = 0.01
    config_overrides: dict = field(default_factory=dict)


CASES = [
    VariantCase("tartanair", "tartanair", write_tartanair_scene),
    VariantCase("mvs_synth", "mvs_synth", write_mvs_synth_scene),
    VariantCase("scannet", "scannet", write_scannet_scene),
    VariantCase(
        "blendermvs_npz",
        "blendermvs",
        lambda root, frames: write_blendermvs_scene(root, frames, cam_format="npz"),
    ),
    VariantCase(
        "blendermvs_safetensor",
        "blendermvs",
        lambda root, frames: write_blendermvs_scene(root, frames, cam_format="safetensor"),
    ),
    VariantCase("co3d", "co3d", write_co3d_sequence),
    VariantCase("vkitti", "vkitti", write_vkitti_scene),
]


@pytest.fixture(params=CASES, ids=[c.name for c in CASES])
def variant_dataset(request, tmp_path):
    case: VariantCase = request.param
    frames = make_plane_frames(num_frames=CLIP_FRAMES)
    root = tmp_path / case.name
    root.mkdir()
    case.write_scene(root, frames)
    ds = VggtProcessedDataset(make_config(root, case.variant, tmp_path, **case.config_overrides))
    return ds, frames, case


class TestAllVariants:
    def test_discovers_single_scene(self, variant_dataset):
        ds, _, _ = variant_dataset
        assert len(ds.scenes) == 1
        assert len(ds) == 1

    def test_sample_schema(self, variant_dataset):
        ds, _, _ = variant_dataset
        assert_sample_schema(ds[0], CLIP_FRAMES, OUT_HW)

    def test_sample_matches_synthetic_geometry(self, variant_dataset):
        ds, frames, case = variant_dataset
        assert_sample_matches_frames(ds[0], frames, OUT_HW, depth_rel_tol=case.depth_rel_tol)

    def test_query_targets_consistent_with_depth(self, variant_dataset):
        ds, _, _ = variant_dataset
        assert_query_targets_consistent(ds[0])

    def test_meta_identifies_variant(self, variant_dataset):
        ds, _, case = variant_dataset
        sample = ds[0]
        assert sample["meta"]["dataset"] == f"{case.variant}_vggt"
        assert sample["meta"]["scene_id"]


class TestTartanairStructure:
    def test_scene_shorter_than_clip_is_dropped(self, tmp_path):
        frames = make_plane_frames(num_frames=CLIP_FRAMES - 1)
        root = tmp_path / "tartanair"
        write_tartanair_scene(root, frames)
        with pytest.raises(ValueError, match="No valid"):
            VggtProcessedDataset(make_config(root, "tartanair", tmp_path))

    def test_frame_with_missing_depth_is_excluded(self, tmp_path):
        frames = make_plane_frames(num_frames=CLIP_FRAMES + 1)
        root = tmp_path / "tartanair"
        scene_dir = write_tartanair_scene(root, frames)
        (scene_dir / f"{CLIP_FRAMES:06d}_depth.npy").unlink()
        ds = VggtProcessedDataset(make_config(root, "tartanair", tmp_path))
        assert len(ds.scenes) == 1
        assert len(ds.scenes[0].frames) == CLIP_FRAMES

    def test_hash_split_partitions_scenes(self, tmp_path):
        frames = make_plane_frames(num_frames=CLIP_FRAMES)
        root = tmp_path / "tartanair"
        for i in range(30):
            write_tartanair_scene(root, frames, env=f"env{i:03d}")

        seen: dict[str, set[str]] = {}
        for split in ("train", "val", "test"):
            ds = VggtProcessedDataset(make_config(root, "tartanair", tmp_path, split=split))
            seen[split] = {s.scene_id for s in ds.scenes}
        assert seen["train"] and seen["val"] and seen["test"]
        assert not (seen["train"] & seen["val"])
        assert not (seen["train"] & seen["test"])
        assert not (seen["val"] & seen["test"])

        ds_all = VggtProcessedDataset(make_config(root, "tartanair", tmp_path, split="all"))
        assert {s.scene_id for s in ds_all.scenes} == seen["train"] | seen["val"] | seen["test"]

    def test_training_mode_oversamples_scenes(self, tmp_path):
        frames = make_plane_frames(num_frames=CLIP_FRAMES)
        root = tmp_path / "tartanair"
        write_tartanair_scene(root, frames)
        ds = VggtProcessedDataset(make_config(root, "tartanair", tmp_path, training=True))
        assert len(ds) > len(ds.scenes)
        sample = ds[0]
        assert sample["video"].shape[0] == CLIP_FRAMES


class TestBlendermvsMultiRoot:
    def test_scenes_collected_from_all_roots(self, tmp_path):
        frames = make_plane_frames(num_frames=CLIP_FRAMES)
        root_a = tmp_path / "base"
        root_b = tmp_path / "plus"
        root_a.mkdir()
        root_b.mkdir()
        write_blendermvs_scene(root_a, frames, scene="0" * 24, cam_format="npz")
        write_blendermvs_scene(root_b, frames, scene="57f8d9bbe73f6760f10e916a", cam_format="safetensor")
        cfg = make_config(None, "blendermvs", tmp_path, roots=(root_a, root_b))
        ds = VggtProcessedDataset(cfg)
        assert len(ds.scenes) == 2
        scene_ids = {s.scene_id for s in ds.scenes}
        assert len(scene_ids) == 2


class TestCo3dMasks:
    def test_masked_region_is_depth_invalid(self, tmp_path):
        frames = make_plane_frames(num_frames=CLIP_FRAMES)
        root = tmp_path / "co3d"
        root.mkdir()
        write_co3d_sequence(root, frames, mask_pattern="left_zero")
        ds = VggtProcessedDataset(make_config(root, "co3d", tmp_path))
        sample = ds[0]
        valid = sample["depth_valid"].numpy()
        w = OUT_HW[1]
        assert not valid[:, :, : w // 2].any(), "masked-out left half must be depth-invalid"
        assert valid[:, :, w // 2 :].mean() > 0.9, "unmasked right half must stay valid"

    def test_masks_can_be_disabled(self, tmp_path):
        frames = make_plane_frames(num_frames=CLIP_FRAMES)
        root = tmp_path / "co3d"
        root.mkdir()
        write_co3d_sequence(root, frames, mask_pattern="left_zero")
        ds = VggtProcessedDataset(make_config(root, "co3d", tmp_path, use_co3d_masks=False))
        sample = ds[0]
        valid = sample["depth_valid"].numpy()
        assert valid.mean() > 0.9


class TestVkittiQuirks:
    def test_sentinel_depth_is_invalid(self, tmp_path):
        frames = make_plane_frames(num_frames=CLIP_FRAMES)
        for fr in frames:
            fr.depth[:8, :8] = 1000.0  # saturates to the raw uint16 sentinel 65535 in cm encoding
        root = tmp_path / "vkitti"
        root.mkdir()
        write_vkitti_scene(root, frames)
        ds = VggtProcessedDataset(make_config(root, "vkitti", tmp_path))
        sample = ds[0]
        valid = sample["depth_valid"].numpy()
        assert not valid[:, :4, :4].any(), "sentinel (65535) depth must be invalid"
        assert valid[:, 8:, 8:].mean() > 0.9

    def test_variant_and_camera_filtering(self, tmp_path):
        frames = make_plane_frames(num_frames=CLIP_FRAMES)
        root = tmp_path / "vkitti"
        root.mkdir()
        write_vkitti_scene(root, frames, scene="Scene01", variant="clone", camera="Camera_0")
        write_vkitti_scene(root, frames, scene="Scene01", variant="fog", camera="Camera_0")
        write_vkitti_scene(root, frames, scene="Scene01", variant="clone", camera="Camera_1")
        ds = VggtProcessedDataset(make_config(root, "vkitti", tmp_path))
        assert len(ds.scenes) == 1
        assert "clone" in ds.scenes[0].scene_id and "Camera_0" in ds.scenes[0].scene_id

        ds_all = VggtProcessedDataset(
            make_config(
                root,
                "vkitti",
                tmp_path,
                vkitti_variants=("clone", "fog"),
                vkitti_cameras=("Camera_0", "Camera_1"),
            )
        )
        assert len(ds_all.scenes) == 3


class TestStaticConsistencyFilter:
    def test_moved_object_pixels_are_invalidated(self, tmp_path):
        frames = make_plane_frames(num_frames=CLIP_FRAMES)
        for t, fr in enumerate(frames):
            if t >= 1:
                fr.depth[16:32, 24:44] *= 0.6  # an "object" jumps closer after frame 0
        root = tmp_path / "vkitti"
        root.mkdir()
        write_vkitti_scene(root, frames)
        ds = VggtProcessedDataset(
            make_config(root, "vkitti", tmp_path, static_consistency_filter=True)
        )
        sample = ds[0]
        valid = sample["depth_valid"].numpy()
        # The depth step between frame 0 and 1 violates static consistency around
        # the object region for both frames adjacent to the jump.
        h_lo, h_hi = int(16 / 48 * OUT_HW[0]) + 1, int(32 / 48 * OUT_HW[0]) - 1
        w_lo, w_hi = int(24 / 64 * OUT_HW[1]) + 1, int(44 / 64 * OUT_HW[1]) - 1
        assert valid[0, h_lo:h_hi, w_lo:w_hi].mean() < 0.2
        assert valid[1, h_lo:h_hi, w_lo:w_hi].mean() < 0.2
        # Far away from the object everything stays valid.
        assert valid[:, : h_lo - 2, : w_lo - 2].mean() > 0.8

    def test_static_scene_unaffected(self, tmp_path):
        frames = make_plane_frames(num_frames=CLIP_FRAMES)
        root = tmp_path / "vkitti"
        root.mkdir()
        write_vkitti_scene(root, frames)
        ds = VggtProcessedDataset(
            make_config(root, "vkitti", tmp_path, static_consistency_filter=True)
        )
        sample = ds[0]
        assert sample["depth_valid"].numpy().mean() > 0.9


class TestCorruptFileResilience:
    def test_corrupt_cam_file_falls_back_to_good_scene(self, tmp_path):
        frames = make_plane_frames(num_frames=CLIP_FRAMES)
        root = tmp_path / "tartanair"
        write_tartanair_scene(root, frames, env="env_bad")
        good_dir = write_tartanair_scene(root, frames, env="env_good")
        bad_dir = root / "env_bad" / "Easy" / "P000"
        for cam_file in bad_dir.glob("*_cam.npz"):
            cam_file.write_bytes(b"not an npz")
        ds = VggtProcessedDataset(make_config(root, "tartanair", tmp_path, training=True))
        for index in range(4):
            sample = ds[index]
            assert sample["meta"]["scene_id"].startswith("env_good")
        assert (tmp_path / "bad_samples.json").exists(), "corrupt files must be recorded in the registry"
        del good_dir


class TestSceneIndexCache:
    def test_cache_file_written_and_reused(self, tmp_path):
        frames = make_plane_frames(num_frames=CLIP_FRAMES)
        root = tmp_path / "tartanair"
        scene_dir = write_tartanair_scene(root, frames)
        cache_dir = tmp_path / "scene_cache"

        ds1 = VggtProcessedDataset(
            make_config(root, "tartanair", tmp_path, scene_index_cache=cache_dir)
        )
        assert len(ds1.scenes) == 1
        cache_files = list(cache_dir.glob("tartanair_*.json"))
        assert len(cache_files) == 1, "first construction must persist the scene index"

        # Hide the on-disk scenes: a second construction must still discover them
        # from the cache (proving the cache is read instead of rescanning).
        scene_dir.rename(scene_dir.with_name("P000_hidden"))
        ds2 = VggtProcessedDataset(
            make_config(root, "tartanair", tmp_path, scene_index_cache=cache_dir)
        )
        assert {s.scene_id for s in ds2.scenes} == {s.scene_id for s in ds1.scenes}
        assert [f.frame_id for f in ds2.scenes[0].frames] == [f.frame_id for f in ds1.scenes[0].frames]

    def test_cache_key_distinguishes_roots(self, tmp_path):
        frames = make_plane_frames(num_frames=CLIP_FRAMES)
        root_a = tmp_path / "ta_a"
        root_b = tmp_path / "ta_b"
        write_tartanair_scene(root_a, frames, env="env_a")
        write_tartanair_scene(root_b, frames, env="env_b")
        cache_dir = tmp_path / "scene_cache"
        ds_a = VggtProcessedDataset(make_config(root_a, "tartanair", tmp_path, scene_index_cache=cache_dir))
        ds_b = VggtProcessedDataset(make_config(root_b, "tartanair", tmp_path, scene_index_cache=cache_dir))
        assert {s.scene_id for s in ds_a.scenes} == {"env_a/Easy/P000"}
        assert {s.scene_id for s in ds_b.scenes} == {"env_b/Easy/P000"}
        assert len(list(cache_dir.glob("tartanair_*.json"))) == 2

    def test_no_cache_dir_means_no_cache_io(self, tmp_path):
        frames = make_plane_frames(num_frames=CLIP_FRAMES)
        root = tmp_path / "tartanair"
        write_tartanair_scene(root, frames)
        ds = VggtProcessedDataset(make_config(root, "tartanair", tmp_path))
        assert len(ds.scenes) == 1
        assert not list(tmp_path.glob("**/tartanair_*.json"))


class TestUnknownVariant:
    def test_unknown_variant_raises(self, tmp_path):
        with pytest.raises(ValueError, match="variant"):
            VggtProcessedDataset(make_config(tmp_path, "nonexistent_dataset", tmp_path))
