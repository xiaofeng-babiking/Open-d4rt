"""Tests for VGGT-processed adapter registration in the dataset builder."""

from __future__ import annotations

from pathlib import Path

import pytest

from helpers_synthetic import make_plane_frames, write_mvs_synth_scene, write_tartanair_scene, write_vkitti_scene

from src.core.config import ConfigNode
from src.data.builder import DATASET_REGISTRY, build_dataset
from src.data.mixture_dataset import MixtureDataset
from src.data.vggt_processed_dataset import KNOWN_VARIANTS, VggtProcessedDataset

CLIP_FRAMES = 4


def base_config(tmp_path: Path, **data_overrides) -> ConfigNode:
    data = {
        "clip_frames": CLIP_FRAMES,
        "image_size": [32, 32],
        "bad_sample_registry": {"path": str(tmp_path / "bad_samples.json")},
    }
    data.update(data_overrides)
    return ConfigNode(
        {
            "data": data,
            "train_sampling": {
                "queries_per_clip": 256,
                "hard_query_ratio": 0.2,
                "timestep_sampling": {"prob_t_tgt_equals_t_cam": 0.4},
            },
        }
    )


class TestRegistry:
    def test_all_variants_registered(self):
        for variant in KNOWN_VARIANTS:
            assert DATASET_REGISTRY.get(f"{variant}_vggt") is not None


class TestBuildFromConfig:
    def test_build_tartanair_vggt(self, tmp_path):
        root = tmp_path / "tartanair"
        write_tartanair_scene(root, make_plane_frames(num_frames=CLIP_FRAMES))
        cfg = base_config(tmp_path, tartanair_vggt={"root": str(root), "split_override": "all"})
        builder = DATASET_REGISTRY.get("tartanair_vggt")
        ds = builder(split="train", cfg=cfg, manifest_paths=None)
        assert isinstance(ds, VggtProcessedDataset)
        assert ds.cfg.training is True
        assert ds.cfg.clip_frames == CLIP_FRAMES
        assert ds.cfg.queries_per_clip == 256
        sample = ds[0]
        assert sample["meta"]["dataset"] == "tartanair_vggt"

    def test_vkitti_defaults_enable_static_filter(self, tmp_path):
        root = tmp_path / "vkitti"
        write_vkitti_scene(root, make_plane_frames(num_frames=CLIP_FRAMES))
        cfg = base_config(tmp_path, vkitti_vggt={"root": str(root), "split_override": "all"})
        ds = DATASET_REGISTRY.get("vkitti_vggt")(split="train", cfg=cfg, manifest_paths=None)
        assert ds.cfg.static_consistency_filter is True
        assert ds.cfg.vkitti_variants == ("clone",)
        assert ds.cfg.vkitti_cameras == ("Camera_0",)

    def test_non_vkitti_defaults_disable_static_filter(self, tmp_path):
        root = tmp_path / "tartanair"
        write_tartanair_scene(root, make_plane_frames(num_frames=CLIP_FRAMES))
        cfg = base_config(tmp_path, tartanair_vggt={"root": str(root), "split_override": "all"})
        ds = DATASET_REGISTRY.get("tartanair_vggt")(split="train", cfg=cfg, manifest_paths=None)
        assert ds.cfg.static_consistency_filter is False

    def test_missing_root_raises(self, tmp_path):
        cfg = base_config(tmp_path, tartanair_vggt={"root": str(tmp_path / "nope")})
        with pytest.raises(FileNotFoundError):
            DATASET_REGISTRY.get("tartanair_vggt")(split="train", cfg=cfg, manifest_paths=None)

    def test_scene_index_cache_passthrough(self, tmp_path):
        root = tmp_path / "tartanair"
        write_tartanair_scene(root, make_plane_frames(num_frames=CLIP_FRAMES))
        cache_dir = tmp_path / "scene_cache"
        cfg = base_config(
            tmp_path,
            tartanair_vggt={
                "root": str(root),
                "split_override": "all",
                "scene_index_cache": str(cache_dir),
            },
        )
        ds = DATASET_REGISTRY.get("tartanair_vggt")(split="train", cfg=cfg, manifest_paths=None)
        assert ds.cfg.scene_index_cache == cache_dir
        assert list(cache_dir.glob("tartanair_*.json"))

    def test_scene_index_cache_default_off(self, tmp_path):
        root = tmp_path / "tartanair"
        write_tartanair_scene(root, make_plane_frames(num_frames=CLIP_FRAMES))
        cfg = base_config(tmp_path, tartanair_vggt={"root": str(root), "split_override": "all"})
        ds = DATASET_REGISTRY.get("tartanair_vggt")(split="train", cfg=cfg, manifest_paths=None)
        assert ds.cfg.scene_index_cache is None


class TestMixtureIntegration:
    def test_mixture_builds_with_vggt_sources(self, tmp_path):
        frames = make_plane_frames(num_frames=CLIP_FRAMES)
        tartanair_root = tmp_path / "tartanair"
        mvs_root = tmp_path / "mvs_synth"
        write_tartanair_scene(tartanair_root, frames)
        mvs_root.mkdir()
        write_mvs_synth_scene(mvs_root, frames)
        cfg = base_config(
            tmp_path,
            train_dataset_type="mixture_raw",
            train_dataset_mixture=["tartanair_vggt", "mvs_synth_vggt"],
            mixture_sampling_weights={"tartanair_vggt": 1.0, "mvs_synth_vggt": 2.0},
            tartanair_vggt={"root": str(tartanair_root), "split_override": "all"},
            mvs_synth_vggt={"root": str(mvs_root), "split_override": "all"},
        )
        ds = build_dataset(split="train", cfg=cfg)
        assert isinstance(ds, MixtureDataset)
        assert len(ds.datasets) == 2
        sample = ds[0]
        assert sample["meta"]["dataset"] in {"tartanair_vggt", "mvs_synth_vggt"}
