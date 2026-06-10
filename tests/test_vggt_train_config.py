"""Validate the shipped VGGT-processed training config resolves to the right builders."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.core.config import ConfigNode
from src.data.builder import _resolve_mixture_sources

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "train_vggt_sixmix.yaml"

EXPECTED_SOURCES = {
    "tartanair_vggt",
    "mvs_synth_vggt",
    "scannet_vggt",
    "blendermvs_vggt",
    "co3d_vggt",
    "vkitti_vggt",
}


def load_cfg() -> ConfigNode:
    return ConfigNode(yaml.safe_load(CONFIG_PATH.read_text()))


def test_config_exists():
    assert CONFIG_PATH.exists()


def test_train_mixture_resolves_to_vggt_builders():
    cfg = load_cfg()
    resolved = dict(_resolve_mixture_sources("train", cfg))
    assert set(resolved) == EXPECTED_SOURCES
    for source, builder in resolved.items():
        assert builder == source, f"{source} should map to its own vggt builder"


def test_val_mixture_resolves():
    cfg = load_cfg()
    resolved = dict(_resolve_mixture_sources("val", cfg))
    assert set(resolved) == EXPECTED_SOURCES


def test_every_source_has_root_and_weight():
    cfg = load_cfg()
    weights = cfg.get_path("data.mixture_sampling_weights")
    for source in EXPECTED_SOURCES:
        assert float(weights[source]) > 0
        block = cfg.get_path(f"data.{source}")
        assert block is not None, f"missing data.{source} block"
        root = block.get("root") or block.get("roots")
        assert root, f"data.{source} must define root(s)"


def test_blendermvs_has_both_processed_roots():
    cfg = load_cfg()
    roots = cfg.get_path("data.blendermvs_vggt.roots")
    assert isinstance(roots, list) and len(roots) == 2


def test_every_source_sets_scene_index_cache():
    cfg = load_cfg()
    for source in EXPECTED_SOURCES:
        cache = cfg.get_path(f"data.{source}.scene_index_cache")
        assert cache, f"data.{source} must set scene_index_cache (full /jfs discovery takes minutes)"
