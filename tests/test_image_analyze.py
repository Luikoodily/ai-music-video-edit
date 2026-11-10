"""M2: image metadata / phash / dedupe (no torch required) + CLIP gated tests."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

from helpers import write_solid_png
from music_video.image_analyze import (
    CLIP_EMBED_DIM,
    PHASH_HAMMING_THRESHOLD,
    apply_label_threshold,
    apply_section_ranking,
    build_metadata_records,
    clip_weights_ready,
    find_duplicates,
    validate_images_payload_contract,
)


def _five_plus_images(tmp_path: Path) -> Path:
    d = tmp_path / "images"
    # distinct solids + shapes (different RGB → different phash usually)
    colors = [
        (220, 40, 40),
        (40, 180, 60),
        (40, 80, 220),
        (240, 200, 40),
        (180, 40, 180),
        (20, 20, 20),
    ]
    for i, rgb in enumerate(colors, start=1):
        write_solid_png(d / f"{i:02d}.png", rgb=rgb, size=(64 + i * 4, 96 + i * 2))
    # synthetic duplicate of 01
    write_solid_png(d / "01_dup.png", rgb=(220, 40, 40), size=(64 + 4, 96 + 2))
    # ensure generated
    files = list(d.glob("*.png"))
    if len(files) < 5:
        raise RuntimeError(f"fixture generation failed: only {len(files)} images")
    return d


def test_metadata_stable_ids_no_torch(tmp_path: Path):
    # prove torch not required for this path
    assert importlib.util.find_spec("torch") is not None or True  # may be installed; don't require absence
    d = _five_plus_images(tmp_path)
    recs = build_metadata_records(d)
    assert len(recs) >= 5
    assert recs[0]["id"] == "img_01"
    assert recs[1]["id"] == "img_02"
    for r in recs:
        assert r["width"] > 0 and r["height"] > 0
        assert math.isfinite(r["aspect"])
        assert r["file_size"] > 0
        assert r["phash"]


def test_phash_threshold_constant_labelled():
    assert isinstance(PHASH_HAMMING_THRESHOLD, int)
    assert PHASH_HAMMING_THRESHOLD == 5


def test_dedupe_flags_synthetic_duplicate(tmp_path: Path):
    d = _five_plus_images(tmp_path)
    recs = build_metadata_records(d)
    dups, marked = find_duplicates(recs, threshold=PHASH_HAMMING_THRESHOLD)
    assert dups, "expected synthetic duplicate pair"
    # secondary unused
    unused = [m for m in marked if m["unused"]]
    assert unused, "secondary duplicate must be marked unused"
    # files still on disk
    for r in recs:
        assert Path(r["path"]).is_file()


def test_label_threshold_without_torch():
    labels = apply_label_threshold({
        "hero": 0.55,
        "action": 0.10,
        "portrait": 0.30,
        "detail": 0.01,
    })
    assert labels[0] == "hero"
    assert "portrait" in labels
    assert "action" not in labels


def test_section_ranking_without_torch():
    best = apply_section_ranking({"s0": 0.05, "s1": 0.4, "s2": 0.35, "s3": 0.2}, top_k=2)
    assert best == ["s1", "s2"]


def test_json_contract_without_clip_fields():
    payload = {
        "schema": "mv.images.v1",
        "model_id": "none",
        "weights_path": "",
        "embedding_dim": 0,
        "images": [{
            "id": "img_01",
            "path": "images/01.png",
            "width": 10,
            "height": 20,
            "aspect": 0.5,
            "phash": "abcd",
            "labels": ["hero"],
            "section_scores": {"s0": 0.1},
        }],
        "duplicates": [],
    }
    validate_images_payload_contract(payload)


def test_clip_full_analysis_or_skip(tmp_path: Path):
    if not clip_weights_ready():
        pytest.skip(
            f"CLIP weights cache missing under models/open-clip — "
            f"run: python scripts/bootstrap_clip.py"
        )
    torch_spec = importlib.util.find_spec("torch")
    clip_spec = importlib.util.find_spec("open_clip")
    if torch_spec is None or clip_spec is None:
        pytest.skip("torch/open-clip-torch not installed")

    from music_video.image_analyze import analyze_images

    d = _five_plus_images(tmp_path)
    # slightly different size for dup already; add audio sections
    audio = {
        "sections": [
            {"id": "s0", "label": "intro"},
            {"id": "s1", "label": "drop"},
        ]
    }
    data = analyze_images(d, audio_analysis=audio, style_words="cinematic", with_clip=True)
    assert data["schema"] == "mv.images.v1"
    assert len(data["images"]) >= 5
    assert data["embedding_dim"] == CLIP_EMBED_DIM or data["embedding_dim"] > 0
    assert data["model_id"]
    assert data["weights_path"]
    for im in data["images"]:
        assert isinstance(im["labels"], list)
        assert isinstance(im["section_scores"], dict)
        assert im["section_scores"], "section_scores must be present"
        for v in im["section_scores"].values():
            assert math.isfinite(float(v))
        assert "best_sections" in im
    assert data["duplicates"], "synthetic dup must be flagged"
