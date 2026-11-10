"""Deterministic timeline expander tests (no network, no LLM)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from helpers import write_click_track_wav, write_solid_png
from music_video.errors import TimelineExpandError
from music_video.render_ffmpeg import ffmpeg_available, ffprobe_available, render_job, verify_final
from music_video.timeline_expand import cut_points_for_density, expand_section_plan, snap_section_edge


def _audio(duration: float = 12.0, bpm: float = 120.0) -> dict:
    period = 60.0 / bpm
    beats = [round(i * period, 6) for i in range(int(duration / period) + 1) if i * period <= duration]
    return {
        "duration_s": duration,
        "tempo_bpm": bpm,
        "beats_s": beats,
        "downbeats_s": beats[::4],
        "grid_used": False,
        "beat_confidence": {"final": 0.9},
        "sections": [
            {"id": "s0", "label": "intro", "start_s": 0.0, "end_s": 4.0},
            {"id": "s1", "label": "drop", "start_s": 4.0, "end_s": 8.0},
            {"id": "s2", "label": "outro", "start_s": 8.0, "end_s": 12.0},
        ],
    }


def _section_plan() -> dict:
    return {
        "schema": "mv.sectionplan.v1",
        "aspect": "9:16",
        "sections": [
            {"section_id": "s0", "image_ids": ["img_01", "img_02"], "density": "sparse",
             "title": {"text": "INTRO", "style": {"color": "#FFFFFF", "bg": "#000000", "bg_pad": 12}}},
            {"section_id": "s1", "image_ids": ["img_02", "img_03", "img_01"], "density": "dense"},
            {"section_id": "s2", "image_ids": ["img_03", "img_01"], "density": "medium"},
        ],
        "notes": ["fixture"],
    }


def _images_meta() -> list[dict]:
    return [
        {"id": "img_01", "best_sections": ["s1", "s0"], "labels": ["hero"]},
        {"id": "img_02", "best_sections": ["s0"], "labels": ["detail"]},
        {"id": "img_03", "best_sections": ["s2", "s1"], "labels": ["graphic"]},
    ]


def _norm_plan(plan: dict) -> dict:
    """Strip metadata/notes noise for determinism compare."""
    clips = [
        {
            "image_id": c["image_id"],
            "start_s": round(float(c["start_s"]), 4),
            "end_s": round(float(c["end_s"]), 4),
            "fit": c.get("fit"),
        }
        for c in plan["clips"]
    ]
    titles = [
        {
            "text": t["text"],
            "start_s": round(float(t["start_s"]), 4),
            "end_s": round(float(t["end_s"]), 4),
        }
        for t in plan.get("titles") or []
    ]
    return {"clips": clips, "titles": titles, "duration_s": plan["duration_s"]}


def test_snap_section_edge_tol():
    beats = [0.0, 0.5, 1.0, 1.5]
    assert snap_section_edge(1.05, beats) == 1.0
    assert snap_section_edge(1.40, beats) == 1.5  # |1.4-1.5|=0.1 ≤ 0.15
    assert snap_section_edge(1.30, beats) == 1.30  # nearest 1.5 Δ0.2 > 0.15


def test_density_sparse_medium_dense_cut_counts():
    beats = [i * 0.5 for i in range(0, 21)]
    downs = beats[::4]
    sparse = cut_points_for_density(density="sparse", start=0.0, end=10.0, beats=beats, downbeats=downs)
    medium = cut_points_for_density(density="medium", start=0.0, end=10.0, beats=beats, downbeats=downs)
    dense = cut_points_for_density(density="dense", start=0.0, end=10.0, beats=beats, downbeats=downs)
    assert len(sparse) < len(medium) <= len(dense)


def test_expand_m3_compatible_and_final_frame():
    plan = expand_section_plan(_section_plan(), _audio(), _images_meta(), audio_path="song.wav", fps=30)
    assert plan["schema"] == "mv.plan.v1"
    assert abs(plan["clips"][-1]["end_s"] - 12.0) <= 1.0 / 30.0
    assert plan["clips"][0]["start_s"] == 0.0
    for a, b in zip(plan["clips"], plan["clips"][1:]):
        assert b["start_s"] >= a["end_s"] - 1e-6


def test_determinism_double_run_r7():
    a = expand_section_plan(_section_plan(), _audio(), _images_meta())
    b = expand_section_plan(copy.deepcopy(_section_plan()), copy.deepcopy(_audio()), copy.deepcopy(_images_meta()))
    assert _norm_plan(a) == _norm_plan(b)


def test_missing_beats_hard_fail():
    audio = _audio()
    audio["beats_s"] = []
    with pytest.raises(TimelineExpandError, match="beats_s"):
        expand_section_plan(_section_plan(), audio, _images_meta())


def test_empty_section_images_fail():
    sp = _section_plan()
    sp["sections"][0]["image_ids"] = []
    with pytest.raises(TimelineExpandError):
        expand_section_plan(sp, _audio(), _images_meta())


def test_m1_synthetic_grid_written_into_beats(tmp_path: Path):
    """C-1 audit: when fallback fires, beats_s must hold the synthetic grid."""
    from music_video.audio_analyze import analyze_audio_file

    wav = write_click_track_wav(tmp_path / "c.wav", bpm=120.0, duration_s=12.0)
    data = analyze_audio_file(wav, work_dir=tmp_path / "w")
    assert data["beats_s"], "beats_s must be authoritative grid"
    assert "grid_used" in data
    assert "fallback_reason" in data or data["grid_used"] is False
    if data["grid_used"]:
        # synthetic was written into beats_s (not a parallel field)
        assert len(data["beats_s"]) >= 4


def test_g1_mocked_section_plan_to_final_mp4(tmp_path: Path):
    if not ffmpeg_available() or not ffprobe_available():
        pytest.skip("ffmpeg/ffprobe not on PATH")

    job = tmp_path / "g1"
    images = job / "images"
    images.mkdir(parents=True)
    write_solid_png(images / "01.png", rgb=(200, 30, 30), size=(160, 240))
    write_solid_png(images / "02.png", rgb=(30, 180, 50), size=(160, 240))
    write_solid_png(images / "03.png", rgb=(30, 60, 200), size=(160, 240))
    song = write_click_track_wav(job / "song.wav", bpm=120.0, duration_s=12.0)

    audio = _audio(12.0)
    audio["path"] = str(song)
    (job / "audio_analysis.json").write_text(json.dumps(audio), encoding="utf-8")
    (job / "image_analysis.json").write_text(json.dumps({
        "schema": "mv.images.v1",
        "images": _images_meta(),
        "duplicates": [],
    }), encoding="utf-8")
    (job / "section_plan.json").write_text(json.dumps(_section_plan()), encoding="utf-8")

    from music_video.timeline_expand import expand_job_files

    validated = expand_job_files(job)
    assert validated["clips"]
    # density signal: dense section should contribute more clips than sparse intro
    # (after M3 clamps). Count by time windows.
    intro = [c for c in validated["clips"] if c["end_s"] <= 4.0 + 1e-6]
    drop = [c for c in validated["clips"] if c["start_s"] >= 4.0 - 1e-6 and c["end_s"] <= 8.0 + 1e-6]
    assert len(drop) >= len(intro)

    out = render_job(job)
    verify_final(out, expect_w=1080, expect_h=1920, expect_dur=12.0)
