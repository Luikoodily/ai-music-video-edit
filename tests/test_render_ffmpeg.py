"""M5: FFmpeg render — skip cleanly if ffmpeg absent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from helpers import write_click_track_wav, write_solid_png
from music_video.render_ffmpeg import ffmpeg_available, ffprobe_available, render_job, verify_final
from music_video.validate_plan import validate_plan


def _require_ffmpeg():
    if not ffmpeg_available() or not ffprobe_available():
        pytest.skip("ffmpeg/ffprobe not on PATH")


def _make_job(tmp_path: Path, *, job_name: str = "render1", duration_s: float = 12.0) -> Path:
    job = tmp_path / job_name
    images = job / "images"
    images.mkdir(parents=True)
    write_solid_png(images / "01.png", rgb=(200, 30, 30), size=(128, 192))
    write_solid_png(images / "02.png", rgb=(30, 180, 50), size=(128, 192))
    write_solid_png(images / "03.png", rgb=(30, 60, 200), size=(128, 192))
    song = write_click_track_wav(job / "song.wav", bpm=120.0, duration_s=duration_s)
    if not song.is_file():
        raise RuntimeError("song fixture missing after generate")

    plan = {
        "schema": "mv.plan.v1",
        "aspect": "9:16",
        "width": 1080,
        "height": 1920,
        "fps": 30,
        "duration_s": duration_s,
        "audio_path": str(song),
        "clips": [
            {"id": "c001", "image_id": "img_01", "start_s": 0.0, "end_s": 4.0,
             "transition_in": "cut", "fx": "none", "fx_strength": 0.0, "fit": "cover"},
            {"id": "c002", "image_id": "img_02", "start_s": 4.0, "end_s": 8.0,
             "transition_in": "cut", "fx": "none", "fx_strength": 0.0, "fit": "cover"},
            {"id": "c003", "image_id": "img_03", "start_s": 8.0, "end_s": duration_s,
             "transition_in": "cut", "fx": "none", "fx_strength": 0.0, "fit": "cover"},
        ],
        "titles": [
            {"id": "t001", "text": "TEST", "start_s": 0.5, "end_s": 2.5, "position": "center",
             "style": {"color": "#FFFFFF", "bg": "#000000", "bg_pad": 12}},
        ],
        "notes": [],
    }
    validated = validate_plan(
        plan,
        audio_duration_s=duration_s,
        known_image_ids={"img_01", "img_02", "img_03"},
    )
    (job / "plan.validated.json").write_text(json.dumps(validated, indent=2), encoding="utf-8")
    return job


def test_render_produces_final(tmp_path: Path):
    _require_ffmpeg()
    job = _make_job(tmp_path, duration_s=12.0)
    out = render_job(job)
    assert out.is_file()
    assert out.stat().st_size > 100_000
    verify_final(out, expect_w=1080, expect_h=1920, expect_dur=12.0)


def test_render_windows_spaced_path(tmp_path: Path):
    _require_ffmpeg()
    spaced = tmp_path / "job with spaces"
    job = _make_job(spaced, job_name="inner", duration_s=12.0)
    # move to spaced parent already is spaced root
    out = render_job(job)
    assert out.is_file()
    verify_final(out, expect_w=1080, expect_h=1920, expect_dur=12.0)


def test_ass_escape_braces():
    from music_video.render_ffmpeg import ass_escape

    assert "\\{" in ass_escape("a{b}c")
    assert "\\N" in ass_escape("a\nb")
