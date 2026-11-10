"""M6: run orchestration — offline mocked planner, resume, CLIP fail, spaced paths."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from helpers import write_click_track_wav, write_solid_png
from music_video.image_analyze import clip_weights_ready
from music_video.pipeline import (
    PipelineError,
    RunConfig,
    needed_stages,
    run_log_path,
    run_pipeline,
    stage_analyze_images,
)
from music_video.paths import ensure_job, section_plan_path, validated_plan_path
from music_video.render_ffmpeg import ffmpeg_available, ffprobe_available, verify_final


def _require_ffmpeg():
    if not ffmpeg_available() or not ffprobe_available():
        pytest.skip("ffmpeg/ffprobe not on PATH")


def _require_clip():
    if not clip_weights_ready():
        pytest.skip("CLIP weights cache missing under models/open-clip — run: python scripts/bootstrap_clip.py")


def _make_inputs(root: Path) -> tuple[Path, Path]:
    song = write_click_track_wav(root / "src song.wav", bpm=120.0, duration_s=12.0)
    imgs = root / "src images"
    imgs.mkdir(parents=True, exist_ok=True)
    for i, rgb in enumerate([(200, 30, 30), (30, 180, 50), (30, 60, 200), (240, 200, 40), (20, 20, 20)], 1):
        write_solid_png(imgs / f"{i:02d}.png", rgb=rgb, size=(96, 144))
    if not song.is_file() or len(list(imgs.glob("*.png"))) < 5:
        raise RuntimeError("fixture generation failed")
    return song, imgs


def test_offline_full_run_e2e(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _require_ffmpeg()
    _require_clip()
    monkeypatch.setenv("MV_MOCK_PLANNER", "1")
    # isolate jobs root via symlink? paths.JOBS is package-fixed — use real jobs id under package
    # Instead call run_pipeline with job under music-video/jobs via ensure_job unique name
    song, imgs = _make_inputs(tmp_path)
    # copy into a unique job by using pipeline prepare which writes under paths.JOBS
    job_id = f"e2e_{tmp_path.name}"
    # originals stay in tmp_path; pipeline copies
    orig_song_bytes = song.read_bytes()
    result = run_pipeline(RunConfig(
        job_id=job_id,
        style="test cinematic",
        aspect="9:16",
        out="final",
        song=song,
        images_dir=imgs,
    ))
    assert result["ok"]
    out = Path(result["out"])
    assert out.is_file() and out.stat().st_size > 100_000
    verify_final(out, expect_w=1080, expect_h=1920, expect_dur=12.0)
    # originals untouched
    assert song.read_bytes() == orig_song_bytes
    assert song.is_file()
    job = Path(result["job"])
    log = run_log_path(job).read_text(encoding="utf-8")
    for stage in ("analyze-audio", "analyze-images", "plan", "expand", "validate", "render"):
        assert f"STAGE START {stage}" in log
        assert f"STAGE END {stage}" in log
        assert "wall_s=" in log


def test_resume_mid_chain_and_skip_rerender(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _require_ffmpeg()
    _require_clip()
    monkeypatch.setenv("MV_MOCK_PLANNER", "1")
    song, imgs = _make_inputs(tmp_path)
    job_id = f"resume_{tmp_path.name}"
    run_pipeline(RunConfig(job_id=job_id, style="t", aspect="9:16", out="final", song=song, images_dir=imgs))
    job = ensure_job(job_id)
    # delete mid-chain artifacts
    section_plan_path(job).unlink()
    validated_plan_path(job).unlink(missing_ok=True)
    (job / "plan.json").unlink(missing_ok=True)
    (job / "final.mp4").unlink(missing_ok=True)
    (job / "final.planhash").unlink(missing_ok=True)
    stages = needed_stages(job, "final")
    assert "plan" in stages
    assert "analyze-audio" not in stages
    assert "analyze-images" not in stages
    result = run_pipeline(RunConfig(job_id=job_id, out="final"))
    assert "plan" in result["stages"]
    assert "analyze-audio" not in result["stages"]
    log = run_log_path(job).read_text(encoding="utf-8")
    assert "mode=B-resume" in log

    # re-run unchanged → skip render
    result2 = run_pipeline(RunConfig(job_id=job_id, out="final"))
    assert result2.get("skipped_all") or "render" not in result2["stages"]
    log2 = run_log_path(job).read_text(encoding="utf-8")
    assert "SKIP" in log2 or result2.get("skipped_all")


def test_missing_clip_cache_clear_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    job = ensure_job(f"noclip_{tmp_path.name}")
    (job / "images").mkdir(parents=True, exist_ok=True)
    write_solid_png(job / "images" / "01.png", rgb=(1, 2, 3))
    monkeypatch.setattr("music_video.pipeline.clip_weights_ready", lambda cache_dir=None: False)
    with pytest.raises(PipelineError, match="bootstrap_clip.py"):
        stage_analyze_images(job, "style")


def test_missing_inputs_clear_error(tmp_path: Path):
    with pytest.raises(PipelineError, match="song not found"):
        run_pipeline(RunConfig(
            job_id=f"bad_{tmp_path.name}",
            song=tmp_path / "nope.wav",
            images_dir=tmp_path / "imgs",
        ))


def test_spaced_paths_full_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _require_ffmpeg()
    _require_clip()
    monkeypatch.setenv("MV_MOCK_PLANNER", "1")
    spaced = tmp_path / "path with spaces"
    song, imgs = _make_inputs(spaced)
    result = run_pipeline(RunConfig(
        job_id=f"space_{tmp_path.name}",
        style="t",
        aspect="9:16",
        out="final",
        song=song,
        images_dir=imgs,
    ))
    assert Path(result["out"]).is_file()


@pytest.mark.skipif(
    os.environ.get("MV_E2E") != "1" or not os.environ.get("OPENROUTER_API_KEY"),
    reason="live OpenRouter gated: need MV_E2E=1 AND OPENROUTER_API_KEY",
)
def test_live_full_run_gated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _require_ffmpeg()
    _require_clip()
    monkeypatch.delenv("MV_MOCK_PLANNER", raising=False)
    song, imgs = _make_inputs(tmp_path)
    result = run_pipeline(RunConfig(
        job_id=f"live_{tmp_path.name}",
        style="short cinematic test",
        aspect="9:16",
        out="final",
        song=song,
        images_dir=imgs,
    ))
    assert Path(result["out"]).is_file()
