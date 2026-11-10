"""M1: audio analysis on synthetic 120 BPM click track."""

from __future__ import annotations

from pathlib import Path

import pytest

from helpers import write_click_track_wav
from music_video.audio_analyze import analyze_audio_file, write_audio_analysis
from music_video.errors import AudioAnalysisError


def test_click_track_120bpm(tmp_path: Path, click_wav: Path):
    data = analyze_audio_file(click_wav, work_dir=tmp_path / "work")
    assert data["schema"] == "mv.audio.v1"
    assert data["sr"] == 22050
    assert data["channels"] == 1
    assert data["duration_s"] >= 11.5
    assert "beat_confidence" in data
    assert set(data["beat_confidence"]) >= {"interval_regularity", "onset_agreement", "final"}
    assert "grid_used" in data
    assert data["sections"]
    assert data["cut_candidates_s"]

    bpm = 120.0
    period = 60.0 / bpm
    expected_n = int(data["duration_s"] / period)
    beats = data["beats_s"]
    assert abs(len(beats) - expected_n) <= max(2, 0.10 * expected_n)

    for b in beats:
        nearest = round(b / period) * period
        assert abs(b - nearest) <= 0.10, f"beat {b} far from grid {nearest}"


def test_write_audio_analysis_json(tmp_path: Path, click_wav: Path):
    out = tmp_path / "audio_analysis.json"
    data = write_audio_analysis(click_wav, out, work_dir=tmp_path / "work")
    assert out.is_file()
    assert data["tempo_bpm"] > 0


def test_silent_audio_hard_error(tmp_path: Path, silent_wav: Path):
    with pytest.raises(AudioAnalysisError, match="silent"):
        analyze_audio_file(silent_wav, work_dir=tmp_path / "work")


def test_too_short_hard_error(tmp_path: Path):
    short = write_click_track_wav(tmp_path / "short.wav", bpm=120.0, duration_s=3.0)
    with pytest.raises(AudioAnalysisError, match="too short"):
        analyze_audio_file(short, work_dir=tmp_path / "work")


def test_missing_file(tmp_path: Path):
    with pytest.raises(AudioAnalysisError, match="not found"):
        analyze_audio_file(tmp_path / "nope.wav", work_dir=tmp_path / "work")
