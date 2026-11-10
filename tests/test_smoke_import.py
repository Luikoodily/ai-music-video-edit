"""M0: package importable."""

from __future__ import annotations


def test_import_music_video():
    import music_video

    assert music_video.__version__


def test_core_modules_import():
    from music_video import audio_analyze, errors, models, paths, render_ffmpeg, validate_plan

    assert errors.HardValidationError
    assert models.MusicVideoPlan
    assert paths.JOBS
    assert audio_analyze.TARGET_SR == 22050
    assert validate_plan.DEFAULT_MIN_CLIP
    assert render_ffmpeg.ffmpeg_available
