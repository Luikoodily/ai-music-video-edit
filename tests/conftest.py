"""Shared fixtures: synthetic audio/images generated at test time."""

from __future__ import annotations

from pathlib import Path

import pytest

from helpers import write_click_track_wav, write_silent_wav, write_solid_png


@pytest.fixture
def click_wav(tmp_path: Path) -> Path:
    return write_click_track_wav(tmp_path / "click_120.wav", bpm=120.0, duration_s=12.0)


@pytest.fixture
def silent_wav(tmp_path: Path) -> Path:
    return write_silent_wav(tmp_path / "silent.wav", duration_s=12.0)


@pytest.fixture
def three_images(tmp_path: Path) -> Path:
    d = tmp_path / "images"
    write_solid_png(d / "01.png", rgb=(220, 40, 40))
    write_solid_png(d / "02.png", rgb=(40, 180, 60))
    write_solid_png(d / "03.png", rgb=(40, 80, 220))
    return d
