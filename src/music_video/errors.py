"""User-facing and internal errors for the music-video pipeline."""

from __future__ import annotations


class MusicVideoError(Exception):
    """Base error."""


class HardValidationError(MusicVideoError):
    """Plan HARD failure — must not silently repair."""


class SoftValidationNote(MusicVideoError):
    """Not raised for control flow; notes are recorded on the plan."""


class AudioAnalysisError(MusicVideoError):
    """Invalid / undecodable / too-short / silent audio."""


class RenderError(MusicVideoError):
    """FFmpeg or post-render verification failure."""
