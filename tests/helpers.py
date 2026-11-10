"""Test-time fixture generators (missing generated file = ERROR, never silent pass)."""

from __future__ import annotations

import struct
import wave
import zlib
from pathlib import Path

import numpy as np


def write_click_track_wav(path: Path, *, bpm: float = 120.0, duration_s: float = 12.0, sr: int = 22050) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(duration_s * sr)
    y = np.zeros(n, dtype=np.float32)
    period = 60.0 / bpm
    click_len = int(0.01 * sr)
    t = 0.0
    while t < duration_s:
        i0 = int(round(t * sr))
        i1 = min(n, i0 + click_len)
        if i0 < n:
            env = np.linspace(1.0, 0.0, i1 - i0, dtype=np.float32)
            y[i0:i1] += 0.9 * env
        t += period
    y += 0.002 * np.sin(2 * np.pi * 220.0 * np.arange(n) / sr).astype(np.float32)
    pcm = np.clip(y, -1.0, 1.0)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((pcm * 32767.0).astype(np.int16).tobytes())
    if not path.is_file() or path.stat().st_size < 100:
        raise RuntimeError(f"failed to generate fixture wav: {path}")
    return path


def write_silent_wav(path: Path, *, duration_s: float = 12.0, sr: int = 22050) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(duration_s * sr)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(np.zeros(n, dtype=np.int16).tobytes())
    if not path.is_file():
        raise RuntimeError(f"failed to generate silent wav: {path}")
    return path


def write_solid_png(path: Path, *, rgb: tuple[int, int, int] = (255, 0, 0), size: tuple[int, int] = (64, 96)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    r, g, b = rgb
    raw = b"".join(b"\x00" + bytes([r, g, b]) * w for _ in range(h))
    compressed = zlib.compress(raw, 9)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", compressed)
    png += chunk(b"IEND", b"")
    path.write_bytes(png)
    if not path.is_file():
        raise RuntimeError(f"failed to generate PNG: {path}")
    return path
