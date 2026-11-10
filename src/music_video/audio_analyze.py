"""Local audio analysis: BPM, beats, energy, sections → audio_analysis.json."""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

from music_video.errors import AudioAnalysisError

# tempfile used when work_dir is omitted

TARGET_SR = 22050
MIN_DURATION_S = 10.0
TEMPO_MIN, TEMPO_MAX = 50.0, 200.0
CONF_FAIL = 0.5


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def decode_to_mono_wav(src: Path, out_wav: Path, sr: int = TARGET_SR) -> None:
    """Decode any ffmpeg-readable audio to mono PCM wav at `sr`."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-ac", "1",
        "-ar", str(sr),
        "-c:a", "pcm_s16le",
        str(out_wav),
    ]
    proc = _run(cmd)
    if proc.returncode != 0 or not out_wav.is_file() or out_wav.stat().st_size < 100:
        err = (proc.stderr or proc.stdout or "").strip()[-800:]
        raise AudioAnalysisError(f"cannot decode audio via ffmpeg: {src}\n{err}")


def load_mono(path: Path, sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    y, file_sr = sf.read(str(path), always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    y = np.asarray(y, dtype=np.float64)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
        file_sr = sr
    if not np.isfinite(y).all():
        raise AudioAnalysisError("audio contains NaN/Inf samples")
    return y, file_sr


def _validate_signal(y: np.ndarray, sr: int) -> float:
    if y.size == 0:
        raise AudioAnalysisError("audio is empty")
    dur = float(len(y) / sr)
    if dur < MIN_DURATION_S:
        raise AudioAnalysisError(f"audio too short ({dur:.2f}s); need >= {MIN_DURATION_S}s")
    rms = float(np.sqrt(np.mean(np.square(y))))
    if not math.isfinite(rms) or rms < 1e-6:
        raise AudioAnalysisError("audio is silent or near-silent")
    return dur


def _beat_confidence(beats_s: np.ndarray, onset_env: np.ndarray, onset_times: np.ndarray) -> dict[str, float]:
    """Interval regularity + onset agreement. Components recorded separately."""
    if beats_s.size < 3:
        return {
            "interval_regularity": 0.0,
            "onset_agreement": 0.0,
            "final": 0.0,
        }
    intervals = np.diff(beats_s)
    d_hub = float(np.median(intervals))
    if d_hub <= 1e-9:
        c_r = 0.0
    else:
        c_r = float(max(0.0, 1.0 - float(np.mean(np.abs(intervals - d_hub))) / d_hub))

    # onset strength at nearest onset frame to each beat
    if onset_env.size == 0 or onset_times.size == 0:
        onset_agree = 0.0
    else:
        strengths = []
        peak = float(np.max(onset_env)) or 1.0
        for b in beats_s:
            idx = int(np.argmin(np.abs(onset_times - b)))
            strengths.append(float(onset_env[idx]) / peak)
        onset_agree = float(np.clip(np.mean(strengths), 0.0, 1.0))

    final = float(0.6 * c_r + 0.4 * onset_agree)
    return {
        "interval_regularity": c_r,
        "onset_agreement": onset_agree,
        "final": final,
    }


def _synthetic_grid(duration_s: float, bpm: float = 120.0) -> tuple[list[float], float]:
    period = 60.0 / bpm
    beats = []
    t = 0.0
    while t < duration_s - 1e-6:
        beats.append(round(t, 6))
        t += period
    if not beats or beats[-1] < duration_s - period * 0.5:
        # ensure coverage hint
        pass
    return beats, bpm


def _sections_from_energy(times: np.ndarray, rms_norm: np.ndarray, duration_s: float) -> list[dict[str, Any]]:
    """Heuristic 3–5 sections from energy novelty."""
    if times.size < 4:
        return [{"id": "s0", "label": "full", "start_s": 0.0, "end_s": duration_s, "energy_mean": float(np.mean(rms_norm) if rms_norm.size else 0.0)}]

    # smooth
    win = max(3, min(11, times.size // 8 * 2 + 1))
    kernel = np.ones(win) / win
    smooth = np.convolve(rms_norm, kernel, mode="same")
    # boundaries at large positive derivative peaks + midpoints
    deriv = np.diff(smooth, prepend=smooth[0])
    thr = float(np.percentile(np.abs(deriv), 85))
    cand_idx = np.where(np.abs(deriv) >= thr)[0]
    bounds = [0.0]
    min_gap = max(4.0, duration_s / 8.0)
    for i in cand_idx:
        t = float(times[i])
        if t - bounds[-1] >= min_gap and duration_s - t >= min_gap:
            bounds.append(t)
    if duration_s - bounds[-1] < min_gap and len(bounds) > 1:
        bounds.pop()
    bounds.append(duration_s)

    labels_cycle = ["intro", "build", "drop", "bridge", "outro"]
    sections = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        mask = (times >= a) & (times < b)
        em = float(np.mean(rms_norm[mask])) if np.any(mask) else 0.0
        label = labels_cycle[min(i, len(labels_cycle) - 1)]
        sections.append({
            "id": f"s{i}",
            "label": label,
            "start_s": round(a, 6),
            "end_s": round(b, 6),
            "energy_mean": round(em, 6),
        })
    return sections


def analyze_audio_file(src: Path, work_dir: Path | None = None) -> dict[str, Any]:
    """Full analysis pipeline. HARD errors for bad input; synthetic grid only on detection failure."""
    src = Path(src)
    if not src.is_file():
        raise AudioAnalysisError(f"audio file not found: {src}")

    tmp_root = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="mv_audio_"))
    tmp_root.mkdir(parents=True, exist_ok=True)
    pcm = tmp_root / "decoded_22050_mono.wav"
    decode_to_mono_wav(src, pcm, TARGET_SR)
    y, sr = load_mono(pcm, TARGET_SR)
    duration_s = _validate_signal(y, sr)

    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    peak = float(np.max(rms)) or 1.0
    rms_norm = np.clip(rms / peak, 0.0, 1.0)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    onset_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=hop)

    grid_used = False
    fallback_reason: str | None = None
    beats_s: np.ndarray
    tempo = 0.0

    try:
        tempo_arr, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop, units="frames")
        tempo = float(np.atleast_1d(tempo_arr)[0])
        beats_s = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
        beats_s = beats_s[(beats_s >= 0) & (beats_s < duration_s)]
    except Exception as exc:  # noqa: BLE001
        beats_s = np.array([], dtype=np.float64)
        fallback_reason = f"beat_track_exception:{type(exc).__name__}"

    conf = _beat_confidence(beats_s, onset_env, onset_times)
    if beats_s.size == 0:
        grid_used = True
        fallback_reason = fallback_reason or "no_beats"
        beats_list, tempo = _synthetic_grid(duration_s, 120.0)
        conf = {
            "interval_regularity": 0.0,
            "onset_agreement": 0.0,
            "final": 0.0,
        }
    elif tempo < TEMPO_MIN or tempo > TEMPO_MAX:
        grid_used = True
        fallback_reason = f"tempo_out_of_range:{tempo:.2f}"
        beats_list, tempo = _synthetic_grid(duration_s, 120.0)
        # do not fabricate high confidence
        conf = {
            "interval_regularity": conf["interval_regularity"],
            "onset_agreement": conf["onset_agreement"],
            "final": conf["final"],
        }
    elif conf["final"] < CONF_FAIL:
        grid_used = True
        fallback_reason = f"low_confidence:{conf['final']:.3f}"
        beats_list, tempo = _synthetic_grid(duration_s, 120.0 if tempo < TEMPO_MIN or tempo > TEMPO_MAX else float(tempo))
    else:
        beats_list = [round(float(b), 6) for b in beats_s.tolist()]

    # downbeats: every 4th beat as MVP heuristic
    downbeats = beats_list[::4]

    sections = _sections_from_energy(times, rms_norm, duration_s)
    section_bounds = []
    for s in sections:
        section_bounds.extend([s["start_s"], s["end_s"]])
    cut_candidates = sorted({round(float(x), 6) for x in (beats_list + section_bounds) if 0 <= x <= duration_s})

    hop_s = float(hop / sr)
    payload: dict[str, Any] = {
        "schema": "mv.audio.v1",
        "path": str(src),
        "duration_s": round(duration_s, 6),
        "sr": sr,
        "channels": 1,
        "tempo_bpm": round(float(tempo), 4),
        "beats_s": beats_list,
        "downbeats_s": downbeats,
        "beat_confidence": {
            "interval_regularity": round(conf["interval_regularity"], 6),
            "onset_agreement": round(conf["onset_agreement"], 6),
            "final": round(conf["final"], 6),
        },
        "grid_used": grid_used,
        "fallback_reason": fallback_reason,
        "energy": {
            "hop_s": hop_s,
            "times_s": [round(float(t), 6) for t in times.tolist()],
            "rms_norm": [round(float(v), 6) for v in rms_norm.tolist()],
        },
        "sections": sections,
        "cut_candidates_s": cut_candidates,
    }
    # finite check
    _assert_finite_numbers(payload)
    return payload


def _assert_finite_numbers(obj: Any, path: str = "$") -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _assert_finite_numbers(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_finite_numbers(v, f"{path}[{i}]")
    elif isinstance(obj, (float, int)) and not isinstance(obj, bool):
        if not math.isfinite(float(obj)):
            raise AudioAnalysisError(f"non-finite analysis value at {path}")


def write_audio_analysis(src: Path, out_json: Path, work_dir: Path | None = None) -> dict[str, Any]:
    data = analyze_audio_file(src, work_dir=work_dir)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data
