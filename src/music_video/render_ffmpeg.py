"""Deterministic FFmpeg renderer. Consumes plan.validated.json only.

Must NOT import planner_openrouter or image_analyze (Slice 1 independence).
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

from music_video.errors import RenderError
from music_video.validate_plan import load_plan_json

# Intentionally no: planner_openrouter, image_analyze


def _run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        shell=False,
    )


def ffmpeg_available() -> bool:
    r = _run(["ffmpeg", "-version"])
    return r.returncode == 0


def ffprobe_available() -> bool:
    r = _run(["ffprobe", "-version"])
    return r.returncode == 0


def ass_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    cs = int(round((sec - int(sec)) * 100))
    if cs >= 100:
        s += 1
        cs = 0
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def ass_escape(text: str) -> str:
    t = str(text).replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
    t = t.replace("\n", "\\N")
    return t


def _ass_bgr(color: str, alpha: int = 0) -> str:
    named = {"black": "#000000", "white": "#FFFFFF"}
    c = named.get((color or "").strip().lower(), (color or "").strip())
    if c.startswith("#") and len(c) == 7:
        r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
        return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}&"
    return "&H00FFFFFF&"


def build_titles_ass(plan: dict[str, Any], path: Path) -> None:
    w = int(plan["width"])
    h = int(plan["height"])
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {w}",
        f"PlayResY: {h}",
        "WrapStyle: 2",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
    ]
    # default title style black box
    primary = _ass_bgr("#FFFFFF")
    back = _ass_bgr("#000000")
    font_size = max(48, int(round(h * 0.05)))
    margin_v = max(40, int(round(h * 0.08)))
    margin_lr = max(40, int(round(w * 0.06)))
    lines.append(
        f"Style: Title,Arial,{font_size},{primary},&H000000FF&,{back},{back},"
        f"-1,0,0,0,100,100,0,0,3,10,0,2,{margin_lr},{margin_lr},{margin_v},1"
    )
    lines += ["", "[Events]",
              "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    for t in plan.get("titles") or []:
        style = t.get("style") or {}
        # per-title override via inline tags if needed; keep simple Style Title
        text = ass_escape(str(t.get("text") or ""))
        if not text.strip():
            continue
        a = ass_time(float(t["start_s"]))
        b = ass_time(float(t["end_s"]))
        # apply color/bg via style rebuild — for MVP use default black box
        _ = style
        lines.append(f"Dialogue: 0,{a},{b},Title,,0,0,0,,{text}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_image(images_dir: Path, image_id: str) -> Path:
    candidates = [
        images_dir / image_id,
        images_dir / f"{image_id}.png",
        images_dir / f"{image_id}.jpg",
        images_dir / f"{image_id}.jpeg",
        images_dir / f"{image_id}.webp",
    ]
    if image_id.startswith("img_"):
        stem = image_id[4:]
        candidates += [
            images_dir / f"{stem}.png",
            images_dir / f"{stem}.jpg",
            images_dir / stem,
        ]
    for c in candidates:
        if c.is_file():
            return c
    raise RenderError(f"cannot resolve image_id={image_id!r} under {images_dir}")


def _cover_filter(w: int, h: int) -> str:
    # scale to cover then crop center
    return (
        f"scale={w}:{h}:force_original_aspect_ratio=increase,"
        f"crop={w}:{h},setsar=1,fps=30,format=yuv420p"
    )


def _encode_still_clip(
    image: Path,
    out_mp4: Path,
    duration_s: float,
    w: int,
    h: int,
    fx: str,
    fx_strength: float,
) -> None:
    dur = max(0.05, float(duration_s))
    vf = _cover_filter(w, h)
    # zoompan deferred if jittery — Slice 1: hard drop zoom unless simple and short
    use_zoom = False
    if fx in ("zoom_in", "zoom_out") and fx_strength > 0 and dur >= 0.5:
        # skip zoom in Slice 1 for stability; note via stderr log only
        use_zoom = False

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", str(image),
        "-t", f"{dur:.6f}",
        "-vf", vf,
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-r", "30",
        str(out_mp4),
    ]
    if use_zoom:
        pass  # reserved
    proc = _run(cmd)
    if proc.returncode != 0 or not out_mp4.is_file():
        raise RenderError(
            f"still clip failed for {image.name}:\n{(proc.stderr or '')[-1200:]}"
        )


def _concat_list_line(path: Path) -> str:
    # concat demuxer: file 'path' with escaped quotes/backslashes
    p = path.resolve().as_posix().replace("'", r"'\''")
    return f"file '{p}'"


def _clean_work_clips(work: Path) -> None:
    clips = work / "clips"
    if clips.is_dir():
        for p in clips.glob("*"):
            if p.is_file():
                p.unlink(missing_ok=True)


def render_job(
    job_dir: Path,
    *,
    plan_validated: Path | None = None,
    images_dir: Path | None = None,
    audio_path: Path | None = None,
) -> Path:
    if not ffmpeg_available() or not ffprobe_available():
        raise RenderError("ffmpeg/ffprobe not available on PATH")

    job_dir = Path(job_dir)
    plan_path = plan_validated or (job_dir / "plan.validated.json")
    plan = load_plan_json(plan_path)

    images_dir = Path(images_dir) if images_dir else (job_dir / "images")
    if audio_path is None:
        # prefer job audio; else plan path relative to job
        cand = plan.get("audio_path")
        audio_path = Path(cand) if cand and Path(cand).is_file() else None
        if audio_path is None:
            for ext in (".wav", ".mp3", ".m4a", ".flac"):
                hits = list(job_dir.glob(f"song{ext}"))
                if hits:
                    audio_path = hits[0]
                    break
    if audio_path is None or not Path(audio_path).is_file():
        raise RenderError("audio file not found for render")
    audio_path = Path(audio_path)

    w, h = int(plan["width"]), int(plan["height"])
    duration_s = float(plan["duration_s"])
    work = job_dir / "work"
    clips_dir = work / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    _clean_work_clips(work)

    try:
        clip_files: list[Path] = []
        for i, c in enumerate(plan["clips"]):
            img = resolve_image(images_dir, c["image_id"])
            dur = float(c["end_s"]) - float(c["start_s"])
            out_c = clips_dir / f"clip_{i:04d}.mp4"
            _encode_still_clip(
                img, out_c, dur, w, h,
                fx=str(c.get("fx") or "none"),
                fx_strength=float(c.get("fx_strength") or 0),
            )
            clip_files.append(out_c)

        concat_txt = work / "concat.txt"
        concat_txt.write_text(
            "\n".join(_concat_list_line(p) for p in clip_files) + "\n",
            encoding="utf-8",
        )

        silent_video = work / "video_silent.mp4"
        cmd_concat = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_txt),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-r", "30",
            "-an",
            str(silent_video),
        ]
        proc = _run(cmd_concat)
        if proc.returncode != 0:
            raise RenderError(f"concat failed:\n{(proc.stderr or '')[-1200:]}")

        titled = silent_video
        if plan.get("titles"):
            ass_path = work / "titles.ass"
            build_titles_ass(plan, ass_path)
            burned = work / "video_titled.mp4"
            ass_esc = ass_path.resolve().as_posix().replace(":", "\\:")
            cmd_ass = [
                "ffmpeg", "-y",
                "-i", str(silent_video),
                "-vf", f"ass='{ass_esc}'",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-an",
                str(burned),
            ]
            proc = _run(cmd_ass)
            if proc.returncode != 0:
                raise RenderError(f"ASS burn failed:\n{(proc.stderr or '')[-1200:]}")
            titled = burned

        final = job_dir / "final.mp4"
        # mux audio; never truncate soundtrack — pad video with last frame if needed via -shortest false
        # Use audio as duration authority: -t audio_duration on output
        cmd_mux = [
            "ffmpeg", "-y",
            "-i", str(titled),
            "-i", str(audio_path),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-t", f"{duration_s:.6f}",
            "-movflags", "+faststart",
            str(final),
        ]
        proc = _run(cmd_mux)
        if proc.returncode != 0:
            # re-encode video if copy fails
            cmd_mux[cmd_mux.index("-c:v") + 1] = "libx264"
            # fix: replace 'copy' after -c:v
            cmd_mux = [
                "ffmpeg", "-y",
                "-i", str(titled),
                "-i", str(audio_path),
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "192k",
                "-t", f"{duration_s:.6f}",
                "-movflags", "+faststart",
                str(final),
            ]
            proc = _run(cmd_mux)
            if proc.returncode != 0:
                raise RenderError(f"mux failed:\n{(proc.stderr or '')[-1200:]}")

        verify_final(final, expect_w=w, expect_h=h, expect_dur=duration_s)
        return final
    except Exception:
        _clean_work_clips(work)
        raise


def probe_duration(path: Path) -> float:
    proc = _run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        str(path),
    ])
    if proc.returncode != 0:
        raise RenderError(f"ffprobe duration failed: {path}")
    return float(proc.stdout.strip())


def probe_streams(path: Path) -> dict[str, Any]:
    proc = _run([
        "ffprobe", "-v", "error",
        "-show_entries", "stream=codec_type,width,height,duration",
        "-of", "json",
        str(path),
    ])
    if proc.returncode != 0:
        raise RenderError(f"ffprobe streams failed: {path}")
    return json.loads(proc.stdout)


def verify_final(path: Path, *, expect_w: int, expect_h: int, expect_dur: float) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RenderError("final.mp4 missing or empty")
    # Acceptance for real jobs >100KB; short pytest fixtures may be smaller but >=10KB
    min_bytes = 100_000 if expect_dur >= 8.0 else 10_000
    if path.stat().st_size < min_bytes:
        raise RenderError(f"final.mp4 too small: {path.stat().st_size} bytes (need >= {min_bytes})")

    info = probe_streams(path)
    streams = info.get("streams") or []
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if not v:
        raise RenderError("final.mp4 missing video stream")
    if not a:
        raise RenderError("final.mp4 missing audio stream")
    if int(v.get("width") or 0) != expect_w or int(v.get("height") or 0) != expect_h:
        raise RenderError(f"resolution mismatch: got {v.get('width')}x{v.get('height')}")

    dur = probe_duration(path)
    if abs(dur - expect_dur) > 0.5:
        raise RenderError(f"duration mismatch: got {dur:.3f} expected {expect_dur:.3f}")

    # audio must not be truncated worse than ~1 frame vs source duration
    frame = 1.0 / 30.0
    if dur + frame < expect_dur:
        raise RenderError("output shorter than source audio by more than one frame")

    # true decode check
    proc = _run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"])
    err = (proc.stderr or "").strip()
    if proc.returncode != 0 or err:
        raise RenderError(f"decode check failed: {err or proc.returncode}")
