"""Validate / soft-repair MusicVideoPlan. HARD fails raise; SOFT fixes recorded in notes."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from music_video.errors import HardValidationError
from music_video.models import ASPECT_ENUM, FIT_ENUM, FX_ENUM, TRANSITION_ENUM, aspect_size

DEFAULT_MIN_CLIP = 0.35
DEFAULT_MAX_CLIP = 2.5
DEFAULT_TITLE_MAX = 4
DEFAULT_TITLE_CHARS = 48
SNAP_TOL = 0.08  # soft preferred snap only — not mandatory beat law


def _reject_constants(s: str) -> Any:
    raise HardValidationError(f"non-finite JSON constant: {s}")


def load_plan_json(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw, parse_constant=_reject_constants)
    except json.JSONDecodeError as exc:
        raise HardValidationError(f"malformed plan JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise HardValidationError("plan root must be object")
    return data


def _finite(name: str, v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError) as exc:
        raise HardValidationError(f"{name} not a number") from exc
    if not math.isfinite(x):
        raise HardValidationError(f"{name} is NaN/Inf")
    return x


def _snap(t: float, beats: list[float], tol: float) -> tuple[float, bool]:
    if not beats:
        return t, False
    best = min(beats, key=lambda b: abs(b - t))
    if abs(best - t) <= tol:
        return float(best), abs(best - t) > 1e-9
    return t, False


def validate_plan(
    plan: dict[str, Any],
    *,
    audio_duration_s: float,
    known_image_ids: set[str] | None = None,
    beats_s: list[float] | None = None,
    min_clip_s: float = DEFAULT_MIN_CLIP,
    max_clip_s: float = DEFAULT_MAX_CLIP,
    title_max: int = DEFAULT_TITLE_MAX,
    title_max_chars: int = DEFAULT_TITLE_CHARS,
    fps: int = 30,
) -> dict[str, Any]:
    """Return validated plan dict. Audio duration is authoritative."""
    notes: list[str] = list(plan.get("notes") or [])
    if isinstance(notes, str):
        notes = [notes]

    audio_duration_s = _finite("audio_duration_s", audio_duration_s)
    if audio_duration_s <= 0:
        raise HardValidationError("audio_duration_s must be > 0")

    schema = plan.get("schema")
    if schema != "mv.plan.v1":
        raise HardValidationError("schema must be mv.plan.v1")

    aspect = plan.get("aspect")
    if aspect not in ASPECT_ENUM:
        raise HardValidationError(f"aspect must be one of {sorted(ASPECT_ENUM)}")

    w, h = aspect_size(aspect)
    # ignore plan width/height if wrong — set from aspect (soft note)
    if int(plan.get("width") or 0) != w or int(plan.get("height") or 0) != h:
        notes.append(f"soft: width/height set from aspect {aspect} -> {w}x{h}")

    fps = int(plan.get("fps") or fps)
    if fps <= 0:
        raise HardValidationError("fps must be > 0")
    frame_eps = 1.0 / fps

    # duration from audio wins
    if "duration_s" in plan:
        try:
            pd = float(plan["duration_s"])
            if math.isfinite(pd) and abs(pd - audio_duration_s) > frame_eps:
                notes.append(
                    f"soft: plan.duration_s {pd} overridden by audio_analysis {audio_duration_s}"
                )
        except (TypeError, ValueError):
            notes.append("soft: plan.duration_s unreadable; using audio duration")

    audio_path = plan.get("audio_path")
    if not isinstance(audio_path, str) or not audio_path.strip():
        raise HardValidationError("audio_path required")

    clips_in = plan.get("clips")
    if not isinstance(clips_in, list) or not clips_in:
        raise HardValidationError("clips must be a non-empty list")

    beats = [float(b) for b in (beats_s or []) if math.isfinite(float(b))]
    known = known_image_ids  # None => skip unknown check (render may supply)

    clips: list[dict[str, Any]] = []
    for i, raw in enumerate(clips_in):
        if not isinstance(raw, dict):
            raise HardValidationError(f"clips[{i}] must be object")
        cid = str(raw.get("id") or f"c{i:03d}")
        image_id = raw.get("image_id")
        if not isinstance(image_id, str) or not image_id.strip():
            raise HardValidationError(f"clips[{i}].image_id missing")
        if known is not None and image_id not in known:
            raise HardValidationError(f"unknown image_id: {image_id}")

        start = _finite(f"clips[{i}].start_s", raw.get("start_s"))
        end = _finite(f"clips[{i}].end_s", raw.get("end_s"))
        # HARD: out-of-range timestamps (distinct from SOFT duration clamp)
        if start < 0 or end < 0 or start >= audio_duration_s or end > audio_duration_s + frame_eps:
            raise HardValidationError(f"clips[{i}] invalid timestamps start={start} end={end}")
        if end <= start:
            raise HardValidationError(f"clips[{i}] end_s must be > start_s")

        # preferred soft snap (not mandatory beat law)
        start2, s1 = _snap(start, beats, SNAP_TOL)
        end2, s2 = _snap(end, beats, SNAP_TOL)
        if s1 or s2:
            notes.append(f"soft: snapped clip {cid} toward beats (±{SNAP_TOL}s)")
        start, end = start2, end2
        if end <= start:
            end = min(audio_duration_s, start + min_clip_s)

        dur = end - start
        if dur < min_clip_s:
            end = min(audio_duration_s, start + min_clip_s)
            notes.append(f"soft: clamp {cid} up to min_clip_s={min_clip_s}")
        elif dur > max_clip_s and end < audio_duration_s - frame_eps:
            # clamp body clips; last clip handled after sort
            end = start + max_clip_s
            notes.append(f"soft: clamp {cid} down to max_clip_s={max_clip_s}")

        fit = raw.get("fit", "cover")
        if fit not in FIT_ENUM:
            raise HardValidationError(f"clips[{i}].fit must be one of {sorted(FIT_ENUM)}")

        tr = raw.get("transition_in", "cut")
        if tr not in TRANSITION_ENUM:
            raise HardValidationError(f"clips[{i}].transition_in invalid")

        fx = raw.get("fx", "none")
        if fx not in FX_ENUM:
            raise HardValidationError(f"clips[{i}].fx invalid")
        fx_strength = _finite(f"clips[{i}].fx_strength", raw.get("fx_strength", 0.0))

        clips.append({
            "id": cid,
            "image_id": image_id,
            "start_s": float(start),
            "end_s": float(end),
            "transition_in": "cut",  # MVP hard cuts only
            "fx": fx,
            "fx_strength": fx_strength,
            "fit": fit,
        })

    clips.sort(key=lambda c: c["start_s"])

    # HARD overlap check before soft gap fill
    for a, b in zip(clips, clips[1:]):
        if b["start_s"] < a["end_s"] - 1e-6:
            raise HardValidationError(
                f"overlapping clips {a['id']} and {b['id']}: "
                f"{a['start_s']}-{a['end_s']} vs {b['start_s']}-{b['end_s']}"
            )

    # adjacent duplicate soft note
    for a, b in zip(clips, clips[1:]):
        if a["image_id"] == b["image_id"]:
            notes.append(f"soft: adjacent duplicate image_id {a['image_id']} ({a['id']}/{b['id']})")

    # gap fill — previous image persists
    filled: list[dict[str, Any]] = []
    cursor = 0.0
    for c in clips:
        if c["start_s"] > cursor + frame_eps:
            # gap
            prev_img = filled[-1]["image_id"] if filled else c["image_id"]
            gap_id = f"gap_{len(filled):03d}"
            filled.append({
                "id": gap_id,
                "image_id": prev_img,
                "start_s": cursor,
                "end_s": c["start_s"],
                "transition_in": "cut",
                "fx": "none",
                "fx_strength": 0.0,
                "fit": "cover",
            })
            notes.append(f"soft: gap fill {cursor:.3f}-{c['start_s']:.3f} with {prev_img}")
        # if clip starts slightly before cursor due to snap, push start
        if c["start_s"] < cursor - 1e-9:
            raise HardValidationError(f"clip {c['id']} overlaps after gap policy")
        if abs(c["start_s"] - cursor) <= frame_eps:
            c["start_s"] = cursor
        filled.append(c)
        cursor = c["end_s"]

    if cursor < audio_duration_s - frame_eps:
        prev_img = filled[-1]["image_id"]
        filled.append({
            "id": f"gap_tail",
            "image_id": prev_img,
            "start_s": cursor,
            "end_s": audio_duration_s,
            "transition_in": "cut",
            "fx": "none",
            "fx_strength": 0.0,
            "fit": "cover",
        })
        notes.append(f"soft: tail fill to audio duration with {prev_img}")
        cursor = audio_duration_s

    # ensure last end == duration within 1 frame
    filled[-1]["end_s"] = audio_duration_s
    if abs(filled[-1]["end_s"] - audio_duration_s) > frame_eps:
        raise HardValidationError("final clip end != audio duration")

    # re-check overlaps after fill
    for a, b in zip(filled, filled[1:]):
        if b["start_s"] < a["end_s"] - 1e-6:
            # abut: force
            b["start_s"] = a["end_s"]
        if b["start_s"] >= b["end_s"]:
            raise HardValidationError(f"degenerate clip after abut {b['id']}")

    titles_in = plan.get("titles") or []
    if not isinstance(titles_in, list):
        raise HardValidationError("titles must be a list")
    if len(titles_in) > title_max:
        raise HardValidationError(f"titles exceed max {title_max}")

    titles: list[dict[str, Any]] = []
    for i, raw in enumerate(titles_in):
        if not isinstance(raw, dict):
            raise HardValidationError(f"titles[{i}] must be object")
        text = str(raw.get("text") or "")
        if len(text) > title_max_chars:
            text = text[:title_max_chars]
            notes.append(f"soft: trimmed title {raw.get('id')} to {title_max_chars} chars")
        ts = _finite(f"titles[{i}].start_s", raw.get("start_s"))
        te = _finite(f"titles[{i}].end_s", raw.get("end_s"))
        if te <= ts or ts < 0 or te > audio_duration_s + frame_eps:
            raise HardValidationError(f"titles[{i}] invalid timestamps")
        style = raw.get("style") or {"color": "#FFFFFF", "bg": "#000000", "bg_pad": 12}
        if not isinstance(style, dict):
            raise HardValidationError(f"titles[{i}].style must be object")
        titles.append({
            "id": str(raw.get("id") or f"t{i:03d}"),
            "text": text,
            "start_s": ts,
            "end_s": min(te, audio_duration_s),
            "position": str(raw.get("position") or "center"),
            "style": {
                "color": str(style.get("color") or "#FFFFFF"),
                "bg": str(style.get("bg") or "#000000"),
                "bg_pad": int(style.get("bg_pad") or 12),
            },
        })

    out = {
        "schema": "mv.plan.v1",
        "aspect": aspect,
        "width": w,
        "height": h,
        "fps": fps,
        "duration_s": audio_duration_s,
        "audio_path": audio_path,
        "clips": filled,
        "titles": titles,
        "notes": notes,
    }
    return out


def validate_plan_files(
    plan_path: Path,
    audio_analysis_path: Path,
    out_path: Path,
    images_dir: Path | None = None,
) -> dict[str, Any]:
    plan = load_plan_json(plan_path)
    audio = json.loads(
        audio_analysis_path.read_text(encoding="utf-8"),
        parse_constant=_reject_constants,
    )
    duration = _finite("audio.duration_s", audio["duration_s"])
    beats = audio.get("beats_s") or []

    known: set[str] | None = None
    if images_dir and images_dir.is_dir():
        known = {p.stem for p in images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}}
        # also allow img_XX style ids if files named that way
        known |= {p.name for p in images_dir.iterdir() if p.is_file()}
        # map plan image ids: accept stem matches
        # For golden plans we use logical ids; pass explicit map via optional sidecar later.
        # If plan uses img_01 and file is 01.png, allow both stem and stem with img_ prefix helpers:
        known |= {f"img_{p.stem}" for p in images_dir.iterdir() if p.is_file()}
        known |= {p.stem for p in images_dir.iterdir() if p.is_file()}

    # If known set empty-ish, don't enforce for hand-written prove jobs that resolve paths separately
    if known is not None and len(known) == 0:
        known = None

    validated = validate_plan(
        plan,
        audio_duration_s=duration,
        known_image_ids=None,  # path resolution deferred to renderer map; schema tests pass ids explicitly
        beats_s=[float(b) for b in beats],
        fps=int(plan.get("fps") or 30),
    )
    # optional known check when caller wants it
    if images_dir is not None and known:
        for c in validated["clips"]:
            # renderer will resolve; soft note if stem missing
            iid = c["image_id"]
            if iid not in known and not any(Path(images_dir, f"{iid}{ext}").is_file() for ext in (".png", ".jpg", ".jpeg", ".webp")):
                # try stripping img_
                alt = iid[4:] if iid.startswith("img_") else iid
                if not any(Path(images_dir, f"{alt}{ext}").is_file() for ext in (".png", ".jpg", ".jpeg", ".webp")):
                    raise HardValidationError(f"unknown image_id: {iid}")

    out_path.write_text(json.dumps(validated, indent=2), encoding="utf-8")
    return validated
