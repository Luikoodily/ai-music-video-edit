"""Deterministic timeline expander: mv.sectionplan.v1 → mv.plan.v1.

Independent of OpenRouter and FFmpeg. Consumes beats_s from audio_analysis.json
as the single authoritative grid (including M1 synthetic fallback).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from music_video.errors import TimelineExpandError
from music_video.models import aspect_size
from music_video.validate_plan import validate_plan

SECTION_EDGE_TOL = 0.15  # snap section edges to beat only within this
DENSITY_ENUM = {"sparse", "medium", "dense"}


def _finite(name: str, v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError) as exc:
        raise TimelineExpandError(f"{name} not a number") from exc
    if not math.isfinite(x):
        raise TimelineExpandError(f"{name} is NaN/Inf")
    return x


def _nearest_beat(t: float, beats: list[float]) -> float | None:
    if not beats:
        return None
    return min(beats, key=lambda b: abs(b - t))


def snap_section_edge(t: float, beats: list[float], *, tol: float = SECTION_EDGE_TOL) -> float:
    nb = _nearest_beat(t, beats)
    if nb is not None and abs(nb - t) <= tol:
        return float(nb)
    return float(t)


def beats_in_range(beats: list[float], start: float, end: float) -> list[float]:
    return [b for b in beats if start - 1e-9 <= b <= end + 1e-9]


def half_beats(beats: list[float], start: float, end: float) -> list[float]:
    """Dense helper: beat midpoints inside [start, end]."""
    inside = beats_in_range(beats, start, end)
    mids: list[float] = []
    for a, b in zip(inside, inside[1:]):
        mid = (a + b) / 2.0
        if start < mid < end:
            mids.append(mid)
    return mids


def cut_points_for_density(
    *,
    density: str,
    start: float,
    end: float,
    beats: list[float],
    downbeats: list[float],
) -> list[float]:
    """Return cut boundaries including start; caller adds end."""
    if density not in DENSITY_ENUM:
        raise TimelineExpandError(f"bad density: {density}")
    if end <= start:
        raise TimelineExpandError(f"section empty: {start}-{end}")

    local = beats_in_range(beats, start, end)
    downs = beats_in_range(downbeats, start, end)

    cuts = [start]
    if density == "sparse":
        # every 4–8 beats, prefer downbeats (use every downbeat ≈ every 4 beats)
        preferred = downs if len(downs) >= 2 else local[::6]  # middle of 4–8
        if not preferred:
            preferred = local[::6]
        for b in preferred:
            if b > start + 1e-6 and b < end - 1e-6:
                cuts.append(b)
    elif density == "medium":
        # every 1–2 beats
        step = 2 if len(local) > 8 else 1
        for b in local[::step]:
            if b > start + 1e-6 and b < end - 1e-6:
                cuts.append(b)
    else:  # dense
        for b in local:
            if b > start + 1e-6 and b < end - 1e-6:
                cuts.append(b)
        # half-beats where appropriate (section longer than ~2 beat periods)
        if len(local) >= 2:
            for m in half_beats(beats, start, end):
                cuts.append(m)

    cuts = sorted(set(round(c, 6) for c in cuts))
    # ensure start present
    if not cuts or abs(cuts[0] - start) > 1e-9:
        cuts = [round(start, 6)] + [c for c in cuts if c > start]
    return cuts


def _rank_key(image_id: str, section_id: str, image_meta: dict[str, dict[str, Any]], usage: dict[str, int]) -> tuple:
    """Lower is better: prefer best_sections hit, then lower usage, then id."""
    meta = image_meta.get(image_id) or {}
    best = list(meta.get("best_sections") or [])
    try:
        rank = best.index(section_id)
    except ValueError:
        rank = 10_000
    return (rank, usage.get(image_id, 0), image_id)


def merge_short_intervals(
    intervals: list[tuple[float, float]],
    *,
    min_clip_s: float,
) -> list[tuple[float, float]]:
    """Merge adjacent intervals until each (except possibly last) is >= min_clip_s."""
    if not intervals:
        return []
    out: list[tuple[float, float]] = [intervals[0]]
    for a, b in intervals[1:]:
        pa, pb = out[-1]
        if (pb - pa) < min_clip_s - 1e-9:
            out[-1] = (pa, b)
        elif (b - a) < min_clip_s - 1e-9 and len(out) >= 1:
            # absorb short next into previous when still under 2*min would be odd —
            # instead start new and let later merge
            out.append((a, b))
        else:
            out.append((a, b))
    # second pass: merge trailing shorts into previous
    changed = True
    while changed and len(out) > 1:
        changed = False
        merged: list[tuple[float, float]] = []
        i = 0
        while i < len(out):
            a, b = out[i]
            if (b - a) < min_clip_s - 1e-9 and merged:
                pa, _pb = merged[-1]
                merged[-1] = (pa, b)
                changed = True
                i += 1
                continue
            if (b - a) < min_clip_s - 1e-9 and i + 1 < len(out):
                _na, nb = out[i + 1]
                merged.append((a, nb))
                changed = True
                i += 2
                continue
            merged.append((a, b))
            i += 1
        out = merged
    return out


def allocate_images(
    image_ids: list[str],
    n_slots: int,
    *,
    section_id: str,
    image_meta: dict[str, dict[str, Any]],
    usage: dict[str, int],
    prev_image: str | None,
    notes: list[str],
) -> list[str]:
    """Deterministic allocation; avoid consecutive duplicates when alternatives exist."""
    if n_slots <= 0:
        return []
    if not image_ids:
        raise TimelineExpandError(f"section {section_id} has no image_ids")

    pool = list(dict.fromkeys(image_ids))  # stable unique preserve order
    chosen: list[str] = []
    for _ in range(n_slots):
        candidates = sorted(pool, key=lambda iid: _rank_key(iid, section_id, image_meta, usage))
        pick = None
        for cand in candidates:
            if prev_image is None or cand != prev_image or len(candidates) == 1:
                pick = cand
                break
        if pick is None:
            pick = candidates[0]
        if prev_image is not None and pick == prev_image:
            notes.append(f"soft: consecutive duplicate {pick} in {section_id}")
        chosen.append(pick)
        usage[pick] = usage.get(pick, 0) + 1
        prev_image = pick
    return chosen


def expand_section_plan(
    section_plan: dict[str, Any],
    audio: dict[str, Any],
    images: list[dict[str, Any]] | None = None,
    *,
    audio_path: str = "song.wav",
    fps: int = 30,
    min_clip_s: float = 0.35,
    max_clip_s: float = 2.5,
    title_max: int = 4,
) -> dict[str, Any]:
    """Expand to mv.plan.v1 then run through M3 validate_plan (never bypass)."""
    if section_plan.get("schema") != "mv.sectionplan.v1":
        raise TimelineExpandError("section_plan.schema must be mv.sectionplan.v1")

    duration_s = _finite("audio.duration_s", audio["duration_s"])
    beats_raw = audio.get("beats_s")
    if not isinstance(beats_raw, list) or not beats_raw:
        raise TimelineExpandError("audio_analysis.beats_s missing/empty — refuse to invent grid")
    beats = [_finite(f"beats[{i}]", b) for i, b in enumerate(beats_raw)]
    downbeats = [_finite(f"downbeats[{i}]", b) for i, b in enumerate(audio.get("downbeats_s") or beats[::4])]

    audio_sections = {str(s["id"]): s for s in (audio.get("sections") or [])}
    if not audio_sections:
        raise TimelineExpandError("audio_analysis.sections missing")

    aspect = section_plan.get("aspect") or "9:16"
    w, h = aspect_size(aspect)

    image_meta = {im["id"]: im for im in (images or []) if isinstance(im, dict) and "id" in im}
    usage: dict[str, int] = {}
    notes: list[str] = list(section_plan.get("notes") or [])
    if isinstance(notes, str):
        notes = [notes]

    # record grid provenance
    if audio.get("grid_used"):
        notes.append(
            f"expander: using audio beats_s (grid_used=true, fallback_reason={audio.get('fallback_reason')})"
        )
    conf = audio.get("beat_confidence") or {}
    if isinstance(conf, dict) and conf.get("final") is not None:
        notes.append(f"expander: beat_confidence.final={conf.get('final')}")

    clips: list[dict[str, Any]] = []
    titles: list[dict[str, Any]] = []
    prev_image: str | None = None
    clip_i = 0
    title_i = 0

    plan_sections = section_plan.get("sections") or []
    if not plan_sections:
        raise TimelineExpandError("section_plan.sections empty")

    for sec in plan_sections:
        sid = str(sec["section_id"])
        if sid not in audio_sections:
            raise TimelineExpandError(f"unknown section_id: {sid}")
        asec = audio_sections[sid]
        raw_start = _finite(f"{sid}.start_s", asec["start_s"])
        raw_end = _finite(f"{sid}.end_s", asec["end_s"])
        start = snap_section_edge(raw_start, beats)
        end = snap_section_edge(raw_end, beats)
        start = max(0.0, min(start, duration_s))
        end = max(0.0, min(end, duration_s))
        if end <= start:
            raise TimelineExpandError(f"section {sid} collapsed after snap")

        density = str(sec.get("density") or "medium")
        cuts = cut_points_for_density(
            density=density,
            start=start,
            end=end,
            beats=beats,
            downbeats=downbeats,
        )
        # build intervals cuts[i] → cuts[i+1] / end
        bounds = cuts + [round(end, 6)]
        # drop zero-length
        intervals: list[tuple[float, float]] = []
        for a, b in zip(bounds, bounds[1:]):
            if b - a > 1e-6:
                intervals.append((a, b))
        if not intervals:
            intervals = [(start, end)]
        # prevent M3 min_clip soft-extend from creating HARD overlaps
        intervals = merge_short_intervals(intervals, min_clip_s=min_clip_s)

        ids = list(sec.get("image_ids") or [])
        allocated = allocate_images(
            ids,
            len(intervals),
            section_id=sid,
            image_meta=image_meta,
            usage=usage,
            prev_image=prev_image,
            notes=notes,
        )
        for (a, b), iid in zip(intervals, allocated):
            clip_i += 1
            clips.append({
                "id": f"c{clip_i:03d}",
                "image_id": iid,
                "start_s": float(a),
                "end_s": float(b),
                "transition_in": "cut",
                "fx": "none",
                "fx_strength": 0.0,
                "fit": "cover",
            })
            prev_image = iid

        title = sec.get("title")
        if title and isinstance(title, dict) and str(title.get("text") or "").strip():
            if title_i < title_max:
                title_i += 1
                # snap title window to beats inside section: start at first cut, ~2 beats or section end
                t0 = start
                nb = _nearest_beat(start, beats)
                if nb is not None and abs(nb - start) <= 0.08:
                    t0 = nb
                # end ~ min(section_end, t0 + 2 beat periods)
                period = 0.5
                if len(beats) >= 2:
                    period = max(0.25, float(beats[1] - beats[0]))
                t1 = min(end, t0 + 2.0 * period)
                nb1 = _nearest_beat(t1, beats)
                if nb1 is not None and abs(nb1 - t1) <= 0.08 and nb1 > t0:
                    t1 = nb1
                if t1 <= t0:
                    t1 = min(end, t0 + min_clip_s)
                style = title.get("style") or {"color": "#FFFFFF", "bg": "#000000", "bg_pad": 12}
                titles.append({
                    "id": f"t{title_i:03d}",
                    "text": str(title["text"]),
                    "start_s": float(t0),
                    "end_s": float(t1),
                    "position": "center",
                    "style": {
                        "color": str(style.get("color") or "#FFFFFF"),
                        "bg": str(style.get("bg") or "#000000"),
                        "bg_pad": int(style.get("bg_pad") or 12),
                    },
                })
            else:
                notes.append(f"soft: dropped title for {sid} — title_max={title_max}")

    # abut sections: force contiguous coverage [0, duration]
    if clips:
        clips.sort(key=lambda c: c["start_s"])
        clips[0]["start_s"] = 0.0
        for a, b in zip(clips, clips[1:]):
            if abs(a["end_s"] - b["start_s"]) > 1e-6:
                # close tiny gaps / overlaps by abutting
                mid = a["end_s"]
                b["start_s"] = mid
                if b["end_s"] <= b["start_s"]:
                    raise TimelineExpandError(f"degenerate abut at {b['id']}")
        clips[-1]["end_s"] = duration_s

    raw_plan = {
        "schema": "mv.plan.v1",
        "aspect": aspect,
        "width": w,
        "height": h,
        "fps": fps,
        "duration_s": duration_s,
        "audio_path": audio_path,
        "clips": clips,
        "titles": titles,
        "notes": notes,
    }

    known = set(image_meta) if image_meta else None
    # also allow ids from section plan
    plan_ids = {c["image_id"] for c in clips}
    if known is None:
        known = plan_ids
    else:
        known |= plan_ids

    validated = validate_plan(
        raw_plan,
        audio_duration_s=duration_s,
        known_image_ids=known,
        beats_s=beats,
        min_clip_s=min_clip_s,
        max_clip_s=max_clip_s,
        title_max=title_max,
        fps=fps,
    )
    return validated


def expand_job_files(
    job_dir: Path,
    *,
    section_plan_file: Path | None = None,
    audio_file: Path | None = None,
    image_file: Path | None = None,
    write_plan: bool = True,
    write_validated: bool = True,
) -> dict[str, Any]:
    from music_video.paths import (
        audio_analysis_path,
        image_analysis_path,
        plan_path,
        section_plan_path,
        validated_plan_path,
    )

    job_dir = Path(job_dir)
    sp = json.loads((section_plan_file or section_plan_path(job_dir)).read_text(encoding="utf-8"))
    audio = json.loads((audio_file or audio_analysis_path(job_dir)).read_text(encoding="utf-8"))
    images = []
    img_path = image_file or image_analysis_path(job_dir)
    if Path(img_path).is_file():
        images = json.loads(Path(img_path).read_text(encoding="utf-8")).get("images") or []

    # find song path
    audio_path = "song.wav"
    for ext in (".wav", ".mp3", ".m4a"):
        hits = list(job_dir.glob(f"song{ext}"))
        if hits:
            audio_path = hits[0].name
            break

    validated = expand_section_plan(sp, audio, images, audio_path=audio_path)
    if write_plan:
        # raw-ish: store validated as plan.json too for M5 path (validator already applied)
        plan_path(job_dir).write_text(json.dumps(validated, indent=2), encoding="utf-8")
    if write_validated:
        validated_plan_path(job_dir).write_text(json.dumps(validated, indent=2), encoding="utf-8")
    return validated
