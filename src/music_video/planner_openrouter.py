"""M4: OpenRouter creative director — section-level JSON only (mv.sectionplan.v1).

urllib only. No FFmpeg. No per-clip timestamps from the LLM.
Transport is injectable for mocked tests (zero network in ordinary pytest).
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from typing import Any, Callable

from music_video.env_load import load_dotenv
from music_video.errors import PlannerError

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TIMEOUT_S = 180.0
TEMPERATURE = 0.4
MAX_BEAT_SAMPLES = 400
DENSITY_ENUM = {"sparse", "medium", "dense"}

HttpTransport = Callable[[str, dict[str, str], bytes, float], tuple[int, str]]

SYSTEM_PROMPT = """You are a music-video creative director.
Return ONLY valid JSON (no markdown fences) with schema mv.sectionplan.v1.
Assign ordered image_ids and a density (sparse|medium|dense) PER AUDIO SECTION.
Optional title suggestions may include text + style {color,bg,bg_pad} only.
Rules:
- Use ONLY provided image ids. Never invent ids.
- Never output per-clip timestamps, frame times, FFmpeg, or renderer instructions.
- Never leave a section with an empty image_ids list; if stuck, put a note and reuse best available ids.
- Prefer denser cuts on high-energy sections (drop/chorus); sparse on intro/outro.
JSON shape:
{
  "schema": "mv.sectionplan.v1",
  "aspect": "9:16",
  "sections": [
    {"section_id": "s0", "image_ids": ["img_01"], "density": "sparse",
     "title": {"text": "RISE", "style": {"color": "#FFFFFF", "bg": "#000000", "bg_pad": 12}}}
  ],
  "notes": []
}
"""


def _reject_constants(s: str) -> Any:
    raise PlannerError(f"non-finite JSON constant: {s}")


def _finite(name: str, v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError) as exc:
        raise PlannerError(f"{name} not a number") from exc
    if not math.isfinite(x):
        raise PlannerError(f"{name} is NaN/Inf")
    return x


def strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def default_http_transport(url: str, headers: dict[str, str], body: bytes, timeout_s: float) -> tuple[int, str]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return int(resp.status), resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
        return int(exc.code), raw
    except Exception as exc:  # noqa: BLE001
        raise PlannerError(f"OpenRouter transport error: {type(exc).__name__}: {exc}") from exc


def require_openrouter_config(*, load_env: bool = True) -> tuple[str, str, list[str]]:
    if load_env:
        load_dotenv()
    key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    model = (os.environ.get("OPENROUTER_MODEL") or "").strip()
    if not key:
        raise PlannerError("OPENROUTER_API_KEY missing (set env or music-video/.env)")
    if not model:
        raise PlannerError("OPENROUTER_MODEL missing (set env or music-video/.env); model is never assumed")
    # strip optional openrouter/ prefix
    if model.startswith("openrouter/"):
        model = model[len("openrouter/") :]
    fallbacks_raw = (os.environ.get("OPENROUTER_FALLBACK_MODELS") or "").strip()
    fallbacks: list[str] = []
    if fallbacks_raw:
        for part in fallbacks_raw.split(","):
            m = part.strip()
            if m.startswith("openrouter/"):
                m = m[len("openrouter/") :]
            if m and m != model:
                fallbacks.append(m)
    return key, model, fallbacks


def downsample_beats(beats: list[float], cap: int = MAX_BEAT_SAMPLES) -> list[float]:
    if len(beats) <= cap:
        return [round(float(b), 6) for b in beats]
    step = max(1, int(math.ceil(len(beats) / cap)))
    sampled = [beats[i] for i in range(0, len(beats), step)]
    if sampled[-1] != beats[-1]:
        sampled.append(beats[-1])
    return [round(float(b), 6) for b in sampled[:cap]]


def build_planner_input(
    *,
    style_instruction: str,
    aspect: str,
    audio: dict[str, Any],
    images: list[dict[str, Any]],
    min_clip_s: float = 0.35,
    max_clip_s: float = 2.5,
    title_max: int = 4,
) -> dict[str, Any]:
    sections = audio.get("sections") or []
    beats = audio.get("beats_s") or []
    img_payload = []
    for im in images:
        if im.get("unused"):
            continue
        img_payload.append({
            "id": im["id"],
            "labels": list(im.get("labels") or []),
            "best_sections": list(im.get("best_sections") or []),
        })
    if not img_payload:
        raise PlannerError("no usable images for planner (all unused or empty)")
    return {
        "style_instruction": style_instruction,
        "aspect": aspect,
        "audio": {
            "duration_s": _finite("duration_s", audio["duration_s"]),
            "tempo_bpm": _finite("tempo_bpm", audio.get("tempo_bpm", 0)),
            "sections": [
                {
                    "id": s.get("id"),
                    "label": s.get("label"),
                    "start_s": _finite(f"{s.get('id')}.start_s", s["start_s"]),
                    "end_s": _finite(f"{s.get('id')}.end_s", s["end_s"]),
                }
                for s in sections
            ],
            "beats_s_sample": downsample_beats([float(b) for b in beats]),
        },
        "images": img_payload,
        "rules": {
            "min_clip_s": min_clip_s,
            "max_clip_s": max_clip_s,
            "prefer_cut_on_beats": True,
            "title_max": title_max,
        },
    }


def validate_section_plan(
    plan: dict[str, Any],
    *,
    known_image_ids: set[str],
    known_section_ids: set[str],
    aspect: str,
) -> dict[str, Any]:
    if not isinstance(plan, dict):
        raise PlannerError("section plan root must be object")
    if plan.get("schema") != "mv.sectionplan.v1":
        raise PlannerError("schema must be mv.sectionplan.v1")
    if plan.get("aspect") not in {aspect, "9:16", "16:9"}:
        # allow model echo; still must be valid enum
        if plan.get("aspect") not in {"9:16", "16:9"}:
            raise PlannerError("aspect invalid")
    sections = plan.get("sections")
    if not isinstance(sections, list) or not sections:
        raise PlannerError("sections must be non-empty list")

    notes: list[str] = list(plan.get("notes") or [])
    if isinstance(notes, str):
        notes = [notes]

    out_sections: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for i, raw in enumerate(sections):
        if not isinstance(raw, dict):
            raise PlannerError(f"sections[{i}] must be object")
        sid = str(raw.get("section_id") or raw.get("id") or "")
        if not sid:
            raise PlannerError(f"sections[{i}].section_id missing")
        if known_section_ids and sid not in known_section_ids:
            raise PlannerError(f"unknown section_id: {sid}")
        if sid in seen_ids:
            raise PlannerError(f"duplicate section_id: {sid}")
        seen_ids.add(sid)

        density = str(raw.get("density") or "").lower()
        if density not in DENSITY_ENUM:
            raise PlannerError(f"sections[{i}].density must be sparse|medium|dense")

        ids = raw.get("image_ids")
        if not isinstance(ids, list) or not ids:
            raise PlannerError(f"sections[{i}].image_ids empty — no silent fabricate")
        clean_ids: list[str] = []
        for iid in ids:
            iid_s = str(iid)
            if iid_s not in known_image_ids:
                raise PlannerError(f"invented image_id: {iid_s}")
            clean_ids.append(iid_s)

        title = raw.get("title")
        title_out = None
        if title is not None:
            if not isinstance(title, dict):
                raise PlannerError(f"sections[{i}].title must be object")
            text = str(title.get("text") or "").strip()
            if text:
                style = title.get("style") or {"color": "#FFFFFF", "bg": "#000000", "bg_pad": 12}
                if not isinstance(style, dict):
                    raise PlannerError(f"sections[{i}].title.style must be object")
                title_out = {
                    "text": text,
                    "style": {
                        "color": str(style.get("color") or "#FFFFFF"),
                        "bg": str(style.get("bg") or "#000000"),
                        "bg_pad": int(style.get("bg_pad") or 12),
                    },
                }

        # HARD: forbid timing fields from LLM
        for banned in ("start_s", "end_s", "clips", "ffmpeg", "timestamps"):
            if banned in raw:
                raise PlannerError(f"sections[{i}] must not include {banned}")

        out_sections.append({
            "section_id": sid,
            "image_ids": clean_ids,
            "density": density,
            "title": title_out,
        })

    # every known section should appear; missing → HARD (no silent fill)
    if known_section_ids:
        missing = sorted(known_section_ids - seen_ids)
        if missing:
            raise PlannerError(f"missing sections in plan: {missing}")

    return {
        "schema": "mv.sectionplan.v1",
        "aspect": plan.get("aspect") or aspect,
        "sections": out_sections,
        "notes": notes,
    }


def parse_llm_json(text: str) -> dict[str, Any]:
    cleaned = strip_code_fences(text)
    try:
        data = json.loads(cleaned, parse_constant=_reject_constants)
    except json.JSONDecodeError as exc:
        raise PlannerError(f"malformed JSON from model: {exc}") from exc
    if not isinstance(data, dict):
        raise PlannerError("model JSON root must be object")
    return data


def _chat_completion(
    *,
    api_key: str,
    model: str,
    user_payload: dict[str, Any],
    extra_system: str = "",
    timeout_s: float = DEFAULT_TIMEOUT_S,
    transport: HttpTransport | None = None,
) -> tuple[str, dict[str, Any]]:
    transport = transport or default_http_transport
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + (("\n" + extra_system) if extra_system else "")},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]
    body_obj = {
        "model": model,
        "temperature": TEMPERATURE,
        "messages": messages,
    }
    body = json.dumps(body_obj).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://local.music-video.editlab",
        "X-Title": "EditLab-MusicVideo",
    }
    t0 = time.perf_counter()
    status, raw = transport(OPENROUTER_URL, headers, body, timeout_s)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    meta = {"http_status": status, "latency_ms": round(latency_ms, 2)}
    if status < 200 or status >= 300:
        raise PlannerError(f"OpenRouter HTTP {status}: {raw[:400]}")
    try:
        envelope = json.loads(raw, parse_constant=_reject_constants)
    except json.JSONDecodeError as exc:
        raise PlannerError(f"OpenRouter envelope not JSON: {exc}") from exc
    try:
        content = envelope["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise PlannerError("OpenRouter response missing choices[0].message.content") from exc
    if not isinstance(content, str) or not content.strip():
        raise PlannerError("OpenRouter empty content")
    return content, meta


def plan_sections(
    *,
    style_instruction: str,
    aspect: str,
    audio: dict[str, Any],
    images: list[dict[str, Any]],
    api_key: str | None = None,
    model: str | None = None,
    fallback_models: list[str] | None = None,
    transport: HttpTransport | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    load_env: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Bounded attempt policy → (section_plan, attempts_log). Never fabricates a plan."""
    if api_key is None or model is None:
        k, m, fb = require_openrouter_config(load_env=load_env)
        api_key = api_key or k
        model = model or m
        fallback_models = fallback_models if fallback_models is not None else fb
    else:
        fallback_models = list(fallback_models or [])

    user_payload = build_planner_input(
        style_instruction=style_instruction,
        aspect=aspect,
        audio=audio,
        images=images,
    )
    known_ids = {im["id"] for im in user_payload["images"]}
    known_sections = {str(s["id"]) for s in user_payload["audio"]["sections"]}

    attempts: list[dict[str, Any]] = []
    # Attempt order: (1) primary (2) primary+HARD errors (3+) fallbacks
    schedule: list[tuple[str, str, str]] = [
        (model, "primary_initial", ""),
        (model, "primary_retry_hard", ""),  # extra filled after first fail
    ]
    for fb in fallback_models:
        schedule.append((fb, "fallback", ""))

    last_hard = ""
    for idx, (mdl, kind, _) in enumerate(schedule, start=1):
        extra = ""
        if kind == "primary_retry_hard":
            if not last_hard:
                # first attempt succeeded path won't reach; if first failed without hard schema, still retry once
                extra = "Previous response failed. Return corrected mv.sectionplan.v1 JSON only."
            else:
                extra = f"Previous HARD validation errors:\n{last_hard}\nReturn corrected mv.sectionplan.v1 JSON only."
        # skip duplicate primary_retry if somehow no prior failure — handled by loop control
        record: dict[str, Any] = {
            "attempt": idx,
            "model": mdl,
            "kind": kind,
            "http_status": None,
            "latency_ms": None,
            "schema_ok": False,
            "failure_reason": None,
        }
        try:
            content, meta = _chat_completion(
                api_key=api_key,
                model=mdl,
                user_payload=user_payload,
                extra_system=extra,
                timeout_s=timeout_s,
                transport=transport,
            )
            record["http_status"] = meta["http_status"]
            record["latency_ms"] = meta["latency_ms"]
            parsed = parse_llm_json(content)
            validated = validate_section_plan(
                parsed,
                known_image_ids=known_ids,
                known_section_ids=known_sections,
                aspect=aspect,
            )
            record["schema_ok"] = True
            attempts.append(record)
            return validated, attempts
        except PlannerError as exc:
            record["failure_reason"] = str(exc)
            if "HTTP" in str(exc):
                # try extract status
                pass
            attempts.append(record)
            last_hard = str(exc)
            continue

    raise PlannerError(
        "all OpenRouter attempts failed; refusing to fabricate a creative plan. "
        f"attempts={len(attempts)}"
    )


def write_section_plan(
    job_dir,
    section_plan: dict[str, Any],
    attempts: list[dict[str, Any]],
) -> None:
    from pathlib import Path
    from music_video.paths import planner_attempts_path, section_plan_path

    job_dir = Path(job_dir)
    section_plan_path(job_dir).write_text(json.dumps(section_plan, indent=2), encoding="utf-8")
    planner_attempts_path(job_dir).write_text(json.dumps({"attempts": attempts}, indent=2), encoding="utf-8")
