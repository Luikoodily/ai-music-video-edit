"""M6: pipeline orchestration for `cli.py run` (full + resume)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from music_video.audio_analyze import write_audio_analysis
from music_video.env_load import load_dotenv
from music_video.errors import (
    AudioAnalysisError,
    HardValidationError,
    ImageAnalysisError,
    MusicVideoError,
    PlannerError,
    RenderError,
    TimelineExpandError,
)
from music_video.image_analyze import clip_weights_ready, write_image_analysis
from music_video.paths import (
    audio_analysis_path,
    ensure_job,
    final_mp4_path,
    image_analysis_path,
    plan_path,
    section_plan_path,
    validated_plan_path,
)
from music_video.planner_openrouter import plan_sections, write_section_plan
from music_video.render_ffmpeg import render_job, verify_final
from music_video.timeline_expand import expand_job_files
from music_video.validate_plan import validate_plan_files

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
PREVIEW_SIZE = {"9:16": (540, 960), "16:9": (960, 540)}
STAGE_ORDER = (
    "analyze-audio",
    "analyze-images",
    "plan",
    "expand",
    "validate",
    "render",
)


class PipelineError(MusicVideoError):
    """User-facing orchestration failure."""


@dataclass
class RunConfig:
    job_id: str
    style: str = "cinematic music video"
    aspect: str = "9:16"
    out: str = "final"  # final | preview
    song: Path | None = None
    images_dir: Path | None = None


def run_log_path(job: Path) -> Path:
    return job / "run.log"


def run_meta_path(job: Path) -> Path:
    return job / "run_meta.json"


def plan_hash_path(job: Path, out: str) -> Path:
    return job / f"{out}.planhash"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def append_log(job: Path, msg: str) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    with run_log_path(job).open("a", encoding="utf-8") as f:
        f.write(line)


def _load_meta(job: Path) -> dict[str, Any]:
    p = run_meta_path(job)
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def _save_meta(job: Path, meta: dict[str, Any]) -> None:
    run_meta_path(job).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def prepare_job(cfg: RunConfig) -> Path:
    job = ensure_job(cfg.job_id)
    run_log_path(job).touch(exist_ok=True)
    meta = _load_meta(job)
    meta.update({
        "job_id": cfg.job_id,
        "style": cfg.style,
        "aspect": cfg.aspect,
        "out": cfg.out,
    })
    _save_meta(job, meta)

    if cfg.song is not None:
        src = Path(cfg.song)
        if not src.is_file():
            raise PipelineError(f"song not found: {src}")
        dest = job / f"song{src.suffix.lower()}"
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)  # copy — never move original
        append_log(job, f"prepare: copied song -> {dest.name}")

    if cfg.images_dir is not None:
        src_dir = Path(cfg.images_dir)
        if not src_dir.is_dir():
            raise PipelineError(f"images dir not found: {src_dir}")
        dest_dir = job / "images"
        dest_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for p in sorted(src_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                target = dest_dir / p.name
                if p.resolve() != target.resolve():
                    shutil.copy2(p, target)
                n += 1
        if n == 0:
            raise PipelineError(f"no images found in {src_dir}")
        append_log(job, f"prepare: copied {n} images -> images/")
    return job


def _artifact_ok_audio(job: Path) -> bool:
    p = audio_analysis_path(job)
    if not p.is_file():
        return False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data.get("schema") == "mv.audio.v1" and float(data.get("duration_s", 0)) > 0 and bool(data.get("beats_s"))
    except Exception:  # noqa: BLE001
        return False


def _artifact_ok_images(job: Path) -> bool:
    p = image_analysis_path(job)
    if not p.is_file():
        return False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data.get("schema") == "mv.images.v1" and isinstance(data.get("images"), list) and len(data["images"]) > 0
    except Exception:  # noqa: BLE001
        return False


def _artifact_ok_section_plan(job: Path) -> bool:
    p = section_plan_path(job)
    if not p.is_file():
        return False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data.get("schema") == "mv.sectionplan.v1" and bool(data.get("sections"))
    except Exception:  # noqa: BLE001
        return False


def _artifact_ok_validated(job: Path) -> bool:
    p = validated_plan_path(job)
    if not p.is_file():
        return False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data.get("schema") == "mv.plan.v1" and bool(data.get("clips"))
    except Exception:  # noqa: BLE001
        return False


def _out_mp4(job: Path, out: str) -> Path:
    return job / ("preview.mp4" if out == "preview" else "final.mp4")


def _artifact_ok_render(job: Path, out: str) -> bool:
    p = _out_mp4(job, out)
    return p.is_file() and p.stat().st_size > 0


def _timed(job: Path, stage: str, fn: Callable[[], Any]) -> Any:
    append_log(job, f"STAGE START {stage}")
    t0 = time.perf_counter()
    try:
        result = fn()
    except Exception as exc:
        dt = time.perf_counter() - t0
        append_log(job, f"STAGE FAIL {stage} wall_s={dt:.3f} error={type(exc).__name__}: {exc}")
        raise
    dt = time.perf_counter() - t0
    append_log(job, f"STAGE END {stage} wall_s={dt:.3f}")
    return result


def _mock_plan_enabled() -> bool:
    return os.environ.get("MV_MOCK_PLANNER", "").strip() == "1"


def _build_mock_section_plan(audio: dict[str, Any], images: list[dict[str, Any]], aspect: str) -> dict[str, Any]:
    usable = [im for im in images if not im.get("unused")]
    if not usable:
        usable = list(images)
    if not usable:
        raise PlannerError("mock planner: no images")
    sections_out = []
    for i, s in enumerate(audio.get("sections") or []):
        sid = str(s.get("id") or f"s{i}")
        label = str(s.get("label") or "").lower()
        if label in {"drop", "chorus", "build"}:
            density = "dense" if label != "build" else "medium"
            if label == "drop":
                density = "dense"
        elif label in {"intro", "outro"}:
            density = "sparse"
        else:
            density = "medium"
        # rotate pool deterministically
        ids = []
        for k in range(min(3, len(usable))):
            ids.append(usable[(i + k) % len(usable)]["id"])
        title = None
        if i == 0:
            title = {
                "text": "TITLE",
                "style": {"color": "#FFFFFF", "bg": "#000000", "bg_pad": 12},
            }
        sections_out.append({
            "section_id": sid,
            "image_ids": ids,
            "density": density,
            "title": title,
        })
    return {
        "schema": "mv.sectionplan.v1",
        "aspect": aspect,
        "sections": sections_out,
        "notes": ["MV_MOCK_PLANNER=1 deterministic mock (test-only)"],
    }


def stage_analyze_audio(job: Path) -> None:
    hits = list(job.glob("song.*"))
    if not hits:
        raise PipelineError("missing song.* in job — pass --song on full run")
    write_audio_analysis(hits[0], audio_analysis_path(job), work_dir=job / "work")
    if not _artifact_ok_audio(job):
        raise PipelineError("audio_analysis.json invalid after analyze-audio")


def stage_analyze_images(job: Path, style: str) -> None:
    if not clip_weights_ready():
        raise PipelineError(
            "CLIP weights cache missing under models/open-clip. "
            "Run: python scripts/bootstrap_clip.py"
        )
    imgs = job / "images"
    if not imgs.is_dir() or not any(imgs.iterdir()):
        raise PipelineError("missing images/ — pass --images on full run")
    write_image_analysis(
        imgs,
        image_analysis_path(job),
        audio_analysis_path=audio_analysis_path(job),
        style_words=style,
    )
    if not _artifact_ok_images(job):
        raise PipelineError("image_analysis.json invalid after analyze-images")


def stage_plan(job: Path, style: str, aspect: str) -> None:
    load_dotenv()
    audio = json.loads(audio_analysis_path(job).read_text(encoding="utf-8"))
    images = json.loads(image_analysis_path(job).read_text(encoding="utf-8")).get("images") or []
    if _mock_plan_enabled():
        append_log(job, "plan: MV_MOCK_PLANNER=1 — using mocked transport (test-only)")
        mock = _build_mock_section_plan(audio, images, aspect)

        def transport(url, headers, body, timeout_s):  # noqa: ANN001
            envelope = {"choices": [{"message": {"content": json.dumps(mock)}}]}
            return 200, json.dumps(envelope)

        section_plan, attempts = plan_sections(
            style_instruction=style,
            aspect=aspect,
            audio=audio,
            images=images,
            api_key="mock-key",
            model="mock/local",
            fallback_models=[],
            transport=transport,
            load_env=False,
        )
    else:
        section_plan, attempts = plan_sections(
            style_instruction=style,
            aspect=aspect,
            audio=audio,
            images=images,
            load_env=True,
        )
    write_section_plan(job, section_plan, attempts)
    if not _artifact_ok_section_plan(job):
        raise PipelineError("section_plan.json invalid after plan")


def stage_expand(job: Path) -> None:
    expand_job_files(job)
    if not _artifact_ok_validated(job):
        raise PipelineError("plan.validated.json invalid after expand")


def stage_validate(job: Path) -> None:
    # expand already validates; re-run M3 for explicit stage + refresh
    if not plan_path(job).is_file():
        # expand writes both; if only validated exists, copy
        if validated_plan_path(job).is_file():
            plan_path(job).write_text(validated_plan_path(job).read_text(encoding="utf-8"), encoding="utf-8")
        else:
            raise PipelineError("missing plan.json for validate")
    validate_plan_files(
        plan_path(job),
        audio_analysis_path(job),
        validated_plan_path(job),
        images_dir=job / "images",
    )
    if not _artifact_ok_validated(job):
        raise PipelineError("plan.validated.json invalid after validate")


def stage_render(job: Path, out: str, aspect: str) -> Path:
    vp = validated_plan_path(job)
    if not vp.is_file():
        raise PipelineError("missing plan.validated.json before render")
    plan_hash = file_sha256(vp)
    out_path = _out_mp4(job, out)
    stamp = plan_hash_path(job, out)
    if out_path.is_file() and stamp.is_file() and stamp.read_text(encoding="utf-8").strip() == plan_hash:
        append_log(
            job,
            f"render: SKIP {out_path.name} — plan.validated.json unchanged (sha256={plan_hash[:12]}…)",
        )
        return out_path

    width = height = None
    if out == "preview":
        width, height = PREVIEW_SIZE.get(aspect, (540, 960))

    result = render_job(
        job,
        plan_validated=vp,
        out_name=out_path.name,
        width=width,
        height=height,
    )
    # D5-ish check
    plan = json.loads(vp.read_text(encoding="utf-8"))
    expect_w = width or int(plan["width"])
    expect_h = height or int(plan["height"])
    verify_final(result, expect_w=expect_w, expect_h=expect_h, expect_dur=float(plan["duration_s"]))
    stamp.write_text(plan_hash, encoding="utf-8")
    return result


def needed_stages(job: Path, out: str) -> list[str]:
    """Resume: return stages still missing, in contract order."""
    needed: list[str] = []
    if not _artifact_ok_audio(job):
        needed.append("analyze-audio")
    if not _artifact_ok_images(job):
        needed.append("analyze-images")
    if not _artifact_ok_section_plan(job):
        needed.append("plan")
    if not _artifact_ok_validated(job):
        # expand produces validated; validate refreshes it
        needed.append("expand")
        needed.append("validate")
    else:
        # validated exists — still ensure plan.json present for validate stage skip logic
        if not plan_path(job).is_file():
            needed.append("validate")
    # render decision: missing output OR plan hash mismatch
    if not _artifact_ok_render(job, out):
        needed.append("render")
    else:
        vp = validated_plan_path(job)
        stamp = plan_hash_path(job, out)
        if vp.is_file():
            cur = file_sha256(vp)
            if not stamp.is_file() or stamp.read_text(encoding="utf-8").strip() != cur:
                needed.append("render")
        else:
            needed.append("render")
    return needed


def run_pipeline(cfg: RunConfig) -> dict[str, Any]:
    """Mode A: song+images provided → prepare + all stages. Mode B: job-only → resume missing."""
    mode_a = cfg.song is not None and cfg.images_dir is not None
    job = prepare_job(cfg) if mode_a else ensure_job(cfg.job_id)
    if not run_log_path(job).is_file():
        run_log_path(job).touch()

    meta = _load_meta(job)
    style = cfg.style or meta.get("style") or "cinematic music video"
    aspect = cfg.aspect or meta.get("aspect") or "9:16"
    out = cfg.out or meta.get("out") or "final"
    meta.update({"style": style, "aspect": aspect, "out": out, "job_id": cfg.job_id})
    _save_meta(job, meta)

    append_log(
        job,
        f"RUN START job={cfg.job_id} mode={'A-full' if mode_a else 'B-resume'} "
        f"out={out} aspect={aspect} mock_planner={int(_mock_plan_enabled())}",
    )

    stages = list(STAGE_ORDER) if mode_a else needed_stages(job, out)

    if not stages:
        append_log(job, "RUN SKIP — all artifacts present; render plan hash matches")
        out_path = _out_mp4(job, out)
        return {"ok": True, "job": str(job), "stages": [], "out": str(out_path), "skipped_all": True}

    append_log(job, f"RUN STAGES: {', '.join(stages)}")

    try:
        for stage in stages:
            if stage == "analyze-audio":
                _timed(job, stage, lambda: stage_analyze_audio(job))
            elif stage == "analyze-images":
                _timed(job, stage, lambda: stage_analyze_images(job, style))
            elif stage == "plan":
                _timed(job, stage, lambda: stage_plan(job, style, aspect))
            elif stage == "expand":
                _timed(job, stage, lambda: stage_expand(job))
            elif stage == "validate":
                _timed(job, stage, lambda: stage_validate(job))
            elif stage == "render":
                _timed(job, stage, lambda: stage_render(job, out, aspect))
            else:
                raise PipelineError(f"unknown stage: {stage}")
            if stage == "analyze-audio" and not _artifact_ok_audio(job):
                raise PipelineError("analyze-audio produced invalid artifact")
            if stage == "analyze-images" and not _artifact_ok_images(job):
                raise PipelineError("analyze-images produced invalid artifact")
            if stage == "plan" and not _artifact_ok_section_plan(job):
                raise PipelineError("plan produced invalid artifact")
            if stage in {"expand", "validate"} and not _artifact_ok_validated(job):
                raise PipelineError(f"{stage} produced invalid artifact")
    except (
        AudioAnalysisError,
        ImageAnalysisError,
        PlannerError,
        TimelineExpandError,
        HardValidationError,
        RenderError,
        PipelineError,
    ) as exc:
        append_log(job, f"RUN FAIL: {exc}")
        raise PipelineError(str(exc)) from exc

    out_path = _out_mp4(job, out)
    append_log(job, f"RUN OK out={out_path}")
    return {
        "ok": True,
        "job": str(job),
        "stages": stages,
        "out": str(out_path),
        "bytes": out_path.stat().st_size if out_path.is_file() else 0,
    }
