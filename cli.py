#!/usr/bin/env python3
"""Music-video MVP CLI (analyze-audio | analyze-images | validate | render)."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from music_video.audio_analyze import write_audio_analysis  # noqa: E402
from music_video.errors import (  # noqa: E402
    AudioAnalysisError,
    HardValidationError,
    ImageAnalysisError,
    RenderError,
)
from music_video.image_analyze import write_image_analysis  # noqa: E402
from music_video.paths import (  # noqa: E402
    audio_analysis_path,
    ensure_job,
    image_analysis_path,
    plan_path,
    validated_plan_path,
)
from music_video.render_ffmpeg import render_job  # noqa: E402
from music_video.validate_plan import validate_plan_files  # noqa: E402


def cmd_analyze_audio(args: argparse.Namespace) -> int:
    job = ensure_job(args.job)
    src = Path(args.song) if args.song else None
    if src is None:
        hits = list(job.glob("song.*"))
        if not hits:
            print("missing --song and no song.* in job", file=sys.stderr)
            return 2
        src = hits[0]
    else:
        dest = job / f"song{src.suffix.lower()}"
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        src = dest
    try:
        data = write_audio_analysis(src, audio_analysis_path(job), work_dir=job / "work")
    except AudioAnalysisError as exc:
        print(f"AUDIO ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, "duration_s": data["duration_s"], "tempo_bpm": data["tempo_bpm"], "n_beats": len(data["beats_s"])}))
    return 0


def cmd_analyze_images(args: argparse.Namespace) -> int:
    job = ensure_job(args.job)
    images = job / "images"
    try:
        data = write_image_analysis(
            images,
            image_analysis_path(job),
            audio_analysis_path=audio_analysis_path(job),
            style_words=args.style,
        )
    except ImageAnalysisError as exc:
        print(f"IMAGE ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "ok": True,
        "n_images": len(data["images"]),
        "n_duplicates": len(data["duplicates"]),
        "model_id": data["model_id"],
        "embedding_dim": data["embedding_dim"],
        "out": str(image_analysis_path(job)),
    }))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    job = ensure_job(args.job)
    try:
        validate_plan_files(
            plan_path(job),
            audio_analysis_path(job),
            validated_plan_path(job),
            images_dir=job / "images",
        )
    except HardValidationError as exc:
        print(f"HARD VALIDATION: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, "validated": str(validated_plan_path(job))}))
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    job = ensure_job(args.job)
    vp = validated_plan_path(job)
    if not vp.is_file():
        # auto-validate if plan + audio analysis exist
        if plan_path(job).is_file() and audio_analysis_path(job).is_file():
            try:
                validate_plan_files(plan_path(job), audio_analysis_path(job), vp, images_dir=job / "images")
            except HardValidationError as exc:
                print(f"HARD VALIDATION: {exc}", file=sys.stderr)
                return 1
        else:
            print("missing plan.validated.json (and cannot auto-build)", file=sys.stderr)
            return 2
    try:
        out = render_job(job)
    except RenderError as exc:
        print(f"RENDER ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, "final": str(out), "bytes": out.stat().st_size}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze-audio")
    a.add_argument("--job", required=True)
    a.add_argument("--song", default=None)
    a.set_defaults(func=cmd_analyze_audio)

    ai = sub.add_parser("analyze-images")
    ai.add_argument("--job", required=True)
    ai.add_argument("--style", default=None, help="optional style words for section prompts")
    ai.set_defaults(func=cmd_analyze_images)

    v = sub.add_parser("validate")
    v.add_argument("--job", required=True)
    v.set_defaults(func=cmd_validate)

    r = sub.add_parser("render")
    r.add_argument("--job", required=True)
    r.set_defaults(func=cmd_render)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
