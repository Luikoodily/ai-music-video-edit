"""Job folder layout under music-video/jobs/<job_id>/."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JOBS = ROOT / "jobs"

ASPECT_SIZE = {
    "9:16": (1080, 1920),
    "16:9": (1920, 1080),
}


def job_dir(job_id: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in job_id.strip())
    if not safe:
        raise ValueError("empty job id")
    return JOBS / safe


def ensure_job(job_id: str) -> Path:
    d = job_dir(job_id)
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "work" / "clips").mkdir(parents=True, exist_ok=True)
    return d


def audio_analysis_path(job: Path) -> Path:
    return job / "audio_analysis.json"


def plan_path(job: Path) -> Path:
    return job / "plan.json"


def validated_plan_path(job: Path) -> Path:
    return job / "plan.validated.json"


def final_mp4_path(job: Path) -> Path:
    return job / "final.mp4"


def work_dir(job: Path) -> Path:
    return job / "work"
