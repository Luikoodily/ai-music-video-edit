"""Build jobs/prove1 from golden plan + synthetic song/images (no network)."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from helpers import write_click_track_wav, write_solid_png  # noqa: E402
from music_video.audio_analyze import write_audio_analysis  # noqa: E402
from music_video.paths import ensure_job  # noqa: E402
from music_video.validate_plan import validate_plan_files  # noqa: E402


def main() -> int:
    job = ensure_job("prove1")
    song = write_click_track_wav(job / "song.wav", bpm=120.0, duration_s=12.0)
    imgs = job / "images"
    write_solid_png(imgs / "01.png", rgb=(220, 40, 40), size=(320, 480))
    write_solid_png(imgs / "02.png", rgb=(40, 180, 60), size=(320, 480))
    write_solid_png(imgs / "03.png", rgb=(40, 80, 220), size=(320, 480))

    golden = ROOT / "tests" / "fixtures" / "golden_plan.json"
    plan = json.loads(golden.read_text(encoding="utf-8"))
    plan["audio_path"] = "song.wav"
    (job / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    write_audio_analysis(song, job / "audio_analysis.json", work_dir=job / "work")
    validate_plan_files(
        job / "plan.json",
        job / "audio_analysis.json",
        job / "plan.validated.json",
        images_dir=imgs,
    )
    print(f"prove1 ready at {job}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
