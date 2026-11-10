# Edit Lab — Music Video MVP

Local song + images → beat-synced hard-cut video. LLM is creative director only (section plan). Timing + FFmpeg stay local.

## Quickstart

```powershell
cd music-video
python -m pip install -r requirements-mvp.txt
copy .env.example .env   # set OPENROUTER_API_KEY + OPENROUTER_MODEL
python scripts/bootstrap_clip.py   # once — downloads CLIP weights to models/open-clip/
```

One-command full run:

```powershell
python cli.py run --song path\to\song.mp3 --images path\to\imgs --style "cinematic marvel" --aspect 9:16 --job demo1
```

Resume (skips finished stages; skips re-render if `plan.validated.json` hash unchanged):

```powershell
python cli.py run --job demo1
```

Preview encode (540×960 for 9:16):

```powershell
python cli.py run --song song.wav --images imgs --style "test" --aspect 9:16 --job demo1 --out preview
```

### Test-only mock planner

```powershell
$env:MV_MOCK_PLANNER="1"   # never default; offline e2e only
python cli.py run --song ... --images ... --job mock1
```

Live OpenRouter requires real `.env` and **no** `MV_MOCK_PLANNER`.

## Stage map

| Stage | Artifact | Notes |
|-------|----------|--------|
| analyze-audio | `audio_analysis.json` | librosa; ffmpeg decode first |
| analyze-images | `image_analysis.json` | CLIP CPU; needs bootstrap cache |
| plan | `section_plan.json` | OpenRouter `mv.sectionplan.v1` |
| expand | `plan.json` + `plan.validated.json` | deterministic beats → clips |
| validate | `plan.validated.json` | M3 HARD/SOFT |
| render | `final.mp4` / `preview.mp4` | hard cuts only |

## Job folder

```
jobs/<id>/
  song.<ext>              # copy — originals never moved
  images/
  audio_analysis.json
  image_analysis.json
  section_plan.json
  planner_attempts.json
  plan.json
  plan.validated.json
  final.mp4 | preview.mp4
  final.planhash          # sha256 of plan used for last render
  run_meta.json
  run.log
  work/…
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ffmpeg/ffprobe not available` | Install FFmpeg; ensure on PATH |
| `CLIP weights cache missing` | `python scripts/bootstrap_clip.py` |
| `OPENROUTER_API_KEY missing` | Set in env or `music-video/.env` (from `.env.example`) |
| `OPENROUTER_MODEL missing` | Set model explicitly — never assumed |
| Silent / short audio | Need ≥10s non-silent input |
| Spaced Windows paths | Supported; pass quoted paths |
| Want offline CI | `MV_MOCK_PLANNER=1` (test-only) |

## Tests

```powershell
$env:HF_HUB_OFFLINE="1"
python -m pytest -rs
```

Live e2e: `MV_E2E=1` + `OPENROUTER_API_KEY`.

## Models / licenses

OpenCLIP `ViT-B-32` / `openai` weights are downloaded by `scripts/bootstrap_clip.py` into local `models/` (gitignored). Those weights have their own upstream license terms — review before public redistribution. This repo does not vend weight files.
