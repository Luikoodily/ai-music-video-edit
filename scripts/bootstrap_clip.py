#!/usr/bin/env python3
"""Download OpenCLIP ViT-B-32 (openai) weights ONCE into local models/ cache.

Run manually outside pytest. Uses network. Reports cache path on success.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from music_video.image_analyze import (  # noqa: E402
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    clip_cache_dir,
    resolve_weights_path,
)


def main() -> int:
    cache = clip_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    print(f"cache_dir={cache.resolve()}")
    try:
        import open_clip
        import torch
    except ImportError as exc:
        print(f"missing deps: {exc}\npip install torch open-clip-torch", file=sys.stderr)
        return 1

    print(f"downloading {CLIP_MODEL_NAME}/{CLIP_PRETRAINED} (CPU)…")
    model, _, _ = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME,
        pretrained=CLIP_PRETRAINED,
        cache_dir=str(cache),
    )
    model.eval()
    _ = torch.device("cpu")
    weights = resolve_weights_path(cache)
    print(f"model_id=open_clip:{CLIP_MODEL_NAME}:{CLIP_PRETRAINED}")
    print(f"weights_path={weights}")
    print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
