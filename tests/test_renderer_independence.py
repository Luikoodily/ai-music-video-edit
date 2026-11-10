"""Guard: render_ffmpeg must not import planner_openrouter or image_analyze."""

from __future__ import annotations

import ast
from pathlib import Path

RENDER_SRC = Path(__file__).resolve().parents[1] / "src" / "music_video" / "render_ffmpeg.py"


def test_render_ffmpeg_has_no_planner_or_image_imports():
    if not RENDER_SRC.is_file():
        raise RuntimeError(f"missing render_ffmpeg.py: {RENDER_SRC}")
    tree = ast.parse(RENDER_SRC.read_text(encoding="utf-8"))
    banned = {
        "planner_openrouter",
        "image_analyze",
        "music_video.planner_openrouter",
        "music_video.image_analyze",
        "open_clip",
        "torch",
    }
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            found.add(mod)
            for alias in node.names:
                found.add(f"{mod}.{alias.name}" if mod else alias.name)
    hits = sorted(found & banned)
    assert not hits, f"render_ffmpeg imports banned modules: {hits}"


def test_render_module_loads_without_m2_m4():
    import music_video.render_ffmpeg as rf

    assert callable(rf.render_job)
    assert "planner_openrouter" not in rf.__dict__
    assert "image_analyze" not in rf.__dict__
