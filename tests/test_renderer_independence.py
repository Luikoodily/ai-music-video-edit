"""Guard: render stays free of CLIP/OpenRouter/expander; expander stays free of render."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "music_video"
RENDER_SRC = ROOT / "render_ffmpeg.py"
EXPAND_SRC = ROOT / "timeline_expand.py"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
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
    return found


def test_render_ffmpeg_has_no_planner_or_image_imports():
    if not RENDER_SRC.is_file():
        raise RuntimeError(f"missing render_ffmpeg.py: {RENDER_SRC}")
    banned = {
        "planner_openrouter",
        "image_analyze",
        "timeline_expand",
        "pipeline",
        "music_video.planner_openrouter",
        "music_video.image_analyze",
        "music_video.timeline_expand",
        "music_video.pipeline",
        "open_clip",
        "torch",
    }
    hits = sorted(_imports(RENDER_SRC) & banned)
    assert not hits, f"render_ffmpeg imports banned modules: {hits}"


def test_timeline_expand_has_no_render_or_planner_imports():
    if not EXPAND_SRC.is_file():
        raise RuntimeError(f"missing timeline_expand.py: {EXPAND_SRC}")
    banned = {
        "render_ffmpeg",
        "planner_openrouter",
        "music_video.render_ffmpeg",
        "music_video.planner_openrouter",
        "open_clip",
        "torch",
    }
    hits = sorted(_imports(EXPAND_SRC) & banned)
    assert not hits, f"timeline_expand imports banned modules: {hits}"


def test_render_module_loads_without_m2_m4():
    import music_video.render_ffmpeg as rf

    assert callable(rf.render_job)
    assert "planner_openrouter" not in rf.__dict__
    assert "image_analyze" not in rf.__dict__
    assert "timeline_expand" not in rf.__dict__
