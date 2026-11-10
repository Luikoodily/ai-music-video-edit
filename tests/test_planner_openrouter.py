"""M4: mocked OpenRouter transport — zero network in ordinary tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from music_video.errors import PlannerError
from music_video.planner_openrouter import (
    build_planner_input,
    parse_llm_json,
    plan_sections,
    require_openrouter_config,
    strip_code_fences,
    validate_section_plan,
)


def _audio() -> dict:
    return {
        "duration_s": 12.0,
        "tempo_bpm": 120.0,
        "beats_s": [i * 0.5 for i in range(25)],
        "sections": [
            {"id": "s0", "label": "intro", "start_s": 0.0, "end_s": 4.0},
            {"id": "s1", "label": "drop", "start_s": 4.0, "end_s": 8.0},
            {"id": "s2", "label": "outro", "start_s": 8.0, "end_s": 12.0},
        ],
    }


def _images() -> list[dict]:
    return [
        {"id": "img_01", "labels": ["hero"], "best_sections": ["s1"], "unused": False},
        {"id": "img_02", "labels": ["detail"], "best_sections": ["s0"], "unused": False},
        {"id": "img_03", "labels": ["graphic"], "best_sections": ["s2"], "unused": False},
    ]


def _valid_section_plan_text() -> str:
    plan = {
        "schema": "mv.sectionplan.v1",
        "aspect": "9:16",
        "sections": [
            {"section_id": "s0", "image_ids": ["img_02", "img_01"], "density": "sparse",
             "title": {"text": "INTRO", "style": {"color": "#FFFFFF", "bg": "#000000", "bg_pad": 12}}},
            {"section_id": "s1", "image_ids": ["img_01", "img_03"], "density": "dense"},
            {"section_id": "s2", "image_ids": ["img_03"], "density": "sparse"},
        ],
        "notes": ["mocked"],
    }
    return json.dumps(plan)


def _transport_ok(content: str):
    def _t(url, headers, body, timeout_s):
        assert "Authorization" in headers
        assert "Bearer " in headers["Authorization"]
        assert timeout_s == 180.0 or timeout_s > 0
        envelope = {"choices": [{"message": {"content": content}}]}
        return 200, json.dumps(envelope)
    return _t


def test_build_planner_input_downsamples_beats():
    audio = _audio()
    audio["beats_s"] = list(range(1000))
    payload = build_planner_input(
        style_instruction="cinematic",
        aspect="9:16",
        audio=audio,
        images=_images(),
    )
    assert payload["style_instruction"] == "cinematic"
    assert len(payload["audio"]["beats_s_sample"]) <= 400
    assert payload["rules"]["prefer_cut_on_beats"] is True
    assert all("id" in im for im in payload["images"])


def test_strip_fences_and_parse():
    raw = "```json\n" + _valid_section_plan_text() + "\n```"
    data = parse_llm_json(raw)
    assert data["schema"] == "mv.sectionplan.v1"


def test_validate_rejects_invented_image_and_timestamps():
    plan = json.loads(_valid_section_plan_text())
    plan["sections"][0]["image_ids"] = ["img_99"]
    with pytest.raises(PlannerError, match="invented image_id"):
        validate_section_plan(
            plan,
            known_image_ids={"img_01", "img_02", "img_03"},
            known_section_ids={"s0", "s1", "s2"},
            aspect="9:16",
        )
    plan = json.loads(_valid_section_plan_text())
    plan["sections"][0]["start_s"] = 1.0
    with pytest.raises(PlannerError, match="start_s"):
        validate_section_plan(
            plan,
            known_image_ids={"img_01", "img_02", "img_03"},
            known_section_ids={"s0", "s1", "s2"},
            aspect="9:16",
        )


def test_plan_sections_mocked_success():
    plan, attempts = plan_sections(
        style_instruction="test",
        aspect="9:16",
        audio=_audio(),
        images=_images(),
        api_key="test-key-not-real",
        model="test/model",
        fallback_models=[],
        transport=_transport_ok(_valid_section_plan_text()),
        load_env=False,
    )
    assert plan["schema"] == "mv.sectionplan.v1"
    assert len(plan["sections"]) == 3
    assert attempts[0]["schema_ok"] is True
    assert attempts[0]["model"] == "test/model"


def test_retry_then_fallback_mocked():
    calls = {"n": 0}

    def transport(url, headers, body, timeout_s):
        calls["n"] += 1
        if calls["n"] == 1:
            return 200, json.dumps({"choices": [{"message": {"content": "NOT JSON"}}]})
        if calls["n"] == 2:
            bad = json.loads(_valid_section_plan_text())
            bad["sections"][0]["image_ids"] = ["nope"]
            return 200, json.dumps({"choices": [{"message": {"content": json.dumps(bad)}}]})
        return 200, json.dumps({"choices": [{"message": {"content": _valid_section_plan_text()}}]})

    plan, attempts = plan_sections(
        style_instruction="test",
        aspect="9:16",
        audio=_audio(),
        images=_images(),
        api_key="k",
        model="primary/m",
        fallback_models=["fallback/m"],
        transport=transport,
        load_env=False,
    )
    assert plan["schema"] == "mv.sectionplan.v1"
    assert len(attempts) == 3
    assert attempts[-1]["model"] == "fallback/m"
    assert attempts[-1]["schema_ok"] is True


def test_all_attempts_fail_no_fabricate():
    def transport(url, headers, body, timeout_s):
        return 500, "nope"

    with pytest.raises(PlannerError, match="refusing to fabricate"):
        plan_sections(
            style_instruction="test",
            aspect="9:16",
            audio=_audio(),
            images=_images(),
            api_key="k",
            model="primary/m",
            fallback_models=[],
            transport=transport,
            load_env=False,
        )


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    with pytest.raises(PlannerError, match="OPENROUTER_API_KEY"):
        require_openrouter_config(load_env=False)


def test_missing_model(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    with pytest.raises(PlannerError, match="OPENROUTER_MODEL"):
        require_openrouter_config(load_env=False)


def test_nan_rejected():
    with pytest.raises(PlannerError, match="non-finite"):
        parse_llm_json('{"schema":"mv.sectionplan.v1","x":NaN}')


@pytest.mark.skipif(
    os.environ.get("MV_E2E") != "1" or not os.environ.get("OPENROUTER_API_KEY"),
    reason="live OpenRouter gated: need MV_E2E=1 AND OPENROUTER_API_KEY",
)
def test_live_openrouter_optional():
    # Ordinary suite never enters here without both gates.
    plan, attempts = plan_sections(
        style_instruction="short cinematic test",
        aspect="9:16",
        audio=_audio(),
        images=_images(),
        load_env=True,
    )
    assert plan["schema"] == "mv.sectionplan.v1"
    assert attempts
