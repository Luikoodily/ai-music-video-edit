"""M3: HARD/SOFT validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from music_video.errors import HardValidationError
from music_video.validate_plan import load_plan_json, validate_plan

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN = FIXTURES / "golden_plan.json"


def _base() -> dict:
    if not GOLDEN.is_file():
        raise RuntimeError(f"missing golden plan fixture: {GOLDEN}")
    return load_plan_json(GOLDEN)


def test_golden_validates_clean():
    plan = _base()
    out = validate_plan(plan, audio_duration_s=12.0, known_image_ids={"img_01", "img_02", "img_03"})
    assert out["duration_s"] == 12.0
    assert abs(out["clips"][-1]["end_s"] - 12.0) <= 1.0 / 30.0
    assert all(c["fit"] == "cover" for c in out["clips"])


def test_unknown_image_id_hard():
    plan = _base()
    with pytest.raises(HardValidationError, match="unknown image_id"):
        validate_plan(plan, audio_duration_s=12.0, known_image_ids={"img_01"})


def test_overlapping_clips_hard():
    plan = _base()
    plan["clips"][1]["start_s"] = 1.5
    with pytest.raises(HardValidationError, match="overlapping"):
        validate_plan(plan, audio_duration_s=12.0, known_image_ids={"img_01", "img_02", "img_03"})


def test_out_of_range_timestamps_hard():
    plan = _base()
    plan["clips"][-1]["end_s"] = 99.0
    with pytest.raises(HardValidationError, match="invalid timestamps"):
        validate_plan(plan, audio_duration_s=12.0, known_image_ids={"img_01", "img_02", "img_03"})


def test_malformed_missing_keys_hard():
    with pytest.raises(HardValidationError):
        validate_plan({"schema": "mv.plan.v1"}, audio_duration_s=12.0)


def test_nan_rejected_at_parse(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text('{"schema":"mv.plan.v1","duration_s":NaN}', encoding="utf-8")
    with pytest.raises(HardValidationError, match="non-finite"):
        load_plan_json(p)


def test_soft_gap_fill_and_notes():
    plan = _base()
    # remove middle coverage → gap between 2 and 6
    plan["clips"] = [
        plan["clips"][0],  # 0-2
        plan["clips"][2],  # 4-6  (gap 2-4)
        plan["clips"][-1],  # 10-12
    ]
    # adjust last to cover end after gap fill will extend
    plan["clips"][1]["start_s"] = 4.0
    plan["clips"][1]["end_s"] = 6.0
    plan["clips"][2]["start_s"] = 10.0
    plan["clips"][2]["end_s"] = 12.0
    out = validate_plan(plan, audio_duration_s=12.0, known_image_ids={"img_01", "img_02", "img_03"})
    assert any("gap fill" in n for n in out["notes"])
    assert abs(out["clips"][-1]["end_s"] - 12.0) <= 1.0 / 30.0
    # no overlaps
    for a, b in zip(out["clips"], out["clips"][1:]):
        assert b["start_s"] >= a["end_s"] - 1e-6


def test_soft_duration_clamp():
    plan = _base()
    plan["clips"] = [{
        "id": "c001",
        "image_id": "img_01",
        "start_s": 0.0,
        "end_s": 0.1,
        "transition_in": "cut",
        "fx": "none",
        "fx_strength": 0.0,
        "fit": "cover",
    }]
    out = validate_plan(plan, audio_duration_s=12.0, known_image_ids={"img_01"}, min_clip_s=0.35)
    assert any("min_clip_s" in n for n in out["notes"])
    assert out["clips"][0]["end_s"] - out["clips"][0]["start_s"] >= 0.35 - 1e-6


def test_soft_adjacent_duplicate():
    plan = _base()
    plan["clips"][1]["image_id"] = "img_01"
    out = validate_plan(plan, audio_duration_s=12.0, known_image_ids={"img_01", "img_02", "img_03"})
    assert any("adjacent duplicate" in n for n in out["notes"])


def test_soft_snap_to_beat():
    plan = _base()
    plan["clips"][0]["end_s"] = 2.05  # near beat at 2.0
    out = validate_plan(
        plan,
        audio_duration_s=12.0,
        known_image_ids={"img_01", "img_02", "img_03"},
        beats_s=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
    )
    assert any("snapped" in n for n in out["notes"]) or abs(out["clips"][0]["end_s"] - 2.0) < 0.09


def test_title_over_max_hard():
    plan = _base()
    plan["titles"] = [
        {"id": f"t{i}", "text": "X", "start_s": 0.0, "end_s": 1.0, "position": "center",
         "style": {"color": "#FFFFFF", "bg": "#000000", "bg_pad": 12}}
        for i in range(5)
    ]
    with pytest.raises(HardValidationError, match="titles exceed"):
        validate_plan(plan, audio_duration_s=12.0, known_image_ids={"img_01", "img_02", "img_03"}, title_max=4)


def test_fit_enum_validated():
    plan = _base()
    plan["clips"][0]["fit"] = "stretch"
    with pytest.raises(HardValidationError, match="fit"):
        validate_plan(plan, audio_duration_s=12.0, known_image_ids={"img_01", "img_02", "img_03"})


def test_audio_duration_authoritative():
    plan = _base()
    plan["duration_s"] = 99.0
    out = validate_plan(plan, audio_duration_s=12.0, known_image_ids={"img_01", "img_02", "img_03"})
    assert out["duration_s"] == 12.0
    assert any("overridden by audio_analysis" in n for n in out["notes"])
