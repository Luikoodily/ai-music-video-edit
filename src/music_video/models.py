"""MusicVideoPlan schema (mv.plan.v1) — independent of speech CutInterval."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class FitMode(str, Enum):
    COVER = "cover"
    CONTAIN = "contain"


class TransitionIn(str, Enum):
    CUT = "cut"
    # xfade deferred — accepted in schema for forward compat but renderer ignores
    XFADE = "xfade"


class FxKind(str, Enum):
    NONE = "none"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"


FIT_ENUM = {m.value for m in FitMode}
TRANSITION_ENUM = {m.value for m in TransitionIn}
FX_ENUM = {m.value for m in FxKind}
ASPECT_ENUM = {"9:16", "16:9"}


@dataclass
class TitleStyle:
    color: str = "#FFFFFF"
    bg: str = "#000000"
    bg_pad: int = 12

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TitleSpec:
    id: str
    text: str
    start_s: float
    end_s: float
    position: str = "center"
    style: TitleStyle = field(default_factory=TitleStyle)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "position": self.position,
            "style": self.style.to_dict() if isinstance(self.style, TitleStyle) else self.style,
        }


@dataclass
class ClipSpec:
    id: str
    image_id: str
    start_s: float
    end_s: float
    transition_in: str = TransitionIn.CUT.value
    fx: str = FxKind.NONE.value
    fx_strength: float = 0.0
    fit: str = FitMode.COVER.value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MusicVideoPlan:
    schema: str
    aspect: str
    width: int
    height: int
    fps: int
    duration_s: float
    audio_path: str
    clips: list[ClipSpec]
    titles: list[TitleSpec] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "aspect": self.aspect,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "duration_s": self.duration_s,
            "audio_path": self.audio_path,
            "clips": [c.to_dict() for c in self.clips],
            "titles": [t.to_dict() for t in self.titles],
            "notes": list(self.notes),
        }


def aspect_size(aspect: str) -> tuple[int, int]:
    if aspect == "9:16":
        return 1080, 1920
    if aspect == "16:9":
        return 1920, 1080
    raise ValueError(f"unsupported aspect: {aspect}")
