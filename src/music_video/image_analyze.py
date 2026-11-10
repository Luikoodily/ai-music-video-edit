"""Local image intelligence: metadata, phash dedupe, CLIP labels + section scores.

Torch / open_clip are imported LAZY — metadata/phash/dedupe paths work without torch.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import imagehash
from PIL import Image

from music_video.errors import ImageAnalysisError

# --- clearly labelled thresholds ---
# Hamming distance on 64-bit perceptual hash; <= this ⇒ duplicate pair.
PHASH_HAMMING_THRESHOLD = 5

# Zero-shot label kept if cosine sim >= this.
LABEL_SCORE_THRESHOLD = 0.22

# Section kept in best_sections if score >= this.
SECTION_SCORE_THRESHOLD = 0.15
BEST_SECTIONS_TOP_K = 3

LABEL_VOCAB = (
    "portrait",
    "landscape",
    "action",
    "detail",
    "graphic",
    "text-heavy",
    "hero",
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"
# open-clip embedding dim for ViT-B-32
CLIP_EMBED_DIM = 512

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLIP_CACHE = PACKAGE_ROOT / "models" / "open-clip"


def clip_cache_dir() -> Path:
    return DEFAULT_CLIP_CACHE


def _weight_candidates(root: Path) -> list[Path]:
    """open-clip/HF cache may use .safetensors, .pt, or .bin."""
    found: list[Path] = []
    for pattern in ("*.safetensors", "*.pt", "*.bin"):
        found.extend(root.rglob(pattern))
    # prefer snapshot files over raw blobs when both exist
    found = [p for p in found if p.is_file() and p.stat().st_size > 1_000_000]
    return sorted(found, key=lambda p: (0 if "snapshots" in p.parts else 1, str(p)))


def clip_weights_ready(cache_dir: Path | None = None) -> bool:
    """True if a local open-clip openai ViT-B-32 weight file is present (no network)."""
    root = Path(cache_dir) if cache_dir else clip_cache_dir()
    if not root.is_dir():
        return False
    return bool(_weight_candidates(root))


def resolve_weights_path(cache_dir: Path | None = None) -> str:
    root = Path(cache_dir) if cache_dir else clip_cache_dir()
    if not root.is_dir():
        return str(root)
    hits = _weight_candidates(root)
    if hits:
        # Prefer logical snapshot path (do not follow HF blob symlinks for reporting)
        return str(hits[0])
    return str(root)


def _assert_finite(obj: Any, path: str = "$") -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _assert_finite(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_finite(v, f"{path}[{i}]")
    elif isinstance(obj, float):
        if not math.isfinite(obj):
            raise ImageAnalysisError(f"non-finite float at {path}")


def list_image_paths(images_dir: Path) -> list[Path]:
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise ImageAnalysisError(f"images dir not found: {images_dir}")
    files = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    files.sort(key=lambda p: p.name.lower())
    if not files:
        raise ImageAnalysisError(f"no images in {images_dir}")
    return files


def stable_image_id(index: int) -> str:
    return f"img_{index:02d}"


def collect_metadata(path: Path, image_id: str) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ImageAnalysisError(f"image missing: {path}")
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            ph = str(imagehash.phash(im))
    except Exception as exc:  # noqa: BLE001
        raise ImageAnalysisError(f"cannot read image {path}: {exc}") from exc
    size = path.stat().st_size
    aspect = float(w) / float(h) if h else 0.0
    if not math.isfinite(aspect):
        raise ImageAnalysisError(f"bad aspect for {path}")
    return {
        "id": image_id,
        "path": path.as_posix(),
        "width": int(w),
        "height": int(h),
        "aspect": round(aspect, 6),
        "file_size": int(size),
        "phash": ph,
        "unused": False,
        "labels": [],
        "section_scores": {},
        "best_sections": [],
    }


def hamming_hex(a: str, b: str) -> int:
    """Hamming distance between two hex phash strings."""
    ha = imagehash.hex_to_hash(a)
    hb = imagehash.hex_to_hash(b)
    return int(ha - hb)


def find_duplicates(
    records: list[dict[str, Any]],
    *,
    threshold: int = PHASH_HAMMING_THRESHOLD,
) -> tuple[list[list[str]], list[dict[str, Any]]]:
    """Return duplicates pairs and records with secondary copies marked unused.

    Never deletes or moves files. First id in sorted pair order stays primary.
    """
    n = len(records)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs: list[list[str]] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = hamming_hex(records[i]["phash"], records[j]["phash"])
            if d <= threshold:
                pairs.append(sorted([records[i]["id"], records[j]["id"]]))
                ri, rj = find(i), find(j)
                if ri != rj:
                    # keep lower index (earlier sorted path) as root
                    if ri < rj:
                        parent[rj] = ri
                    else:
                        parent[ri] = rj

    # unique sorted pairs
    uniq: list[list[str]] = []
    seen: set[tuple[str, str]] = set()
    for a, b in pairs:
        key = (a, b)
        if key not in seen:
            seen.add(key)
            uniq.append([a, b])
    uniq.sort(key=lambda x: (x[0], x[1]))

    out = []
    for i, rec in enumerate(records):
        r = dict(rec)
        root = find(i)
        r["unused"] = root != i  # secondary in cluster
        out.append(r)
    return uniq, out


def apply_label_threshold(
    scores: dict[str, float],
    *,
    threshold: float = LABEL_SCORE_THRESHOLD,
) -> list[str]:
    """Keep fixed-vocab labels whose score >= threshold, sorted by score desc."""
    kept = [(k, float(v)) for k, v in scores.items() if math.isfinite(float(v)) and float(v) >= threshold]
    kept.sort(key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in kept]


def apply_section_ranking(
    scores: dict[str, float],
    *,
    threshold: float = SECTION_SCORE_THRESHOLD,
    top_k: int = BEST_SECTIONS_TOP_K,
) -> list[str]:
    kept = [(k, float(v)) for k, v in scores.items() if math.isfinite(float(v)) and float(v) >= threshold]
    kept.sort(key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in kept[:top_k]]


def build_metadata_records(images_dir: Path) -> list[dict[str, Any]]:
    paths = list_image_paths(images_dir)
    return [collect_metadata(p, stable_image_id(i + 1)) for i, p in enumerate(paths)]


def validate_images_payload_contract(payload: dict[str, Any]) -> None:
    """JSON-contract checks without torch."""
    if payload.get("schema") != "mv.images.v1":
        raise ImageAnalysisError("schema must be mv.images.v1")
    images = payload.get("images")
    if not isinstance(images, list) or not images:
        raise ImageAnalysisError("images must be non-empty list")
    for i, im in enumerate(images):
        for key in ("id", "path", "width", "height", "aspect", "phash", "labels", "section_scores"):
            if key not in im:
                raise ImageAnalysisError(f"images[{i}] missing {key}")
        if not re.fullmatch(r"img_\d{2,}", str(im["id"])):
            raise ImageAnalysisError(f"unstable id: {im['id']}")
    if "duplicates" not in payload or not isinstance(payload["duplicates"], list):
        raise ImageAnalysisError("duplicates must be a list")
    _assert_finite(payload)


def _lazy_torch_clip():
    try:
        import torch
        import open_clip
    except ImportError as exc:
        raise ImageAnalysisError(
            "torch/open-clip-torch not installed; run pip install and scripts/bootstrap_clip.py"
        ) from exc
    return torch, open_clip


def load_clip(cache_dir: Path | None = None):
    """Load ViT-B-32 openai weights from local cache only (no download)."""
    torch, open_clip = _lazy_torch_clip()
    cache = Path(cache_dir) if cache_dir else clip_cache_dir()
    if not clip_weights_ready(cache):
        raise ImageAnalysisError(
            f"CLIP weights missing under {cache}. Run: python scripts/bootstrap_clip.py"
        )
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME,
        pretrained=CLIP_PRETRAINED,
        cache_dir=str(cache),
    )
    model.eval()
    # force CPU — no GPU code path
    device = torch.device("cpu")
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    return model, preprocess, tokenizer, device, torch


def _encode_images(model, preprocess, device, torch, paths: list[Path]):
    tensors = []
    for p in paths:
        with Image.open(p) as im:
            im = im.convert("RGB")
            tensors.append(preprocess(im))
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def _encode_texts(model, tokenizer, device, torch, texts: list[str]):
    tokens = tokenizer(texts).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def section_prompts_from_audio(
    audio_analysis: dict[str, Any] | None,
    style_words: str | None = None,
) -> dict[str, str]:
    style = (style_words or "").strip()
    if audio_analysis and isinstance(audio_analysis.get("sections"), list) and audio_analysis["sections"]:
        out: dict[str, str] = {}
        for s in audio_analysis["sections"]:
            sid = str(s.get("id") or "s0")
            label = str(s.get("label") or "section")
            prompt = f"a music video still for the {label} section"
            if style:
                prompt = f"{prompt}, {style}"
            out[sid] = prompt
        return out
    # generic default
    base = "a cinematic music video still"
    if style:
        base = f"{base}, {style}"
    return {"s0": base}


def analyze_images(
    images_dir: Path,
    *,
    audio_analysis: dict[str, Any] | None = None,
    style_words: str | None = None,
    cache_dir: Path | None = None,
    with_clip: bool = True,
) -> dict[str, Any]:
    """Full image analysis → mv.images.v1 payload."""
    records = build_metadata_records(images_dir)
    duplicates, records = find_duplicates(records, threshold=PHASH_HAMMING_THRESHOLD)

    model_id = f"open_clip:{CLIP_MODEL_NAME}:{CLIP_PRETRAINED}"
    weights_path = resolve_weights_path(cache_dir)
    embedding_dim = CLIP_EMBED_DIM

    if with_clip:
        model, preprocess, tokenizer, device, torch = load_clip(cache_dir)
        paths = [Path(r["path"]) for r in records]
        img_feats = _encode_images(model, preprocess, device, torch, paths)
        if int(img_feats.shape[-1]) != CLIP_EMBED_DIM:
            embedding_dim = int(img_feats.shape[-1])

        # labels zero-shot
        label_texts = [f"a photo of {lab}" for lab in LABEL_VOCAB]
        lab_feats = _encode_texts(model, tokenizer, device, torch, label_texts)
        # section prompts
        prompts = section_prompts_from_audio(audio_analysis, style_words)
        sec_ids = list(prompts.keys())
        sec_feats = _encode_texts(model, tokenizer, device, torch, [prompts[k] for k in sec_ids])

        for i, rec in enumerate(records):
            sims = (img_feats[i] @ lab_feats.T).cpu().tolist()
            raw_scores = {LABEL_VOCAB[j]: float(sims[j]) for j in range(len(LABEL_VOCAB))}
            rec["label_scores"] = {k: round(v, 6) for k, v in raw_scores.items()}
            rec["labels"] = apply_label_threshold(raw_scores)

            ssims = (img_feats[i] @ sec_feats.T).cpu().tolist()
            sec_scores = {sec_ids[j]: float(ssims[j]) for j in range(len(sec_ids))}
            rec["section_scores"] = {k: round(v, 6) for k, v in sec_scores.items()}
            rec["best_sections"] = apply_section_ranking(sec_scores)
    else:
        model_id = "none"
        weights_path = ""
        embedding_dim = 0

    payload = {
        "schema": "mv.images.v1",
        "model_id": model_id,
        "weights_path": weights_path,
        "embedding_dim": int(embedding_dim),
        "phash_hamming_threshold": int(PHASH_HAMMING_THRESHOLD),
        "label_score_threshold": float(LABEL_SCORE_THRESHOLD),
        "section_score_threshold": float(SECTION_SCORE_THRESHOLD),
        "images": records,
        "duplicates": duplicates,
    }
    validate_images_payload_contract(payload)
    _assert_finite(payload)
    return payload


def write_image_analysis(
    images_dir: Path,
    out_json: Path,
    *,
    audio_analysis_path: Path | None = None,
    style_words: str | None = None,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    audio = None
    if audio_analysis_path and Path(audio_analysis_path).is_file():
        audio = json.loads(Path(audio_analysis_path).read_text(encoding="utf-8"))
    data = analyze_images(
        images_dir,
        audio_analysis=audio,
        style_words=style_words,
        cache_dir=cache_dir,
        with_clip=True,
    )
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data
