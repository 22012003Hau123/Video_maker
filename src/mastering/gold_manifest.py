"""
Burberry Gold POS — load manifest from gold_subtitles.yaml, expand job matrix.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

_MANIFEST_NAME = "gold_subtitles.yaml"


def _package_dir() -> Path:
    return Path(__file__).resolve().parent


def default_manifest_path() -> Path:
    return _package_dir() / _MANIFEST_NAME


def load_manifest(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or default_manifest_path()
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def length_group_for(duration_seconds: int, data: Optional[Dict[str, Any]] = None) -> str:
    data = data or load_manifest()
    for name, grp in data.get("length_groups", {}).items():
        if duration_seconds in grp.get("durations_seconds", []):
            return name
    raise ValueError(f"Duration {duration_seconds}s not in manifest length_groups")


def subtitle_lines_for(vo: str, duration_seconds: int, data: Optional[Dict[str, Any]] = None) -> List[str]:
    data = data or load_manifest()
    lg = length_group_for(duration_seconds, data)
    lines = data["length_groups"][lg]["lines"].get(vo)
    if not lines:
        raise ValueError(f"No subtitle lines for vo={vo} in group {lg}")
    return list(lines)


@dataclass
class GoldJobSpec:
    vo: str
    line_id: str
    line_label: str
    layout: str
    branded: bool
    subtitles: bool
    video_format: str
    duration_seconds: int
    export_codec_id: str
    subtitle_lines: List[str] = field(default_factory=list)
    output_basename_hint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def _format_suffix(fmt_id: str) -> str:
    return fmt_id.replace(":", "x")


def build_basename(spec: GoldJobSpec) -> str:
    parts = [
        "BurberryGold",
        spec.vo.upper(),
        spec.line_id,
        _format_suffix(spec.video_format),
        f"{spec.duration_seconds}s",
        spec.export_codec_id.replace("_", ""),
    ]
    return "_".join(parts)


def expand_jobs(
    vo: str,
    line_ids: Optional[Sequence[str]] = None,
    formats: Optional[Sequence[str]] = None,
    durations_seconds: Optional[Sequence[int]] = None,
    export_codec_ids: Optional[Sequence[str]] = None,
    data: Optional[Dict[str, Any]] = None,
) -> List[GoldJobSpec]:
    """
    Cartesian product of selected deliverable lines × formats × durations × codecs.
    """
    data = data or load_manifest()
    lines_def: List[Dict[str, Any]] = data["deliverable_lines"][vo]
    if line_ids:
        wanted = set(line_ids)
        lines_def = [x for x in lines_def if x["id"] in wanted]
        if len(lines_def) != len(wanted):
            missing = wanted - {x["id"] for x in lines_def}
            raise ValueError(f"Unknown line_id(s): {missing}")

    fmt_ids = list(formats) if formats else [f["id"] for f in data.get("formats", [])]
    durs = list(durations_seconds) if durations_seconds else list(data.get("durations_seconds", []))
    codecs = list(export_codec_ids) if export_codec_ids else [c["id"] for c in data.get("export_codecs", [])]

    out: List[GoldJobSpec] = []
    for ld in lines_def:
        for vf in fmt_ids:
            for dur in durs:
                for ec in codecs:
                    try:
                        sub_lines = subtitle_lines_for(vo, dur, data) if ld["subtitles"] else []
                    except ValueError:
                        sub_lines = []
                    spec = GoldJobSpec(
                        vo=vo,
                        line_id=ld["id"],
                        line_label=ld["label"],
                        layout=ld["layout"],
                        branded=ld["branded"],
                        subtitles=ld["subtitles"],
                        video_format=vf,
                        duration_seconds=dur,
                        export_codec_id=ec,
                        subtitle_lines=sub_lines,
                    )
                    spec.output_basename_hint = build_basename(spec)
                    out.append(spec)
    return out


def count_matrix(data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    data = data or load_manifest()
    n_fmt = len(data.get("formats", []))
    n_dur = len(data.get("durations_seconds", []))
    n_line = len(data["deliverable_lines"]["vo1"])
    n_codec = len(data.get("export_codecs", []))
    content_per_vo = n_fmt * n_dur * n_line
    file_outputs_per_vo = content_per_vo * n_codec
    return {
        "formats": n_fmt,
        "durations": n_dur,
        "lines_per_vo": n_line,
        "codecs": n_codec,
        "content_variants_per_vo": content_per_vo,
        "file_outputs_per_vo": file_outputs_per_vo,
        "outputs_per_vo": file_outputs_per_vo,
        "outputs_total_both_vo": file_outputs_per_vo * 2,
    }


def resolve_sound_wav(duration_seconds: int, data: Optional[Dict[str, Any]] = None, project_root: Optional[Path] = None) -> Tuple[Optional[str], bool]:
    data = data or load_manifest()
    root = project_root or Path(__file__).resolve().parents[2]
    pat = data.get("asset_paths", {}).get("sound_wav", {}).get("pattern", "")
    rel = pat.format(duration=duration_seconds)
    abs_path = (root / rel).resolve()
    return str(abs_path), abs_path.is_file()


def resolve_logo_path(video_format: str, data: Optional[Dict[str, Any]] = None, project_root: Optional[Path] = None) -> Tuple[str, bool]:
    data = data or load_manifest()
    root = project_root or Path(__file__).resolve().parents[2]
    pat = data.get("asset_paths", {}).get("logo_png", {}).get("pattern", "")
    rel = pat.format(format=video_format)
    abs_path = (root / rel).resolve()
    return str(abs_path), abs_path.is_file()


def resolve_opening_path(
    data: Optional[Dict[str, Any]] = None,
    project_root: Optional[Path] = None,
) -> Tuple[str, bool]:
    data = data or load_manifest()
    root = project_root or Path(__file__).resolve().parents[2]
    rel = data.get("asset_paths", {}).get("opening", {}).get("default", "")
    if not rel:
        return "", False
    p = (root / rel).resolve()
    return str(p), p.is_file()


def resolve_full_video_path(
    data: Optional[Dict[str, Any]] = None,
    project_root: Optional[Path] = None,
) -> Tuple[str, bool]:
    data = data or load_manifest()
    root = project_root or Path(__file__).resolve().parents[2]
    rel = data.get("asset_paths", {}).get("full_video", {}).get("default", "")
    if not rel:
        return "", False
    p = (root / rel).resolve()
    return str(p), p.is_file()


def resolve_closing_path(
    layout: str,
    video_format: str,
    data: Optional[Dict[str, Any]] = None,
    project_root: Optional[Path] = None,
) -> Tuple[str, bool]:
    data = data or load_manifest()
    root = project_root or Path(__file__).resolve().parents[2]
    closing = data.get("asset_paths", {}).get("closing", {})
    solo = closing.get("solo") or {}
    duo = closing.get("duo") or {}
    fallback = data.get("assembly", {}).get("closing_solo_fallback_to_duo", True)
    rel: Optional[str] = None
    if layout == "solo":
        rel = solo.get(video_format) or solo.get("16x9")
        if not rel and fallback:
            rel = duo.get(video_format) or duo.get("16x9")
    else:
        rel = duo.get(video_format) or duo.get("16x9")
    if not rel:
        return "", False
    p = (root / rel).resolve()
    return str(p), p.is_file()


def resolve_surimp_path(
    vo: str,
    layout: str,
    video_format: str,
    data: Optional[Dict[str, Any]] = None,
    project_root: Optional[Path] = None,
) -> Tuple[Optional[str], bool]:
    data = data or load_manifest()
    root = project_root or Path(__file__).resolve().parents[2]
    sm = data.get("asset_paths", {}).get("surimps", {})
    if vo == "vo1":
        pat = sm.get("vo1", {}).get("pattern", "")
    elif vo == "vo2" and layout == "duo":
        pat = sm.get("vo2_duo", {}).get("pattern", "")
    else:
        pat = sm.get("vo2_solo", {}).get("pattern", "")
    if not pat:
        return None, False
    rel = pat.format(format=video_format)
    abs_path = (root / rel).resolve()
    return str(abs_path), abs_path.is_file()


def _solo_closing_is_placeholder(
    data: Dict[str, Any],
    project_root: Path,
) -> bool:
    """
    True when solo closing points to the same file as duo (chưa có packshot solo riêng).
    Ignored if assembly.allow_solo_closing_same_as_duo is true.
    """
    assembly = data.get("assembly", {})
    if assembly.get("allow_solo_closing_same_as_duo"):
        return False
    ps, ok_s = resolve_closing_path("solo", "16x9", data, project_root)
    pd, ok_d = resolve_closing_path("duo", "16x9", data, project_root)
    if not ok_s or not ok_d:
        return False
    return Path(ps).resolve() == Path(pd).resolve()


def _readiness_for_line(
    vo: str,
    line: Dict[str, Any],
    data: Dict[str, Any],
    project_root: Path,
) -> Tuple[bool, List[str]]:
    """Whether assets exist for this deliverable line (opening, full, closing, WAVs, optional branded PNGs)."""
    missing: List[str] = []
    layout = str(line.get("layout", "solo"))
    branded = bool(line.get("branded", False))

    _, ok = resolve_opening_path(data, project_root)
    if not ok:
        missing.append("opening")
    _, ok = resolve_full_video_path(data, project_root)
    if not ok:
        missing.append("full_video")
    _, ok = resolve_closing_path(layout, "16x9", data, project_root)
    if not ok:
        missing.append(f"closing_{layout}")
    if layout == "solo" and _solo_closing_is_placeholder(data, project_root):
        missing.append("closing_solo_dedicated")

    for d in data.get("durations_seconds", []):
        _, ok = resolve_sound_wav(int(d), data, project_root)
        if not ok:
            missing.append(f"wav_{d}s")

    if branded:
        for f in data.get("formats", []):
            fid = str(f["id"])
            _, ok = resolve_logo_path(fid, data, project_root)
            if not ok:
                missing.append(f"logo_{fid}")
            _, ok = resolve_surimp_path(vo, layout, fid, data, project_root)
            if not ok:
                missing.append(f"surimp_{fid}")

    return len(missing) == 0, missing


def _build_asset_flags(data: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
    """Per-file checks for UI (red chips: wav, logo, surimps, base video)."""
    wav_by_duration: Dict[str, bool] = {}
    for d in data.get("durations_seconds", []):
        _, ok = resolve_sound_wav(int(d), data, project_root)
        wav_by_duration[str(d)] = ok

    logo_by_format: Dict[str, bool] = {}
    surimp_vo1: Dict[str, bool] = {}
    surimp_vo2_solo: Dict[str, bool] = {}
    surimp_vo2_duo: Dict[str, bool] = {}
    for f in data.get("formats", []):
        fid = str(f["id"])
        _, ok = resolve_logo_path(fid, data, project_root)
        logo_by_format[fid] = ok
        _, ok = resolve_surimp_path("vo1", "solo", fid, data, project_root)
        surimp_vo1[fid] = ok
        _, ok = resolve_surimp_path("vo2", "solo", fid, data, project_root)
        surimp_vo2_solo[fid] = ok
        _, ok = resolve_surimp_path("vo2", "duo", fid, data, project_root)
        surimp_vo2_duo[fid] = ok

    return {
        "wav_by_duration": wav_by_duration,
        "logo_by_format": logo_by_format,
        "surimp_vo1": surimp_vo1,
        "surimp_vo2_solo": surimp_vo2_solo,
        "surimp_vo2_duo": surimp_vo2_duo,
        "opening_ok": resolve_opening_path(data, project_root)[1],
        "full_video_ok": resolve_full_video_path(data, project_root)[1],
        "closing_duo_ok": resolve_closing_path("duo", "16x9", data, project_root)[1],
        "closing_solo_placeholder": _solo_closing_is_placeholder(data, project_root),
    }


def manifest_for_api(data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Payload for GET /api/mastering/gold-manifest."""
    data = data or load_manifest()
    matrix = count_matrix(data)
    project_root = Path(__file__).resolve().parents[2]
    asset_flags = _build_asset_flags(data, project_root)
    readiness: Dict[str, List[Dict[str, Any]]] = {}
    for vo in ("vo1", "vo2"):
        rows: List[Dict[str, Any]] = []
        for line in data.get("deliverable_lines", {}).get(vo, []):
            ready, miss = _readiness_for_line(vo, line, data, project_root)
            rows.append(
                {
                    "id": line["id"],
                    "ready": ready,
                    "missing": miss,
                }
            )
        readiness[vo] = rows

    return {
        "campaign": data.get("campaign"),
        "matrix": matrix,
        "variants": list(data.get("deliverable_lines", {}).keys()),
        "formats": data.get("formats", []),
        "durations_seconds": data.get("durations_seconds", []),
        "export_codecs": data.get("export_codecs", []),
        "deliverable_lines": data.get("deliverable_lines", {}),
        "readiness": readiness,
        "asset_flags": asset_flags,
        "length_groups": {
            k: {
                "description": v.get("description"),
                "durations_seconds": v.get("durations_seconds", []),
            }
            for k, v in data.get("length_groups", {}).items()
        },
    }
