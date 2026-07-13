# -*- coding: utf-8 -*-
"""Special burst configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence

from .logging_utils import log
from .session import session

SPECIAL_BURST_YAML_DEFAULT = "special_bursts.yaml"


def _normalize_time_segments(raw_segments: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_segments, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for idx, seg in enumerate(raw_segments):
        if not isinstance(seg, dict):
            continue
        if "start" not in seg or "stop" not in seg:
            continue
        try:
            start = float(seg["start"])
            stop = float(seg["stop"])
        except Exception:
            continue
        normalized.append(
            {
                "name": str(seg.get("name", f"seg{idx + 1}")),
                "start": start,
                "stop": stop,
            }
        )
    return normalized


def _normalize_time_slices(raw_slices: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_slices, list) or len(raw_slices) % 2 != 0:
        return []

    normalized: List[Dict[str, Any]] = []
    for idx in range(0, len(raw_slices), 2):
        try:
            start = float(raw_slices[idx])
            stop = float(raw_slices[idx + 1])
        except Exception:
            return []
        normalized.append({"name": f"seg{idx // 2 + 1}", "start": start, "stop": stop})
    return normalized


def _normalize_special_burst_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    burst = dict(raw)

    active_interval = burst.get("active_interval")
    if active_interval is not None:
        burst["active_interval"] = str(active_interval).strip()

    background_interval = burst.get("background_interval")
    if background_interval is not None:
        burst["background_interval"] = str(background_interval).strip()

    segments = _normalize_time_segments(burst.get("time_segments"))
    if not segments:
        segments = _normalize_time_slices(burst.get("time_slices"))
    burst["time_segments"] = segments
    return burst


def parse_time_segments_text(text: str) -> List[Dict[str, Any]]:
    """Parse Streamlit text-area rows into special-burst time segments.

    Accepted row formats are ``name start stop`` and ``start stop``. Commas may
    be used instead of whitespace. Empty rows and ``#`` comments are ignored.
    """

    segments: List[Dict[str, Any]] = []
    for line_no, raw_line in enumerate(str(text).splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = [part for part in re.split(r"[\s,]+", line) if part]
        if len(parts) == 2:
            name = f"seg{len(segments) + 1}"
            start_text, stop_text = parts
        elif len(parts) == 3:
            name, start_text, stop_text = parts
        else:
            raise ValueError(f"第 {line_no} 行格式无效，请使用 'name start stop' 或 'start stop'")
        try:
            start = float(start_text)
            stop = float(stop_text)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"第 {line_no} 行 start/stop 必须是数字") from exc
        if stop <= start:
            raise ValueError(f"第 {line_no} 行 stop 必须大于 start")
        segments.append({"name": str(name), "start": start, "stop": stop})
    if not segments:
        raise ValueError("至少需要填写一个时间分 bin")
    return segments


def format_time_segments_text(segments: Sequence[dict]) -> str:
    """Format normalized time segments for the Streamlit editor."""

    rows: List[str] = []
    for idx, segment in enumerate(_normalize_time_segments(list(segments)), start=1):
        rows.append(
            f"{segment.get('name') or f'seg{idx}'} {float(segment['start']):g} {float(segment['stop']):g}"
        )
    return "\n".join(rows)


def upsert_special_burst_config(
    yaml_path: str | Path,
    *,
    name: str,
    bnname: str,
    active_interval: str = "",
    background_interval: str = "",
    time_segments: Sequence[dict],
    description: str = "特殊时间分段与背景窗",
) -> Dict[str, Any]:
    """Insert or replace one burst in ``special_bursts.yaml``.

    Existing rows are matched by ``bnname`` first and ``name`` second. Other
    bursts keep their order and content.
    """

    import yaml

    clean_name = str(name).strip()
    clean_bnname = str(bnname).strip()
    if not clean_name:
        raise ValueError("name 不能为空")
    if not clean_bnname:
        raise ValueError("bnname 不能为空")

    normalized_segments = _normalize_time_segments(list(time_segments))
    if not normalized_segments:
        raise ValueError("至少需要一个有效的 time_segments")

    burst: Dict[str, Any] = {
        "name": clean_name,
        "bnname": clean_bnname,
    }
    clean_description = str(description).strip()
    if clean_description:
        burst["description"] = clean_description
    clean_active = str(active_interval).strip()
    if clean_active:
        burst["active_interval"] = clean_active
    clean_background = str(background_interval).strip()
    if clean_background:
        burst["background_interval"] = clean_background
    burst["time_segments"] = normalized_segments

    path = Path(yaml_path).expanduser()
    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    else:
        loaded = {}
    data: Dict[str, Any] = loaded if isinstance(loaded, dict) else {}
    bursts = data.get("special_bursts", [])
    if not isinstance(bursts, list):
        bursts = []

    replace_idx: Optional[int] = None
    for idx, item in enumerate(bursts):
        if isinstance(item, dict) and item.get("bnname") == clean_bnname:
            replace_idx = idx
            break
    if replace_idx is None:
        for idx, item in enumerate(bursts):
            if isinstance(item, dict) and item.get("name") == clean_name:
                replace_idx = idx
                break
    if replace_idx is None:
        bursts.append(burst)
    else:
        bursts[replace_idx] = burst

    data["special_bursts"] = bursts
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return burst


def _load_special_burst_config(yaml_path: Optional[str], bnname: str, grb_name: Optional[str] = None) -> Dict[str, Any]:
    path = Path(yaml_path or SPECIAL_BURST_YAML_DEFAULT)
    if not path.is_absolute():
        path = Path(session.result_root) / path
    if not path.exists():
        path = Path(__file__).parent / SPECIAL_BURST_YAML_DEFAULT
    if not path.exists():
        log(f"{bnname}: 未找到 special_bursts.yaml，尝试路径: {path}")
        return {}
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        bursts = data.get("special_bursts", []) if isinstance(data, dict) else []
        for item in bursts:
            if not isinstance(item, dict):
                continue
            if item.get("bnname") == bnname:
                log(f"{bnname}: 命中特殊暴配置（按 bnname），name={item.get('name')}, grb_name={grb_name}")
                return item
        if grb_name:
            for item in bursts:
                if not isinstance(item, dict):
                    continue
                if item.get("name") == grb_name:
                    log(f"{bnname}: 命中特殊暴配置（按 name 回退），name={item.get('name')}, bnname={item.get('bnname')}")
                    return item
        log(f"{bnname}: special_bursts.yaml 中未命中特例，已回退目录表/默认分段")
    except Exception as exc:  # noqa: BLE001
        log(f"{bnname}: 读取特殊暴 YAML 失败: {exc}")
    return {}


def load_special_burst_config(yaml_path: Optional[str], bnname: str, grb_name: Optional[str] = None) -> Dict[str, Any]:
    """Load and normalize a special burst configuration from YAML."""

    return _normalize_special_burst_config(_load_special_burst_config(yaml_path, bnname, grb_name))
