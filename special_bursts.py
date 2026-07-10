# -*- coding: utf-8 -*-
"""Special burst configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

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
