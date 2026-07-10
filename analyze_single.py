# -*- coding: utf-8 -*-
"""Single GRB analysis compatibility helpers."""

from __future__ import annotations

from typing import Any, Dict, List

from .special_bursts import _load_special_burst_config


def _segments_from_special_burst(burst: Dict[str, Any], fallback_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not burst:
        return fallback_segments
    segments = burst.get("time_segments", [])
    if isinstance(segments, list) and segments:
        out: List[Dict[str, Any]] = []
        for i, seg in enumerate(segments):
            if not isinstance(seg, dict) or "start" not in seg or "stop" not in seg:
                continue
            out.append({"tstart": float(seg["start"]), "tstop": float(seg["stop"]), "tag": str(seg.get("name", f"seg{i + 1}"))})
        return out or fallback_segments
    slices = burst.get("time_slices", [])
    if not slices:
        return fallback_segments
    if len(slices) % 2 != 0:
        return fallback_segments
    out: List[Dict[str, Any]] = []
    for i in range(0, len(slices), 2):
        out.append({"tstart": float(slices[i]), "tstop": float(slices[i + 1]), "tag": f"seg{i // 2 + 1}"})
    return out or fallback_segments


__all__ = ["_load_special_burst_config", "_segments_from_special_burst"]
