# -*- coding: utf-8 -*-
"""
GBM（TTE）+ LAT 联合光变示意绘图。

由 ``plot_lightcurves.py`` 迁入：多 NaI 能段平均率、BGO、LAT 直方图与能量散点；
源/本底时间窗与 threeML ``TimeSeriesBuilder`` 一致。
LAT 子图仅使用 ``lat_extended_three_ml`` 产出的 ``interval*/gll_ft1_tr_bn*_v00_filt_prob.fit``
（含 GRB 概率列），不再读取 Extended 下的 ``L*EV00.fits``。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from matplotlib.figure import Figure
from threeML import TimeSeriesBuilder
from threeML.config.config import threeML_config
from threeML.io.plotting.step_plot import step_plot

from .gbm_core import resolve_active_interval_from_special_and_catalog
from .gbm_detector_selection import select_gbm_detectors
from .runtime_env import ensure_analysis_runtime
from .session import session
from .logging_utils import log
from .interactive_lightcurves import interactive_lightcurve_path, write_interactive_lightcurve_html
from .publication_style import (
    AXES_LINEWIDTH,
    LIGHTCURVE_ANNOTATION_SIZE,
    LIGHTCURVE_AXIS_LABEL_SIZE,
    LIGHTCURVE_FIGSIZE,
    LIGHTCURVE_TICK_LABEL_SIZE,
    RASTER_DPI,
    configure_publication_matplotlib,
    save_publication_figure,
    style_lightcurve_axes,
)

configure_publication_matplotlib()

SPECIAL_BURSTS_YAML = Path(__file__).with_name("special_bursts.yaml")

TITLE_BOX = dict(
    boxstyle="round,pad=0.35",
    facecolor="white",
    edgecolor="0.55",
    linewidth=0.75,
    alpha=0.95,
)
PANEL_TITLE_Y = 0.88

PAPER_DETECTOR_STYLES = {
    "nai_n3": {"marker": "P", "color": "#4e79a7"},
    "nai_n7": {"marker": "X", "color": "#f28e2b"},
    "bgo_b0": {"marker": "s", "color": "#59a14f"},
    "lat": {"marker": "v", "color": "#b07aa1"},
}

PAPER_LEGEND_STYLE = dict(
    frameon=True,
    framealpha=0.92,
    edgecolor="#d0d0d0",
    facecolor="white",
    handlelength=1.3,
    handletextpad=0.5,
    borderpad=0.5,
    labelspacing=0.4,
)


def _parse_source_interval_pair(interval: str) -> Optional[Tuple[float, float]]:
    """解析单段源时选段 ``t0-t1``（中间仅一个连字符，如 ``0.1-8``）。"""
    s = str(interval).strip()
    if s.count("-") != 1:
        return None
    a, b = s.split("-", 1)
    try:
        t0, t1 = float(a), float(b)
    except ValueError:
        return None
    if t1 <= t0:
        return None
    return t0, t1


def resolve_spectral_time_bins_for_lightcurve(
    bnname: str,
    active_interval: str,
    spectral_time_bins: Optional[Sequence[Union[np.ndarray, Sequence[float]]]] = None,
    fixed_num_time_bins: Optional[int] = None,
) -> Optional[List[np.ndarray]]:
    """
    确定用于光变图分段标注的 bin 边序列。

    若已传入 ``spectral_time_bins`` 则原样返回；否则根据 ``active_interval`` 的
    ``t0-t1`` 与 :func:`~grb_project.gbm_core._build_time_bins_list` 自动推断
    （含 ``bn231129799`` 等目录内特例）。总段数 < 2 时返回 ``None``。
    """
    if spectral_time_bins is not None:
        normalized: List[np.ndarray] = []
        for tb in spectral_time_bins:
            if isinstance(tb, dict):
                if "start" in tb and "stop" in tb:
                    normalized.append(np.asarray([float(tb["start"]), float(tb["stop"])], dtype=float))
                    continue
                if "tstart" in tb and "tstop" in tb:
                    normalized.append(np.asarray([float(tb["tstart"]), float(tb["tstop"])], dtype=float))
                    continue
            normalized.append(np.asarray(tb, dtype=float))
        return normalized
    parsed = _parse_source_interval_pair(active_interval)
    if parsed is None:
        return None
    from .gbm_core import _build_time_bins_list

    t0, t1 = parsed
    tb_list, _, _ = _build_time_bins_list(
        bnname,
        t0,
        t1,
        fixed_num_time_bins=fixed_num_time_bins,
    )
    if len(flatten_spectral_time_segments(tb_list)) < 2:
        return None
    return tb_list


def parse_background_interval_tuple(background_interval: str) -> Tuple[str, ...]:
    """
    将光谱流程中的本底字符串（逗号分隔段，如 ``-24--5,100-150``）转为
    ``set_background_interval`` 所需的若干参数。
    """
    parts = [p.strip() for p in str(background_interval).split(",") if p.strip()]
    if not parts:
        raise ValueError(f"无效的本底区间字符串: {background_interval!r}")
    return tuple(parts)


def detectors_for_lightcurve(dets: Sequence[str]) -> Tuple[Tuple[str, str], str]:
    """
    从 :func:`select_gbm_detectors` / ``gbm_selector`` 返回的探测器列表中取前两个 NaI
    与一个 BGO（小写 id）。若仅有一个 NaI，则两个槽位使用同一探测器（平均率退化为单探头曲线）。
    """
    nais = [
        str(d).strip().lower()
        for d in dets
        if str(d).strip().lower().startswith("n")
    ]
    bgos = [
        str(d).strip().lower()
        for d in dets
        if str(d).strip().lower().startswith("b")
    ]
    if not nais:
        raise ValueError("探测器列表中无 NaI（n*）")
    if not bgos:
        raise ValueError("探测器列表中无 BGO（b*）")
    if len(nais) == 1:
        pair: Tuple[str, str] = (nais[0], nais[0])
    else:
        pair = (nais[0], nais[1])
    return pair, bgos[0]


def read_trigger_met_and_grb_name(bnname: str, fermilat_xls: Optional[str] = None) -> Tuple[float, str]:
    """从 fermilat GCN 表读取 ``trigger_met`` 与 ``gcn_name``（去空格）。"""
    path = fermilat_xls or session.fermilat_grb_xls
    df = pd.read_excel(path, sheet_name="GCN", index_col="trigname")
    row = df.loc[bnname]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    trigger_met = float(row["trigger_met"])
    grb_name = re.sub(r"\s+", "", str(row["gcn_name"]))
    return trigger_met, grb_name


def read_special_burst_time_segments(bnname: str, yaml_path: Optional[Union[str, Path]] = None) -> Optional[List[dict]]:
    """从 ``special_bursts.yaml`` 读取当前 ``bnname`` 的时间分段。"""
    path = Path(yaml_path or SPECIAL_BURSTS_YAML)
    if not path.exists():
        return None
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    bursts = data.get("special_bursts", []) if isinstance(data, dict) else []
    bn = str(bnname).strip()
    for burst in bursts:
        if str(burst.get("bnname", "")).strip() != bn:
            continue
        segs = burst.get("time_segments", []) or []
        normalized: List[dict] = []
        for idx, seg in enumerate(segs):
            try:
                t0 = float(seg.get("start"))
                t1 = float(seg.get("stop"))
            except Exception:
                continue
            if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
                continue
            normalized.append({"tstart": t0, "tstop": t1, "tag": str(seg.get("name", f"seg{idx + 1}"))})
        return normalized or None
    return None


def read_activity_interval_from_catalog(bnname: str, catalog_xls: Optional[str] = None) -> Optional[Tuple[float, float]]:
    """从 GBM 目录表读取活动时间窗：优先用 ``t90_start`` 与 ``t90_start + t90``。"""
    path = catalog_xls or session.catalog_xls
    try:
        df = pd.read_excel(path, sheet_name="fermigbrst")
    except Exception:
        return None
    bn_col = "trigger_name" if "trigger_name" in df.columns else ("bnname" if "bnname" in df.columns else None)
    if bn_col is None or "t90_start" not in df.columns or "t90" not in df.columns:
        return None
    row = df[df[bn_col].astype(str).str.strip() == str(bnname).strip()]
    if row.empty:
        return None
    first = row.iloc[0]
    try:
        t0 = float(first["t90_start"])
        t95 = float(first["t90_start"]) + float(first["t90"])
    except Exception:
        return None
    if not np.isfinite(t0) or not np.isfinite(t95):
        return None
    if t95 <= t0:
        return None
    if t0 < 0:
        t0 = 0.0
    return t0, t95


def resolve_lightcurve_display_window(
    active_interval: str,
    pad_before_s: float = 1.0,
    pad_after_s: Optional[float] = None,
) -> Optional[Tuple[float, float]]:
    """把 active interval 转成更适合出图的观测窗口。"""
    parsed = _parse_source_interval_pair(active_interval)
    if parsed is None:
        return None
    start, stop = parsed
    before = float(pad_before_s if pad_before_s is not None else 0.0)
    after = float(pad_after_s if pad_after_s is not None else 0.0)
    start = float(np.floor(start) - before)
    stop = float(np.ceil(stop) + after)
    if not np.isfinite(start) or not np.isfinite(stop) or stop <= start:
        return None
    return start, stop


def resolve_gbm_tte_rsp(grb_dir: Union[str, Path], det: str) -> Tuple[Path, Path]:
    """在暴目录下匹配 ``glg_tte_{det}_*.fit`` 与 ``glg_cspec_{det}_*.rsp2``。"""
    d = Path(grb_dir)
    ttes = sorted(d.glob(f"glg_tte_{det}_*.fit"))
    rsps = sorted(d.glob(f"glg_cspec_{det}_*.rsp2")) or sorted(
        d.glob(f"glg_cspec_{det}_*.rsp")
    )
    if not ttes:
        raise FileNotFoundError(f"{d}: 未找到 glg_tte_{det}_*.fit")
    if not rsps:
        raise FileNotFoundError(f"{d}: 未找到 glg_cspec_{det}_*.rsp2/.rsp")
    return ttes[0], rsps[0]


def resolve_gbm_cspec(grb_dir: Union[str, Path], det: str) -> Path:
    d = Path(grb_dir)
    cspecs = sorted(d.glob(f"glg_cspec_{det}_*.pha"))
    if not cspecs:
        raise FileNotFoundError(f"{d}: 未找到 glg_cspec_{det}_*.pha")
    return cspecs[0]


def discover_lat_prob_fit_files(result_bn_dir: Union[str, Path]) -> List[Path]:
    """
    与 ``lat_extended_three_ml`` 输出一致：在单次暴结果目录下查找
    ``interval{tstart}-{tstop}/gll_ft1_tr_bn*_v00_filt_prob.fit``。
    """
    d = Path(result_bn_dir)
    if not d.is_dir():
        return []
    return sorted(d.glob("interval*/gll_ft1_tr_bn*_v00_filt_prob.fit"))


def load_lat_ft1_prob_events(
    prob_files: Sequence[Union[str, Path]],
    trigger_met: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从 ``filt_prob.fit`` 的 EVENTS 读取相对触发时间 [s]、能量 [MeV]、GRB 概率列（与
    ``lat_extended_three_ml`` 中 ``events["GRB"]`` 一致）。
    """
    if not prob_files:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )
    rel_list: List[np.ndarray] = []
    en_list: List[np.ndarray] = []
    pr_list: List[np.ndarray] = []
    tm = float(trigger_met)
    for pf in prob_files:
        with fits.open(str(pf)) as hdul:
            events = hdul["EVENTS"].data
        rel_list.append(np.asarray(events["TIME"], dtype=float) - tm)
        en_list.append(np.asarray(events["ENERGY"], dtype=float))
        pr_list.append(np.asarray(events["GRB"], dtype=float))
    return (
        np.concatenate(rel_list),
        np.concatenate(en_list),
        np.concatenate(pr_list),
    )


def flatten_spectral_time_segments(
    time_bins_list: Sequence[Union[np.ndarray, Sequence[float]]],
) -> List[Tuple[float, float]]:
    """将 ``_build_time_bins_list`` 的边序列列表展平为若干 ``(tstart, tstop)``。"""
    segments: List[Tuple[float, float]] = []
    for tb in time_bins_list:
        edges = np.asarray(tb, dtype=float).ravel()
        if edges.size < 2:
            continue
        for i in range(len(edges) - 1):
            t0, t1 = float(edges[i]), float(edges[i + 1])
            if t1 > t0:
                segments.append((t0, t1))
    return segments


def _spectral_bin_label(index: int) -> str:
    """第 index 个时间 bin 的标签：0→a, 1→b, …, 25→z, 更大则用序号。"""
    if index < 26:
        return chr(ord("a") + index)
    return str(index + 1)


def annotate_spectral_time_bins(
    ax_list,
    time_bins_list: Sequence[Union[np.ndarray, Sequence[float]]],
    *,
    line_color: str = "#2e7d32",
    line_style: str = "--",
    line_width: float = 0.9,
    line_alpha: float = 0.78,
    label_color: str = "#1b5e20",
    label_y_axes: float = 0.97,
    min_bins: int = 2,
) -> int:
    """
    在光变子图上用竖直虚线标出光谱分析各时间 bin，并在段顶居中标注 a/b/c…。

    ``time_bins_list`` 与 :func:`~grb_project.gbm_core._build_time_bins_list` 返回值一致。
    总 bin 数小于 ``min_bins`` 时不绘制（单段无需分割）。
    返回实际标注的 bin 个数。
    """
    segments = flatten_spectral_time_segments(time_bins_list)
    if len(segments) < min_bins:
        return 0

    boundaries = sorted({t for t0, t1 in segments for t in (t0, t1)})
    for ax in ax_list:
        for t in boundaries:
            ax.axvline(
                t,
                color=line_color,
                ls=line_style,
                lw=line_width,
                alpha=line_alpha,
                zorder=4,
            )
        for idx, (t0, t1) in enumerate(segments):
            ax.text(
                0.5 * (t0 + t1),
                label_y_axes,
                _spectral_bin_label(idx),
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=LIGHTCURVE_ANNOTATION_SIZE,
                color=label_color,
                zorder=5,
                clip_on=True,
            )
    return len(segments)


def shade_active_interval(
    ax_list,
    tsb_ref: TimeSeriesBuilder,
    active_interval: Optional[str] = None,
    face: str = "#c8e6c9",
    alpha: float = 0.5,
    line: str = "#c62828",
) -> Optional[np.ndarray]:
    """优先按传入 ``active_interval`` 高亮；否则回退到 ``set_active_time_interval``。

    返回实际使用的区间数组，便于图下注释与界面文案保持一致。
    """
    sel: Optional[np.ndarray] = None
    parsed = _parse_source_interval_pair(active_interval) if active_interval is not None else None
    if parsed is not None:
        sel = np.asarray([parsed], dtype=float)
    else:
        el = tsb_ref.time_series
        if el.time_intervals is not None:
            sel = np.asarray(el.time_intervals.bin_stack, dtype=float).copy()
            np.round(sel, decimals=4, out=sel)
    if sel is None:
        return None
    for ax in ax_list:
        for t0, t1 in sel:
            ax.axvspan(
                t0,
                t1,
                facecolor=face,
                edgecolor="none",
                alpha=alpha,
                zorder=-200,
            )
            ax.axvline(t0, color=line, ls="--", lw=0.9, alpha=0.75, zorder=-199)
            ax.axvline(t1, color=line, ls="--", lw=0.9, alpha=0.75, zorder=-199)
    return sel


def _special_segment_label(index: int, seg: dict) -> str:
    if index < 26:
        return chr(ord("a") + index)
    return str(index + 1)


def _special_segment_bounds(seg: dict) -> Optional[Tuple[float, float]]:
    """兼容 ``tstart``/``tstop`` 与 ``start``/``stop`` 两种特殊分段格式。"""
    try:
        if "tstart" in seg or "tstop" in seg:
            t0 = float(seg.get("tstart"))
            t1 = float(seg.get("tstop"))
        else:
            t0 = float(seg.get("start"))
            t1 = float(seg.get("stop"))
    except Exception:
        return None
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        return None
    return t0, t1


def _special_segment_labels(time_segments: Optional[Sequence[dict]]) -> List[str]:
    labels: List[str] = []
    if not time_segments:
        return labels
    for seg in time_segments:
        bounds = _special_segment_bounds(seg)
        if bounds is None:
            continue
        labels.append(_special_segment_label(len(labels), seg))
    return labels


def shade_time_segments(
    ax_list,
    time_segments: Optional[Sequence[dict]],
    *,
    line_color: str = "#546e7a",
    line_width: float = 0.8,
    label_y_axes: float = 0.985,
) -> int:
    """把特殊分段画成边界线，并在段顶标注分段名。

    按当前图示约定，只强调分段边界；不为未选中的时间段添加底色。
    同时兼容 ``tstart``/``tstop`` 与 ``start``/``stop`` 两种输入格式。
    """
    if not time_segments:
        return 0
    count = 0
    visible_idx = 0
    for seg in time_segments:
        bounds = _special_segment_bounds(seg)
        if bounds is None:
            continue
        t0, t1 = bounds
        label = _special_segment_label(visible_idx, seg)
        visible_idx += 1
        for ax in ax_list:
            ax.axvspan(
                t0,
                t1,
                facecolor=line_color,
                edgecolor="none",
                alpha=0.10,
                zorder=2,
            )
            ax.axvline(t0, color=line_color, lw=line_width, ls=":", alpha=0.95, zorder=3)
            ax.axvline(t1, color=line_color, lw=line_width, ls=":", alpha=0.95, zorder=3)
            ax.text(
                0.5 * (t0 + t1),
                label_y_axes,
                label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=LIGHTCURVE_ANNOTATION_SIZE,
                color=line_color,
                zorder=6,
                clip_on=True,
            )
        count += 1
    return count


def kev_band_to_echan(tte_path: Union[str, Path], emin_kev: float, emax_kev: float) -> Tuple[int, int]:
    """EBOUNDS 与 [emin, emax] keV 重叠的 PHA 道（闭区间索引）。"""
    with fits.open(str(tte_path)) as hdul:
        eb = hdul["EBOUNDS"].data
    ov = (eb["E_MAX"] > emin_kev) & (eb["E_MIN"] < emax_kev)
    ch = np.where(ov)[0]
    if ch.size == 0:
        raise ValueError(f"No EBOUNDS channels overlap [{emin_kev}, {emax_kev}] keV")
    return int(ch[0]), int(ch[-1])


def _resolve_echan(el, use_echans_start: int, use_echans_stop: int) -> Tuple[int, int]:
    n_ch = el.n_channels
    es, ee = use_echans_start, use_echans_stop
    if es < 0:
        es = n_ch + es
    if ee < 0:
        ee = n_ch + ee
    return es, ee


def lightcurve_rate(
    tsb: TimeSeriesBuilder,
    start: float,
    stop: float,
    dt: float,
    use_echans_start: int,
    use_echans_stop: int,
):
    """与 EventList 线性分箱一致：返回 time_bins, rate, bkg_rate 或 ``bkg_rate is None``。"""
    el = tsb.time_series
    es, ee = _resolve_echan(el, use_echans_start, use_echans_stop)
    echan_bins = np.arange(es, ee + 2, 1) - 0.5
    bins = np.arange(start, stop + dt, dt)
    cnts, bins_arr, _ = np.histogram2d(
        el.arrival_times, el.measurement, bins=(bins, echan_bins)
    )
    cnts = np.sum(cnts, axis=1)
    time_bins = np.column_stack((bins_arr[:-1], bins_arr[1:]))
    width = np.array([el.exposure_over_interval(t0, t1) for t0, t1 in time_bins])
    rate = np.divide(cnts, width, out=np.zeros_like(cnts, dtype=float), where=width > 0)
    if not el.poly_fit_exists:
        return time_bins, rate, None
    bkg = np.empty(len(time_bins))
    for j, (t0, t1) in enumerate(time_bins):
        tmp = sum(p.integral(t0, t1) for p in el.polynomials[es : ee + 1])
        bkg[j] = tmp / (t1 - t0)
    return time_bins, rate, bkg


def plot_mean_lightcurve_on_ax(
    tsb_list,
    ax,
    start: float,
    stop: float,
    dt: float,
    use_echans_start: int,
    use_echans_stop: int,
    title: str = "",
) -> Dict[str, Any]:
    """多探头平均率；本底选中时段 step 填充；需在图外另行 ``shade_active_interval``。"""
    rates = []
    bkgs = []
    time_bins = None
    for tsb in tsb_list:
        tb, r, bkg = lightcurve_rate(
            tsb, start, stop, dt, use_echans_start, use_echans_stop
        )
        if time_bins is None:
            time_bins = tb
        elif not np.allclose(tb, time_bins):
            raise ValueError("各探头时间分箱不一致（检查 start/stop/dt）")
        rates.append(r)
        bkgs.append(bkg)
    mean_rate = np.mean(rates, axis=0)
    np.round(time_bins, decimals=4, out=time_bins)

    positive = mean_rate > 0
    bottom = float(mean_rate[positive].min()) * 0.95 if np.any(positive) else 0.0

    lc_color = threeML_config.time_series.light_curve_color
    bkg_sel_color = threeML_config.time_series.background_selection_color
    bkg_line_color = threeML_config.time_series.background_color

    step_plot(time_bins, mean_rate, ax, color=lc_color)

    el0 = tsb_list[0].time_series

    if el0.bkg_intervals is not None:
        bkg_selections = np.asarray(el0.bkg_intervals.bin_stack, dtype=float).copy()
        np.round(bkg_selections, decimals=4, out=bkg_selections)
        masks_b = []
        for tmin, tmax in bkg_selections:
            masks_b.append(
                np.logical_and(time_bins[:, 0] >= tmin, time_bins[:, 1] <= tmax)
            )
        if masks_b:
            if len(masks_b) > 1:
                for mk in masks_b[1:]:
                    step_plot(
                        time_bins[mk],
                        mean_rate[mk],
                        ax,
                        color=bkg_sel_color,
                        fill=True,
                        alpha=0.4,
                        fill_min=bottom,
                        zorder=-30,
                    )
            step_plot(
                time_bins[masks_b[0]],
                mean_rate[masks_b[0]],
                ax,
                color=bkg_sel_color,
                fill=True,
                alpha=0.4,
                fill_min=bottom,
                label="Bkg. Selections",
                zorder=-30,
            )

    if all(b is not None for b in bkgs):
        ax.plot(
            np.mean(time_bins, axis=1),
            np.mean(np.stack(bkgs, axis=0), axis=0),
            color=bkg_line_color,
            lw=1.0,
            label="Background",
        )

    ax.set_ylabel("Rate (cnts/s)")
    ax.text(
        0.98,
        PANEL_TITLE_Y,
        title,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=LIGHTCURVE_ANNOTATION_SIZE,
        bbox=TITLE_BOX,
    )
    style_lightcurve_axes(ax, show_all_spines=True)
    ax.set_xlim(float(start), float(stop))
    w0 = np.array(
        [tsb_list[0].time_series.exposure_over_interval(t0, t1) for t0, t1 in time_bins]
    )
    m = w0 > 0
    if np.any(m):
        mr = mean_rate[m]
        ax.set_ylim(max(0, float(mr.min()) * 0.9), float(mr.max()) * 1.15)
    mean_background = np.mean(np.stack(bkgs, axis=0), axis=0) if all(b is not None for b in bkgs) else None
    return {
        "title": str(title),
        "time": np.mean(time_bins, axis=1),
        "rate": mean_rate,
        "background": mean_background,
    }


def plot_gbm_lat_lightcurve_figure(
    bnname: str,
    *,
    grb_name: Optional[str] = None,
    trigger_met: Optional[float] = None,
    data_dir: Optional[str] = None,
    lat_prob_bn_dir: Optional[Union[str, Path]] = None,
    lat_prob_threshold: float = 0.9,
    gbm_start: Optional[float] = None,
    gbm_stop: Optional[float] = None,
    gbm_dt: float = 0.1,
    lat_bin_s: float = 0.2,
    lat_emin_mev: float = 100.0,
    nai_detector_ids: Optional[Tuple[str, str]] = None,
    bgo_detector_id: Optional[str] = None,
    active_interval: str = "0.1-8",
    background_intervals: Sequence[str] = ("-24--5", "350-400"),
    background_unbinned: bool = False,
    bands_kev: Sequence[Tuple[float, float]] = ((8.0, 50.0), (50.0, 300.0)),
    nai_bands_kev: Optional[Sequence[Tuple[float, float]]] = None,
    display_window_pad_before_s: float = 1.0,
    display_window_pad_after_s: Optional[float] = 1.0,
    bgo_band_kev: Tuple[float, float] = (300.0, 38000.0),
    band_titles: Optional[Sequence[str]] = None,
    out_path: Optional[Union[str, Path]] = None,
    figure_size: Optional[Tuple[float, float]] = None,
    dpi: int = RASTER_DPI,
    include_lat: bool = True,
    spectral_time_bins: Optional[Sequence[Union[np.ndarray, Sequence[float]]]] = None,
    special_time_segments: Optional[Sequence[dict]] = None,
    fixed_num_time_bins: Optional[int] = None,
) -> Figure:
    """
    绘制光变示意图：两个 NaI 能段均值、BGO；可选第四 panel 为 LAT。
    横轴范围为 ``gbm_start``–``gbm_stop``；源时选段由 ``active_interval`` 标出。

    Parameters
    ----------
    include_lat
        为 False 时仅三幅 GBM 子图，不读取 LAT。
    lat_prob_bn_dir
        含 ``interval*/gll_ft1_tr_bn*_v00_filt_prob.fit`` 的目录（与 ``lat_extended_three_ml``
        在 ``{result_root}/{grb_name}/{bn_name}/`` 下产物一致）。默认使用
        ``session.result_root / grb_name / bnname``。
        若该目录下无 prob 文件，则不绘制 LAT 子图（退化为三幅 GBM），并写一条日志提示。
    lat_prob_threshold
        ``GRB`` 概率列大于等于该阈值时散点为黑色实心圆；严格小于该阈值时仅为黑色空心圆。
    nai_detector_ids, bgo_detector_id
        若为 ``None``，则与光谱流程一致：在 ``grb_dir`` 下调用
        :func:`~grb_project.gbm_detector_selection.select_gbm_detectors` 再经
        :func:`detectors_for_lightcurve` 得到两个 NaI 与一个 BGO。可只覆盖其中一项，未
        给定的部分仍自动选择。
    bands_kev, nai_bands_kev
        前两幅子图为各 NaI 能段的平均率；``bands_kev`` 为默认 NaI 能段列表。
        若显式传入 ``nai_bands_kev``，则以其覆盖 ``bands_kev``（便于单独配置 NaI）。
    bgo_band_kev
        BGO 子图能量范围 [keV]，由 TTE 的 EBOUNDS 映射为道址；默认 ``(300, 38000)``。
    out_path
        若给定则 ``savefig``；默认 ``{grb_name}_Lightcurve.png`` 保存在当前工作目录。
        同时生成同名 ``.html`` 交互图，可离线打开并悬停查看数据点。
    spectral_time_bins
        与光谱拟合一致的时间 bin 边序列列表（``_build_time_bins_list`` 的返回值）。
        当总段数 ≥ 2 时，在各 GBM/LAT 子图上画绿色竖直虚线并在段顶标注 a/b/c…。
        为 ``None`` 时，若 ``active_interval`` 可解析为 ``t0-t1``，则按 ``bnname`` 调用
        ``_build_time_bins_list`` 自动推断（与 ``analyze_single`` 一致）。
    special_time_segments
        来自 ``special_bursts.yaml`` 的特殊时间分段；若给定，将以分割线和标签叠加到
        所有子图上，用于突出显示手工分段。不会为未选中的时间段添加底色。
    fixed_num_time_bins
        传给 ``_build_time_bins_list`` 的固定分 bin 数；仅影响自动推断分段。
    """
    ensure_analysis_runtime()

    # 允许 web/CLI 调用方传入 None，使用与界面默认值一致的兜底范围。
    if gbm_start is None or gbm_stop is None:
        display_window = resolve_lightcurve_display_window(
            active_interval,
            display_window_pad_before_s,
            display_window_pad_after_s,
        )
        if display_window is not None:
            display_start, display_stop = display_window
            gbm_start = display_start if gbm_start is None else gbm_start
            gbm_stop = display_stop if gbm_stop is None else gbm_stop
        else:
            catalog_interval = read_activity_interval_from_catalog(bnname)
            if catalog_interval is None:
                parsed = _parse_source_interval_pair(active_interval)
                if parsed is not None:
                    catalog_interval = parsed
            if catalog_interval is not None:
                active_t0, active_t1 = catalog_interval
                gbm_start = active_t0 - 2.0 if gbm_start is None else gbm_start
                gbm_stop = active_t1 + 2.0 if gbm_stop is None else gbm_stop
            else:
                if gbm_start is None:
                    gbm_start = -2.0
                if gbm_stop is None:
                    gbm_stop = 20.0
    if lat_bin_s is None:
        lat_bin_s = 0.2
    gbm_start = float(gbm_start)
    gbm_stop = float(gbm_stop)
    lat_bin_s = float(lat_bin_s)

    base_data = Path(data_dir or session.data_dir)
    grb_dir = base_data / bnname

    _nai = nai_detector_ids
    _bgo = bgo_detector_id
    if _nai is None or _bgo is None:
        dets, *_ = select_gbm_detectors(str(grb_dir))
        auto_nai, auto_bgo = detectors_for_lightcurve(dets)
        if _nai is None:
            _nai = auto_nai
        if _bgo is None:
            _bgo = auto_bgo
    nai_detector_ids = _nai
    bgo_detector_id = _bgo

    if trigger_met is None or grb_name is None:
        tm, gn = read_trigger_met_and_grb_name(bnname)
        if trigger_met is None:
            trigger_met = tm
        if grb_name is None:
            grb_name = gn

    lat_time_rel: Optional[np.ndarray] = None
    lat_energy_mev: Optional[np.ndarray] = None
    lat_prob: Optional[np.ndarray] = None
    lat_show_panel = False
    if include_lat:
        prob_base = (
            Path(lat_prob_bn_dir)
            if lat_prob_bn_dir is not None
            else Path(session.result_root) / str(grb_name) / str(bnname)
        )
        prob_files = discover_lat_prob_fit_files(prob_base)
        if prob_files:
            lat_time_rel, lat_energy_mev, lat_prob = load_lat_ft1_prob_events(
                prob_files, float(trigger_met)
            )
            lat_show_panel = True
        else:
            log(
                f"{bnname}: 光变 LAT 子图跳过（未在 {prob_base} 找到 "
                "interval*/gll_ft1_tr_bn*_v00_filt_prob.fit；请先跑 LAT Extended 流水线）"
            )

    ref_tte, _ = resolve_gbm_tte_rsp(grb_dir, nai_detector_ids[0])

    builders: List[TimeSeriesBuilder] = []
    for i, nid in enumerate(nai_detector_ids):
        tte_p, rsp_p = resolve_gbm_tte_rsp(grb_dir, nid)
        builders.append(
            TimeSeriesBuilder.from_gbm_tte(f"nai_{nid}", tte_file=str(tte_p), rsp_file=str(rsp_p))
        )

    tb_bgo_tte, tb_bgo_rsp = resolve_gbm_tte_rsp(grb_dir, bgo_detector_id)
    bgo_b = TimeSeriesBuilder.from_gbm_tte(
        f"bgo_{bgo_detector_id}",
        tte_file=str(tb_bgo_tte),
        rsp_file=str(tb_bgo_rsp),
    )

    nai_kev_bands: Sequence[Tuple[float, float]] = (
        tuple((float(lo), float(hi)) for lo, hi in nai_bands_kev)
        if nai_bands_kev is not None
        else tuple((float(lo), float(hi)) for lo, hi in bands_kev)
    )
    bgo_lo, bgo_hi = float(bgo_band_kev[0]), float(bgo_band_kev[1])
    if bgo_lo >= bgo_hi:
        raise ValueError(f"bgo_band_kev 须满足 emin < emax，当前为 {bgo_band_kev!r}")
    bgo_es, bgo_ee = kev_band_to_echan(tb_bgo_tte, bgo_lo, bgo_hi)

    if special_time_segments is None:
        special_time_segments = read_special_burst_time_segments(bnname)
    catalog_interval = read_activity_interval_from_catalog(bnname)
    active_interval = resolve_active_interval_from_special_and_catalog(
        special_time_segments,
        None if catalog_interval is None else pd.Series({"t90_start": catalog_interval[0], "t90": catalog_interval[1] - catalog_interval[0]}),
        active_interval,
    )

    if gbm_start is None:
        if special_time_segments:
            gbm_start = float(special_time_segments[0]["tstart"]) - 2.0
        elif catalog_interval is not None:
            gbm_start = float(catalog_interval[0]) - 2.0
        else:
            gbm_start = -2.0

    for tsb in (*builders, bgo_b):
        tsb.set_active_time_interval(active_interval)
        tsb.set_background_interval(*background_intervals, unbinned=background_unbinned)

    nai_builders = builders
    lat_bins = np.arange(float(gbm_start), float(gbm_stop) + float(lat_bin_s), float(lat_bin_s))
    lat_plot: Optional[np.ndarray] = None
    if lat_show_panel and lat_time_rel is not None and lat_energy_mev is not None:
        lat_win = (lat_time_rel >= gbm_start) & (lat_time_rel <= gbm_stop)
        lat_e_ok = lat_energy_mev >= lat_emin_mev
        lat_plot = lat_win & lat_e_ok

    titles_default = tuple(
        f"NaI {lo:g}–{hi:g} keV" for lo, hi in nai_kev_bands
    )
    if band_titles is not None:
        titles_use = tuple(band_titles)
        if len(titles_use) != len(nai_kev_bands):
            raise ValueError("band_titles 长度须与 NaI 能段条数一致（与 nai_bands_kev 或 bands_kev）")
    else:
        titles_use = titles_default

    n_rows = 4 if lat_show_panel else 3
    fig_wh = figure_size
    if fig_wh is None:
        fig_wh = LIGHTCURVE_FIGSIZE if lat_show_panel else (LIGHTCURVE_FIGSIZE[0], 6.45)

    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=fig_wh,
        sharex=True,
        constrained_layout=False,
        gridspec_kw={"hspace": 0.12},
    )
    fig.subplots_adjust(left=0.12, right=0.90, top=0.98, bottom=0.08)
    for _a in axes:
        _a.set_axisbelow(True)
        for _s in _a.spines.values():
            _s.set_visible(True)
            _s.set_linewidth(AXES_LINEWIDTH)

    interactive_gbm_panels: List[Dict[str, Any]] = []
    for ax, (elo, ehi), ttl in zip(axes[:2], nai_kev_bands, titles_use):
        es, ee = kev_band_to_echan(ref_tte, elo, ehi)
        panel_data = plot_mean_lightcurve_on_ax(
            nai_builders,
            ax,
            start=gbm_start,
            stop=gbm_stop,
            dt=gbm_dt,
            use_echans_start=es,
            use_echans_stop=ee,
            title=str(ttl),
        )
        if isinstance(panel_data, dict):
            interactive_gbm_panels.append(panel_data)

    bgo_panel_data = plot_mean_lightcurve_on_ax(
        [bgo_b],
        axes[2],
        start=gbm_start,
        stop=gbm_stop,
        dt=gbm_dt,
        use_echans_start=bgo_es,
        use_echans_stop=bgo_ee,
        title=f"BGO {bgo_detector_id} ({bgo_lo:g}–{bgo_hi:g} keV)",
    )
    if isinstance(bgo_panel_data, dict):
        interactive_gbm_panels.append(bgo_panel_data)

    if lat_show_panel and lat_time_rel is not None and lat_energy_mev is not None and lat_plot is not None:
        ax_lat = axes[3]
        ax_lat.hist(
            lat_time_rel[lat_plot],
            bins=lat_bins,
            histtype="stepfilled",
            alpha=0.25,
            color="C1",
            zorder=1,
        )
        ax_lat.hist(
            lat_time_rel[lat_plot],
            bins=lat_bins,
            histtype="step",
            color="C1",
            zorder=2,
        )
        ax_lat.set_ylabel(f"LAT events / {lat_bin_s:g} s", fontsize=LIGHTCURVE_AXIS_LABEL_SIZE)
        ax_lat.text(
            0.98,
            PANEL_TITLE_Y,
            f"LAT ≥ {lat_emin_mev:g} MeV",
            transform=ax_lat.transAxes,
            ha="right",
            va="top",
            fontsize=LIGHTCURVE_ANNOTATION_SIZE,
            bbox=TITLE_BOX,
        )
        style_lightcurve_axes(ax_lat, show_all_spines=True)
        ax_lat.grid(True, alpha=0.3)

        ax_e = ax_lat.twinx()
        t_plot = lat_time_rel[lat_plot]
        e_plot = lat_energy_mev[lat_plot]
        p_plot = lat_prob[lat_plot]
        hi = p_plot >= float(lat_prob_threshold)
        lo = p_plot < float(lat_prob_threshold)
        if np.any(hi):
            ax_e.scatter(
                t_plot[hi],
                e_plot[hi],
                c="black",
                s=12,
                zorder=6,
                edgecolors="none",
            )
        if np.any(lo):
            ax_e.scatter(
                t_plot[lo],
                e_plot[lo],
                s=50,
                facecolors="none",
                edgecolors="black",
                linewidths=1.2,
                zorder=5,
            )
        ax_e.set_yscale("log")
        if lat_emin_mev > 0:
            ax_e.set_ylim(bottom=lat_emin_mev)
        ax_e.set_ylabel("Energy [MeV]", fontsize=LIGHTCURVE_AXIS_LABEL_SIZE)
        ax_e.tick_params(axis="y", which="major", labelsize=LIGHTCURVE_TICK_LABEL_SIZE)
        ax_e.tick_params(axis="y", which="minor", labelsize=LIGHTCURVE_TICK_LABEL_SIZE)
        bottom_ax = ax_lat
    else:
        bottom_ax = axes[2]

    interval_for_plot = active_interval
    shade_active_interval(list(axes), nai_builders[0], active_interval=interval_for_plot)
    shade_time_segments(list(axes), special_time_segments)

    bins_to_annotate = resolve_spectral_time_bins_for_lightcurve(
        bnname,
        interval_for_plot,
        spectral_time_bins,
        fixed_num_time_bins=fixed_num_time_bins,
    )
    # 当已经绘制了 special_time_segments 时，它们本身会带分段字母标注；
    # 再叠加光谱分 bin 标注会导致 a/b/c... 重复绘制。
    if bins_to_annotate is not None and not special_time_segments:
        annotate_spectral_time_bins(list(axes), bins_to_annotate)

    bottom_ax.set_xlabel("Time − T0 [s]")
    axes[0].set_xlim(float(gbm_start), float(gbm_stop))

    if out_path is not None:
        save_to = out_path
    else:
        nai_tag = "_".join(
            f"{lo:g}-{hi:g}".replace(".", "p") for lo, hi in nai_kev_bands
        )
        save_to = f"{grb_name}_lightcurve_{nai_tag}_{gbm_start:g}_{gbm_stop:g}.png"
    save_publication_figure(fig, save_to, dpi=dpi, bbox_inches="tight")
    interactive_lat_data: Optional[Dict[str, np.ndarray]] = None
    if lat_show_panel and lat_time_rel is not None and lat_energy_mev is not None and lat_prob is not None and lat_plot is not None:
        lat_counts, lat_edges = np.histogram(lat_time_rel[lat_plot], bins=lat_bins)
        interactive_lat_data = {
            "time": lat_time_rel[lat_plot],
            "energy": lat_energy_mev[lat_plot],
            "probability": lat_prob[lat_plot],
            "bin_edges": lat_edges,
            "counts": lat_counts,
        }
    try:
        write_interactive_lightcurve_html(
            interactive_lightcurve_path(save_to),
            gbm_panels=interactive_gbm_panels,
            lat_data=interactive_lat_data,
            x_range=(float(gbm_start), float(gbm_stop)),
        )
    except Exception as exc:  # noqa: BLE001
        log(f"{bnname}: 交互光变图生成失败，已保留 PNG: {exc}")
    plt.close(fig)

    return fig
