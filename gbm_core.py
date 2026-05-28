# -*- coding: utf-8 -*-
"""GBM 时间窗、本底与单探测器插件构建；探测器几何选择见 :mod:`gbm_detector_selection`。"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threeML import OGIPLike, TimeSeriesBuilder

from .io_utils import find_files
from .logging_utils import log

logg = logging.getLogger("threeML")


def _determine_time_interval_and_position(
    bnname: str,
    df_catalog: pd.DataFrame,
) -> Tuple[float, float, float, float]:
    row = df_catalog.loc[bnname]

    if float(row["t90_start"]) < 0:
        t0 = 0.0
    else:
        t0 = float(row["t90_start"])

    t1 = float(row["t90_start"] + row["t90"])
    ra, dec = float(row["ra"]), float(row["dec"])

    if bnname == "bn250625670":
        t0 = float(0.65)

    return t0, t1, ra, dec


def _determine_time_bins(
    t0: float,
    t1: float,
    fixed_num_time_bins: Optional[int] = None,
) -> Tuple[np.ndarray, int, float]:
    duration = t1 - t0

    fixed_num_time_bins = 1 if fixed_num_time_bins is None else fixed_num_time_bins

    if fixed_num_time_bins is not None and fixed_num_time_bins > 0:
        num_time_bins = fixed_num_time_bins
        time_bins = np.round(np.linspace(t0, t1, num_time_bins + 1), 2)
        log(f"总持续时间: {duration:.2f}s, 使用固定分bin数量: {num_time_bins}")
        return time_bins, num_time_bins, duration

    if duration <= 2:
        num_time_bins = 1
        time_bins = np.round(np.array([t0, t1]), 2)
    elif duration <= 10:
        bin_size = 1.0
        num_time_bins = int(math.ceil(duration / bin_size))
        time_bins = np.round(np.linspace(t0, t1, num_time_bins + 1), 2)
    elif duration <= 100:
        bin_size = 5.0
        num_time_bins = int(math.ceil(duration / bin_size))
        time_bins = np.round(np.linspace(t0, t1, num_time_bins + 1), 2)
    else:
        bin_size = 10.0
        num_time_bins = int(math.ceil(duration / bin_size))
        time_bins = np.round(np.linspace(t0, t1, num_time_bins + 1), 2)

    log(f"总持续时间: {duration:.2f}s, 分为 {num_time_bins} 个时间bin")
    return time_bins, num_time_bins, duration


def _build_time_bins_list(
    bnname: str,
    t0: float,
    t1: float,
    fixed_num_time_bins: Optional[int] = None,
) -> Tuple[List[np.ndarray], int, float]:
    """
    与 :func:`analyze_grb` 一致的时间 bin 边序列列表。

    每个元素是一条 bin 边序列 ``[edge0, edge1, …]``，不是边本身的平铺列表。
    """
    time_bins, num_time_bins, duration = _determine_time_bins(
        t0,
        t1,
        fixed_num_time_bins=fixed_num_time_bins,
    )
    time_bins_list: List[np.ndarray] = [time_bins]

    if bnname == "bn231129799":
        edges_231129 = np.array(
            [0.1, 1, 3, 4.5, 6.2, 8.5],
            dtype=float,
        )
        time_bins_list = [edges_231129]
        num_time_bins = len(edges_231129) - 1
        duration = float(edges_231129[-1] - edges_231129[0])

    if bnname == "bn250313607":
        time_bins_list = [
            np.array([1.09, 3, 7, 10, 15, 25]),
            np.array([260, 270, 276, 285, 299]),
        ]
        num_time_bins = sum(len(tb) - 1 for tb in time_bins_list)
        duration = float(time_bins_list[-1][-1] - time_bins_list[0][0])

    return time_bins_list, num_time_bins, duration


def _build_lat_analysis_segments(
    bnname: str,
    t0: float,
    t1: float,
    fixed_num_time_bins: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    生成 LAT 瞬态分析时段：各 GBM 时间 bin + 每段边序列的全程区间。

    返回元素含 ``tstart``、``tstop``、``tag``（用于输出文件名后缀）。
    """
    time_bins_list, _, _ = _build_time_bins_list(
        bnname,
        t0,
        t1,
        fixed_num_time_bins=fixed_num_time_bins,
    )

    def _fmt_time(t: float) -> str:
        return f"{float(t):.2f}".rstrip("0").rstrip(".")

    segments: List[Dict[str, Any]] = []
    seen: set[tuple[float, float, str]] = set()

    def _add(tstart: float, tstop: float, tag: str) -> None:
        key = (round(tstart, 6), round(tstop, 6), tag)
        if key in seen:
            return
        seen.add(key)
        segments.append(
            {
                "tstart": float(tstart),
                "tstop": float(tstop),
                "tag": tag,
            }
        )

    for block_idx, tb in enumerate(time_bins_list):
        for i in range(len(tb) - 1):
            t0_i, t1_i = float(tb[i]), float(tb[i + 1])
            _add(
                t0_i,
                t1_i,
                f"bin_{_fmt_time(t0_i)}_{_fmt_time(t1_i)}",
            )
        if len(tb) > 2:
            block_tag = (
                f"full_block{block_idx + 1}"
                if len(time_bins_list) > 1
                else "full"
            )
            _add(
                float(tb[0]),
                float(tb[-1]),
                f"{block_tag}_{_fmt_time(tb[0])}_{_fmt_time(tb[-1])}",
            )

    return segments


def _build_background_interval_string(row: pd.Series, bnname: str) -> str:
    t2, t3 = row["back_interval_low_start"], row["back_interval_low_stop"]
    t4, t5 = row["back_interval_high_start"], row["back_interval_high_stop"]

    if bnname == "bn221023862":
        t2, t3 = float(-130), float(-10)
        t4, t5 = float(100), float(200)

    background_interval = f"{t2}-{t3},{t4}-{t5}"
    if bnname == "bn250313607":
        background_interval = "-24--5,100-150,350-400"
    if bnname == "bn220921462":
        background_interval = "-23.960--2.080,75-100,140-160"

    return background_interval


def _build_gbm_plugin_for_detector(
    det: str,
    grb_dir: str,
    background_interval: str,
    bin_start: float,
    bin_end: float,
    source_interval: str,
    time_series: Dict[str, TimeSeriesBuilder],
) -> Optional[OGIPLike]:
    rsp = ""
    tte = ""
    cspec = ""
    try:
        rsp = next(
            f for f in find_files(grb_dir, ".rsp2") if f"_cspec_{det}_" in f
        )
        tte = next(
            f for f in find_files(grb_dir, ".fit") if f"_tte_{det}_" in f
        )
        cspec = next(
            f for f in find_files(grb_dir, ".pha") if f"_cspec_{det}_" in f
        )
    except StopIteration as exc:  # noqa: BLE001
        log(f"警告: 未能找到 {det} 的必要文件: {exc}")
        log(f"rsp: {rsp}")
        log(f"tte: {tte}")
        log(f"cspec: {cspec}")
        return None

    try:
        ts_cspec = TimeSeriesBuilder.from_gbm_cspec_or_ctime(
            det,
            cspec_or_ctime_file=cspec,
            rsp_file=rsp,
        )
        ts_cspec.set_background_interval(*background_interval.split(","))
        ts_cspec.save_background(f"{det}_bkg.h5", overwrite=True)
    except Exception as exc:  # noqa: BLE001
        log(f"警告: 为 {det} 设置 CSPEC 本底失败: {exc}")
        return None

    try:
        ts_tte = TimeSeriesBuilder.from_gbm_tte(
            det,
            tte_file=tte,
            rsp_file=rsp,
            restore_background=f"{det}_bkg.h5",
            poly_order=-1,
        )
        ts_tte.set_background_interval(*background_interval.split(","))
        time_series[det] = ts_tte
    except Exception as exc:  # noqa: BLE001
        log(f"警告: 构建 {det} 的 TTE TimeSeries 失败: {exc}")
        return None

    try:
        print("时间间隔为：", source_interval, f"{bin_start:.2f}-{bin_end:.2f}")
        ts_tte.set_active_time_interval(source_interval)
    except Exception as exc:  # noqa: BLE001
        log(
            "警告: 无法为探测器 "
            f"{det} 设置活动时间间隔 {source_interval}: {exc}"
        )

    try:
        fig = ts_tte.view_lightcurve(bin_start, bin_end, dt=0.02)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        log(f"无法创建时间序列图，跳过探测器 {det}: {exc}")

    try:
        ts_tte.create_time_bins(
            start=[bin_start],
            stop=[bin_end],
            method="custom",
        )
        logg.info("创建时间bins: %.2f-%.2f", bin_start, bin_end)
    except Exception as exc:  # noqa: BLE001
        log(f"警告: 无法为探测器 {det} 创建时间bins: {exc}")
        return None

    try:
        ts_tte.write_pha_from_binner(
            file_name=f"gbm_tte_{det}",
            overwrite=True,
            force_rsp_write=True,
        )
    except Exception as exc:  # noqa: BLE001
        log(f"警告: 无法为探测器 {det} 写入PHA文件: {exc}")
        return None

    try:
        plugin = ts_tte.to_spectrumlike()

        if det.startswith("b"):
            plugin.set_active_measurements("900-30000")
        else:
            plugin.set_active_measurements("40-900")

        plugin.rebin_on_background(1.0)

        return plugin
    except Exception as exc:  # noqa: BLE001
        log(f"警告: 无法为探测器 {det} 创建插件: {exc}")
        return None
