# -*- coding: utf-8 -*-
"""
LAT Extended + GtBurst + threeML 全流程（由原 ``3MLprogram/new.py`` 迁入）。

工作目录须设为 ``结果目录/{grb_name}/{bn_name}/``（由 :func:`run_lat_extended_three_ml_pipeline` 切换 cwd）。
耗时较长，默认仅通过会话或 ``GRBRunOverrides.lat_three_ml_full`` 开启。
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, Iterator, Optional

import numpy as np
import pandas as pd
from astropy.io import fits as pyfits
from astropy import units as u

from threeML import (
    DataList,
    JointLikelihood,
    Model,
    PointSource,
    Powerlaw_flux,
    display_spectrum_model_counts,
    plot_spectra,
)
from threeML.io import update_logging_level
from threeML.utils.data_builders import TransientLATDataBuilder
from threeML.utils.data_download.Fermi_LAT.download_LAT_data import LAT_dataset

from GtBurst.dataHandling import _makeDatasetsOutOfLATdata

from .logging_utils import log
from .runtime_env import ensure_analysis_runtime

# 相对 ``analyze_single`` 的 ``[t0,t1]``，LAT 流水线在时间轴两侧各多取的秒数（建库 / 分档等）
_LAT_PIPELINE_TIME_PAD_S = 5.0


@contextlib.contextmanager
def _working_directory(path: str) -> Iterator[None]:
    prev = os.getcwd()
    os.makedirs(path, exist_ok=True)
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(prev)


@contextlib.contextmanager
def _silence_process_stdio() -> Iterator[None]:
    """屏蔽标准输出/错误（含子进程继承的 fd），用于 ``extract_events`` 等嘈杂步骤。"""
    sys.stdout.flush()
    sys.stderr.flush()
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_out = os.dup(1)
    saved_err = os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        os.close(devnull_fd)


def _resolve_ft_paths(
    grb_dir: str,
    bn_name: str,
    extended_data_dir: str,
) -> tuple[str, str]:
    """按 ``new.py`` 逻辑解析 FT1/FT2（含 EV00/SC00、FT1/SC00）。"""
    ft1_file = os.path.join(grb_dir, f"gll_ft1_tr_{bn_name}_v00.fit")
    ft2_file = os.path.join(grb_dir, f"gll_ft2_tr_{bn_name}_v00.fit")

    if os.path.exists(ft1_file) and os.path.exists(ft2_file):
        return ft1_file, ft2_file

    if not os.path.isdir(extended_data_dir):
        raise FileNotFoundError(
            f"缺少标准 gll 文件且 Extended 目录不存在: {extended_data_dir}"
        )

    names = os.listdir(extended_data_dir)
    ev_files = [
        f
        for f in names
        if f.endswith("EV00.fits") and f.startswith("L")
    ]
    sc_files = [
        f
        for f in names
        if f.endswith("SC00.fits") and f.startswith("L")
    ]
    ft1_files = [
        f
        for f in names
        if f.endswith("FT1.fits") and f.startswith("L")
    ]

    if ft1_files and sc_files:
        log(
            "使用 Extended_data_ex 中的 FT1/SC00: "
            f"{ft1_files[0]}, {sc_files[0]}"
        )
        return (
            os.path.join(extended_data_dir, ft1_files[0]),
            os.path.join(extended_data_dir, sc_files[0]),
        )
    if ev_files and sc_files:
        log(
            "使用 Extended_data_ex 中的 EV00/SC00: "
            f"{ev_files[0]}, {sc_files[0]}"
        )
        return (
            os.path.join(extended_data_dir, ev_files[0]),
            os.path.join(extended_data_dir, sc_files[0]),
        )

    raise FileNotFoundError(
        f"在 {extended_data_dir} 未找到 EV00/SC00 或 FT1/SC00 配对文件"
    )


def _analyze_gbm_aligned_time_window(
    event_times_rel_s: np.ndarray,
    energies_mev: np.ndarray,
    t0_s: float,
    t1_s: float,
) -> Dict[str, Any]:
    """
    在相对触发的时间轴上，统计与 ``analyze_single`` 中 ``t0``–``t1`` 重合时段内的 LAT 事例。

    ``t0_s``/``t1_s`` 须与 ``selection["tstart"]`` / ``selection["tstop"]`` 一致（即单次分析里
    所用的源时间窗，相对 MET 触发）。
    """
    t0_s = float(t0_s)
    t1_s = float(t1_s)
    tt = np.asarray(event_times_rel_s, dtype=float)
    ee = np.asarray(energies_mev, dtype=float)
    n_tot = int(tt.size)
    mask = (tt >= t0_s) & (tt <= t1_s)
    n_in = int(np.sum(mask))
    stats: Dict[str, Any] = {
        "analysis_t0_s": t0_s,
        "analysis_t1_s": t1_s,
        "window_duration_s": float(t1_s - t0_s),
        "n_events_total_filtered_file": n_tot,
        "n_events_in_analyze_single_window": n_in,
        "fraction_events_in_window": (
            float(n_in / n_tot) if n_tot > 0 else 0.0
        ),
    }
    if n_in > 0:
        ew = ee[mask]
        stats["mean_energy_MeV_in_window"] = float(np.mean(ew))
        stats["median_energy_MeV_in_window"] = float(np.median(ew))
        stats["max_energy_MeV_in_window"] = float(np.max(ew))
    else:
        stats["mean_energy_MeV_in_window"] = None
        stats["median_energy_MeV_in_window"] = None
        stats["max_energy_MeV_in_window"] = None

    pd.Series(stats).to_csv("lat_gbm_window_time_analysis.csv")
    with open(
        "lat_gbm_window_time_analysis.json",
        "w",
        encoding="utf-8",
    ) as fj:
        json.dump(stats, fj, indent=2, ensure_ascii=False)
    log(
        "LAT 时间窗统计（与 analyze_single 的 t0、t1 对齐）: "
        f"[{t0_s:g}, {t1_s:g}] s → 窗内事例 {n_in}/{n_tot}"
    )
    return stats


def run_lat_extended_three_ml_pipeline(
    *,
    bn_dir: str,
    bn_name: str,
    grb_name: str,
    selection: Dict[str, Any],
    result_parent: str,
    extended_data_dir: str,
    intervals_count: int = 4,
) -> Dict[str, Any]:
    """
    执行原 ``new.py`` LAT 扩展分析（需在环境中已安装 GtBurst / fermitools）。

    :param bn_dir: ``…/{result_root}/{grb_name}/{bn_name}/``，处理期间会切换到此目录
    :param bn_name: GBM 触发名，如 ``bn231129799``
    :param grb_name: GCN 名，如 ``GRB231129C``
    :param selection: 须含 tstart, tstop, ra, dec, trigger_time；可选 irfs, data_type, Emin, Emax。
        其中 **tstart/tstop** 为 ``analyze_single`` 的 **t0/t1**（相对触发，秒）；本函数会在其两侧各扩展
        ``_LAT_PIPELINE_TIME_PAD_S`` 秒用于 LAT 建库与分段时间轴，统计与图中绿色带仍对应原始
        ``[tstart, tstop]``。
    :param result_parent: ``LAT_dataset.make_LAT_dataset`` 的 destination_directory（一般为 ``…/{grb_name}``）。
        GtBurst / ``TransientLATDataBuilder`` 会在其下查找 ``{result_parent}/{bn_name}/``（例如
        ``…/GRB231129C/bn231129799/``），故 **不可** 把 ``datarepository`` 设为当前工作目录 ``bn_dir``。
    :param extended_data_dir: 复制前的 Extended 目录 ``…/Extended_data_ex/{grb_name}``
    """
    ensure_analysis_runtime()
    update_logging_level("INFO")

    result_data: Dict[str, Any] = {}

    with _working_directory(bn_dir):
        os.makedirs(bn_dir, exist_ok=True)
        # GtBurst：join(datarepository, bn_name, …)；cwd 为 bn_dir 时不可用 "."。
        gtburst_data_repository = os.path.abspath(result_parent)

        t0_core = float(selection["tstart"])
        t1_core = float(selection["tstop"])
        t0_lat = t0_core - _LAT_PIPELINE_TIME_PAD_S
        t1_lat = t1_core + _LAT_PIPELINE_TIME_PAD_S
        log(
            "LAT 时间轴：analyze_single [t0,t1] "
            f"[{t0_core:g}, {t1_core:g}] s → 流水线扩展 ±{_LAT_PIPELINE_TIME_PAD_S:g} s 为 "
            f"[{t0_lat:g}, {t1_lat:g}] s（相对触发）"
        )

        ft1_file, ft2_file = _resolve_ft_paths(
            bn_dir, bn_name, extended_data_dir
        )

        _, eboundsFilename, _, cspecfile = _makeDatasetsOutOfLATdata(
            ft1_file,
            ft2_file,
            bn_name,
            t0_lat,
            t1_lat,
            selection["ra"],
            selection["dec"],
            selection["trigger_time"],
            bn_dir,
        )

        my_lat = LAT_dataset()
        my_lat.make_LAT_dataset(
            selection["ra"],
            selection["dec"],
            12,
            selection["trigger_time"],
            t0_lat,
            t1_lat,
            selection.get("data_type", "Extended"),
            result_parent,
            float(selection.get("Emin", 100.0)),
            float(selection.get("Emax", 100000.0)),
        )

        t05 = float(selection.get("t05", 0.0))
        if float(selection.get("t90", 0.0)) > 0:
            t95 = round(float(selection["t90"]) + t05, 1)
        else:
            t95 = 0.0

        thetamax = 180.0
        irfs_sel = selection.get("irfs", "p8_transient010e")
        strategy = "time"

        results_scan = []
        best_n = 0
        best_combo: Optional[tuple[int, int]] = None

        for zmax_i in range(100, 106):
            for roi_i in range(1, 13):
                with _silence_process_stdio():
                    my_lat.extract_events(
                        roi_i, zmax_i, irfs_sel, thetamax, strategy=strategy
                    )
                n_ev = my_lat.nEvents
                results_scan.append(
                    {"zmax": zmax_i, "roi": roi_i, "nEvents": n_ev}
                )
                if n_ev > best_n:
                    best_n = n_ev
                    best_combo = (roi_i, zmax_i)

        if best_combo is None:
            raise RuntimeError("LAT extract_events 扫描未得到有效组合")

        roi, zmax = best_combo
        with _silence_process_stdio():
            my_lat.extract_events(
                roi, zmax, irfs_sel, thetamax, strategy=strategy
            )

        result_data["roi"] = roi
        result_data["zmax"] = zmax

        df_scan = pd.DataFrame(results_scan)
        log("LAT roi/zmax 扫描前几名:\n" + df_scan.sort_values(
            "nEvents", ascending=False
        ).head().to_string())

        with pyfits.open(my_lat.filt_file) as event_file:
            lat_events = event_file["EVENTS"].data
        event_times = lat_events["TIME"] - float(selection["trigger_time"])

        # 与 analyze_single 中 t0、t1 完全一致的核心分析窗（图中绿色带、窗内统计）
        t0_analysis = t0_core
        t1_analysis = t1_core
        result_data["analyze_single_t0_s"] = t0_core
        result_data["analyze_single_t1_s"] = t1_core
        result_data["lat_pipeline_t0_s"] = t0_lat
        result_data["lat_pipeline_t1_s"] = t1_lat
        result_data["lat_pipeline_pad_each_side_s"] = _LAT_PIPELINE_TIME_PAD_S
        energies_arr = np.asarray(lat_events["ENERGY"], dtype=float)
        win_stats = _analyze_gbm_aligned_time_window(
            np.asarray(event_times, dtype=float),
            energies_arr,
            t0_analysis,
            t1_analysis,
        )
        result_data.update(win_stats)

        tstart_ev = max(
            float(np.min(event_times)) - 15.0,
            t0_lat,
        )
        tstop_ev = min(
            float(np.max(event_times)) + 15.0,
            t1_lat,
        )

        bin_width_s = 2.0
        intervals = np.arange(
            tstart_ev, tstop_ev + bin_width_s, bin_width_s
        )

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
        _pad_lab = f"LAT pipeline ±{_LAT_PIPELINE_TIME_PAD_S:g}s"
        for ax in axs:
            ax.axvspan(
                t0_lat,
                t1_lat,
                alpha=0.12,
                color="tab:orange",
                zorder=0,
                label=_pad_lab,
            )
            ax.axvspan(
                t0_analysis,
                t1_analysis,
                alpha=0.22,
                color="tab:green",
                zorder=1,
                label="analyze_single [t0,t1]",
            )
        axs[0].hist(
            event_times,
            bins=intervals,
            histtype="stepfilled",
            alpha=0.25,
            color="C0",
        )
        axs[0].hist(event_times, bins=intervals, histtype="step", color="C0")
        axs[0].set_ylabel("Events")
        axs[1].scatter(
            event_times,
            lat_events["ENERGY"],
            marker="o",
            c=lat_events["ENERGY"],
            norm="log",
            alpha=0.5,
            zorder=20,
        )
        axs[1].set_yscale("log")
        axs[1].set_ylabel("Energy [MeV]")
        axs[1].set_xlabel("Time - T0 [s]")
        axs[1].grid(True)
        h0, l0 = axs[0].get_legend_handles_labels()
        if h0:
            axs[0].legend(h0, l0, loc="upper right", fontsize=9)
        fig.savefig("events.png")
        plt.close(fig)

        # 聚焦扩展后的 LAT 窗附近（含 ±pad），核心 [t0,t1] 仍以绿色标出
        dur_ext = max(t1_lat - t0_lat, 1e-6)
        pad_ext = max(0.05 * dur_ext, 1.0)
        x0_win = t0_lat - pad_ext
        x1_win = t1_lat + pad_ext
        et = np.asarray(event_times, dtype=float)
        mask_w = (et >= x0_win) & (et <= x1_win)
        tw = et[mask_w]
        ew = energies_arr[mask_w]
        bins_w = np.arange(x0_win, x1_win + bin_width_s, bin_width_s)

        fig_w, axw = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
        for ax in axw:
            ax.axvspan(
                t0_lat,
                t1_lat,
                alpha=0.12,
                color="tab:orange",
                zorder=0,
            )
            ax.axvspan(
                t0_analysis,
                t1_analysis,
                alpha=0.22,
                color="tab:green",
                zorder=1,
            )
        if tw.size:
            axw[0].hist(tw, bins=bins_w, histtype="stepfilled", alpha=0.3, color="C0")
            axw[0].hist(tw, bins=bins_w, histtype="step", color="C0")
            axw[1].scatter(
                tw,
                ew,
                marker="o",
                c=ew,
                norm="log",
                alpha=0.55,
                zorder=20,
            )
        axw[0].set_ylabel("Events")
        axw[0].set_title(
            "LAT: core [t0,t1] "
            f"[{t0_analysis:g},{t1_analysis:g}] s; pipeline "
            f"±{_LAT_PIPELINE_TIME_PAD_S:g}s → "
            f"[{t0_lat:g},{t1_lat:g}] s (relative to trigger)"
        )
        axw[1].set_yscale("log")
        axw[1].set_ylabel("Energy [MeV]")
        axw[1].set_xlabel("Time - T0 [s]")
        axw[1].grid(True)
        axw[0].set_xlim(x0_win, x1_win)
        fig_w.tight_layout()
        fig_w.savefig("events_analyze_single_window.png", dpi=150)
        plt.close(fig_w)

        tstarts_list: list[str] = []
        tstops_list: list[str] = []
        for t0_i, t1_i in zip(intervals[:-1], intervals[1:]):
            tstarts_list.append(f"{t0_i:.1f}")
            tstops_list.append(f"{t1_i:.1f}")
        tstarts_str = ",".join(tstarts_list).replace("-", "\\-")
        tstops_str = ",".join(tstops_list).replace("-", "\\-")

        result_data["T95"] = t95

        if t95 >= tstart_ev:
            builder_transient = TransientLATDataBuilder(
                my_lat.grb_name,
                outfile=f"{my_lat.grb_name}_transient",
                roi=float(roi),
                tstarts=f"{tstart_ev:.1f}",
                tstops=f"{t95:.1f}",
                irf=irfs_sel,
                zmax=float(zmax),
                galactic_model="template",
                particle_model="isotr template",
                datarepository=gtburst_data_repository,
            )
            builder_transient.display(get=True)
            obs_transient = builder_transient.run(
                include_previous_intervals=False
            )
        else:
            log(
                f"跳过瞬时段分析：t95 ({t95}) < tstart ({tstart_ev})"
            )
            obs_transient = []

        iv_c = intervals_count
        if t95 > tstart_ev:
            after_edges = np.logspace(
                np.log10(t95), np.log10(tstop_ev), iv_c - 1
            )
        else:
            after_edges = np.logspace(
                np.log10(tstart_ev), np.log10(tstop_ev), iv_c
            )
            iv_c = iv_c + 1

        source_obs_list = []
        for _i in range(iv_c - 2):
            seg_tag = f"after{_i + 1}"
            t1_e, t2_e = (
                f"{after_edges[_i]:.1f}",
                f"{after_edges[_i + 1]:.1f}",
            )
            builder_seg = TransientLATDataBuilder(
                my_lat.grb_name,
                outfile=f"{my_lat.grb_name}_{seg_tag}",
                roi=float(roi),
                tstarts=t1_e,
                tstops=t2_e,
                irf="p8_transient010e",#p8_source
                zmax=float(zmax),
                galactic_model="template",
                particle_model="isotr template",
                datarepository=gtburst_data_repository,
            )
            builder_seg.display(get=True)
            obs_seg = builder_seg.run(include_previous_intervals=False)
            source_obs_list.append(obs_seg)

        if obs_transient:
            lat_observations = [obs_transient] + source_obs_list
        else:
            lat_observations = source_obs_list

        builder_all = TransientLATDataBuilder(
            my_lat.grb_name,
            outfile=f"{my_lat.grb_name}_all",
            roi=float(roi),
            tstarts=f"{tstart_ev:.1f}",
            tstops=f"{tstop_ev:.1f}",
            irf="p8_transient010e",#p8_transient010e，p8_source
            zmax=float(zmax),
            galactic_model="template",
            particle_model="isotr template",
            datarepository=gtburst_data_repository,
        )
        builder_all.display(get=True)
        builder_all.run(include_previous_intervals=False)

        lat_plugins: Dict[str, Any] = {}
        for obs_list in lat_observations:
            for lob in obs_list:
                lat_name = "LAT_%.1f-%.1f" % (
                    float(lob.tstart),
                    float(lob.tstop),
                )
                lat_plugins[lat_name] = lob.to_LATLike()

        fit_results: Dict[str, Any] = {}
        for T0_i, T1_i in zip(intervals[:-1], intervals[1:]):
            lat_name = "LAT_%.1f-%.1f" % (T0_i, T1_i)
            lat_model_name = ("LAT%dX%d" % (T0_i, T1_i)).replace("-", "n")
            prob_path = (
                f"./interval{T0_i:.1f}-{T1_i:.1f}/"
                f"gll_ft1_tr_bn{my_lat.grb_name}_v00_filt_prob.fit"
            )
            if lat_name not in lat_plugins:
                fit_results[lat_name] = None
                continue

            if os.path.exists(prob_path):
                ts_path = (
                    f"./interval{T0_i:.1f}-{T1_i:.1f}/source_TS.txt"
                )
                data_ts: Dict[str, str] = {}
                try:
                    with open(ts_path, encoding="utf-8") as fts:
                        for ln in fts:
                            ln = ln.strip()
                            if not ln:
                                continue
                            parts = [p.strip() for p in ln.split(",")]
                            if len(parts) >= 2:
                                data_ts[parts[0]] = parts[1]
                except OSError:
                    fit_results[lat_name] = None
                    continue

                if "GRB" not in data_ts:
                    fit_results[lat_name] = None
                    continue
                grb_ts = float(data_ts["GRB"])
                if grb_ts > 16:
                    grb = PointSource(
                        "GRB",
                        ra=float(selection["ra"]),
                        dec=float(selection["dec"]),
                        spectral_shape=Powerlaw_flux(),
                    )
                    model = Model(grb)
                    model.GRB.spectrum.main.Powerlaw_flux.a = 100.0 * u.MeV
                    model.GRB.spectrum.main.Powerlaw_flux.b = 100000.0 * u.MeV
                    model.GRB.spectrum.main.Powerlaw_flux.F = 1.0

                    datalist = DataList(lat_plugins[lat_name])
                    model[
                        "GRB.spectrum.main.Powerlaw_flux.F"
                    ].bounds = (1e-7, 1e6)
                    model[
                        "GRB.spectrum.main.Powerlaw_flux.F"
                    ].value = 1e-5
                    model[
                        "GRB.spectrum.main.Powerlaw_flux.index"
                    ].value = -2.2
                    model[
                        "GRB.spectrum.main.Powerlaw_flux.index"
                    ].bounds = (-4, 0)
                    jl = JointLikelihood(model, datalist, verbose=False)
                    jl.set_minimizer("minuit")
                    jl.fit(compute_covariance=True)
                    fit_results[lat_name] = jl
                else:
                    fit_results[lat_name] = None
            else:
                fit_results[lat_name] = None

        for _i in range(len(intervals) - 1):
            T0_i, T1_i = intervals[_i], intervals[_i + 1]
            lat_name = f"LAT_{T0_i:.1f}-{T1_i:.1f}"
            jl_i = fit_results.get(lat_name)
            if jl_i is not None:
                fig_sp = display_spectrum_model_counts(
                    jl_i, figsize=(10, 8)
                )
                fig_sp.savefig(
                    f"{lat_name}_spectrum.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close(fig_sp)

        valid_jl = [x for x in fit_results.values() if x is not None]
        if valid_jl:
            fig_sp2 = plot_spectra(
                *[a.results for a in valid_jl],
                ene_min=100 * u.MeV,
                ene_max=100 * u.GeV,
                flux_unit="MeV2/(cm2 s MeV)",
                energy_unit="MeV",
                fit_cmap="viridis",
                contour_cmap="viridis",
                contour_style_kwargs=dict(alpha=0.1),
            )
            fig_sp2.set_size_inches(10, 8)
            fig_sp2.savefig("spectra.png")
            plt.close(fig_sp2)

        variates = ["F", "index"]
        yv: Dict[str, list] = {n: [] for n in variates}
        for n in variates:
            yv[n + "_p"] = []
            yv[n + "_n"] = []
        xv: list[float] = []
        dxv: list[float] = []

        try:
            for T0_i, T1_i in zip(intervals[:-1], intervals[1:]):
                lat_name = "LAT_%.1f-%.1f" % (T0_i, T1_i)
                xv.append((T1_i + T0_i) / 2)
                dxv.append((T1_i - T0_i) / 2)
                jl_i = fit_results.get(lat_name)
                if jl_i is not None:
                    res = jl_i.results
                    for n in variates:
                        mv = res.get_variates(
                            "GRB.spectrum.main.Powerlaw_flux.%s" % n
                        )
                        yv[n].append(mv.median)
                        lo, hi = mv.equal_tail_interval()
                        yv[n + "_p"].append(hi - mv.median)
                        yv[n + "_n"].append(mv.median - lo)
                fig_v = plt.figure(figsize=(8, 12))
                colors_v = ["r", "b"]
                ylabels_v = [
                    "Flux [100MeV - 10GeV] \n $\\gamma$ cm$^{-2}$ s$^{-1}$",
                    "index",
                ]
                for ii, n in enumerate(variates):
                    plt.subplot(len(variates) + 1, 1, ii + 1)
                    plt.errorbar(
                        xv,
                        yv[n],
                        xerr=dxv,
                        yerr=(yv[n + "_n"], yv[n + "_p"]),
                        ls="",
                        c=colors_v[ii],
                    )
                    if ii == 0:
                        plt.yscale("log")
                    if ii == 1:
                        plt.ylim(-4, 0)
                    plt.ylabel(ylabels_v[ii])
                    plt.xlim(tstart_ev, tstop_ev)
                fig_v.savefig("variates.png")
                plt.close(fig_v)
        except Exception as exc:  # noqa: BLE001
            log(f"variates 图绘制跳过: {exc}")

        prob_files: list[str] = []
        for obs_list in lat_observations:
            for lob in obs_list:
                pp = (
                    f"interval{lob.tstart}-{lob.tstop}/"
                    f"gll_ft1_tr_bn{my_lat.grb_name}_v00_filt_prob.fit"
                )
                if os.path.exists(pp):
                    prob_files.append(pp)

        all_highest: list[Dict[str, Any]] = []
        glob_hi = None
        glob_prob = None
        glob_rtime = None

        for prob_file in prob_files:
            with pyfits.open(prob_file) as hdul:
                events = hdul["EVENTS"].data
                prob = events["GRB"]
                rel_t = events["TIME"] - float(selection["trigger_time"])

                df_all = pd.DataFrame(
                    {
                        "ENERGY": events["ENERGY"],
                        "TIME": events["TIME"],
                        "RELATIVE_TIME": rel_t,
                        "PROB": prob,
                        "RA": events["RA"],
                        "DEC": events["DEC"],
                    }
                )
                csv_all = "all_photon.csv"
                df_all.to_csv(
                    csv_all,
                    index=False,
                    mode="a",
                    header=not os.path.exists(csv_all),
                )

                mask_hi = prob > 0.9
                hi_ev = events[mask_hi]
                hi_pr = prob[mask_hi]
                if len(hi_ev["ENERGY"]) == 0:
                    continue

                rel_hi = hi_ev["TIME"] - float(selection["trigger_time"])
                pd.DataFrame(
                    {
                        "ENERGY": hi_ev["ENERGY"],
                        "TIME": hi_ev["TIME"],
                        "RELATIVE_TIME": rel_hi,
                        "PROB": hi_pr,
                        "RA": hi_ev["RA"],
                        "DEC": hi_ev["DEC"],
                    }
                ).to_csv(
                    "all_high_prob.csv",
                    index=False,
                    mode="a",
                    header=not os.path.exists("all_high_prob.csv"),
                )

                midx = int(np.argmax(hi_ev["ENERGY"]))
                hp = hi_ev[midx]
                hprob = float(hi_pr[midx])
                hrt = float(hp["TIME"]) - float(selection["trigger_time"])

                all_highest.append(
                    {
                        "ENERGY": hp["ENERGY"],
                        "TIME": hp["TIME"],
                        "RELATIVE_TIME": hrt,
                        "PROB": hprob,
                        "RA": hp["RA"],
                        "DEC": hp["DEC"],
                        "INTERVAL_FILE": prob_file,
                    }
                )

                if glob_hi is None or hp["ENERGY"] > glob_hi["ENERGY"]:
                    glob_hi = hp
                    glob_prob = hprob
                    glob_rtime = hrt

        if all_highest:
            pd.DataFrame(all_highest).to_csv(
                "highest_photon_per_interval.csv", index=False
            )

        result_data["grb"] = grb_name
        result_data["t95"] = t95
        result_data["ra"] = selection["ra"]
        result_data["dec"] = selection["dec"]
        result_data["tstart"] = tstart_ev
        result_data["tstop"] = tstop_ev
        result_data["irf"] = irfs_sel

        if glob_hi is not None:
            result_data["global_highest_energy"] = glob_hi["ENERGY"]
            result_data["global_highest_prob"] = glob_prob
            result_data["global_highest_relative_time"] = glob_rtime
            ap = "all_high_prob.csv"
            result_data["total_high_prob_photons"] = (
                len(pd.read_csv(ap)) if os.path.isfile(ap) else 0
            )

        out_all = Path(f"{my_lat.grb_name}_all")
        if out_all.is_file():
            with out_all.open("r", encoding="utf-8") as fh:
                hdr = fh.readline().split()
                dat = fh.readline().split()
            if hdr and dat and len(hdr) == len(dat):
                result_data.update(dict(zip(hdr, dat)))

        template = dedent(
            """\
            事件 {grb}
            ra = {ra}
            dec = {dec}
            分析事件段： tstart = {tstart} tstop = {tstop}
            t95 = {t95}
            roi = {roi}
            zmax = {zmax}
            irf = {irf}
            Pindex = {photonIndex} +/- {photonIndexError}
            flux = {flux} +/- {fluxError}
            photonFlux = {photonFlux} +/- {photonFluxError}

            ==================================================
            高能光子信息：

            能量 = {global_highest_energy} MeV
            概率 = {global_highest_prob}
            相对时间 = {global_highest_relative_time} s

            总高概率光子数 = {total_high_prob_photons}

        """
        )

        safe = {k: result_data.get(k, "") for k in [
            "grb", "ra", "dec", "tstart", "tstop", "t95", "roi", "zmax",
            "irf", "photonIndex", "photonIndexError", "flux", "fluxError",
            "photonFlux", "photonFluxError",
            "global_highest_energy", "global_highest_prob",
            "global_highest_relative_time", "total_high_prob_photons",
        ]}
        out_txt = Path(f"{safe['grb']}_fit_results.txt")
        try:
            out_txt.write_text(template.format(**safe), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            log(f"写入拟合摘要失败（缺字段可忽略）: {exc}")

        log(f"LAT Extended threeML 流水线结束，输出目录: {bn_dir}")
        return result_data
