# -*- coding: utf-8 -*-
"""LAT Extended + GtBurst + threeML 的单文件简化版本。

这个版本尽量保持原有流水线能力，但把流程收敛成清晰的几步：
1. 解析 FT1/FT2
2. 构建 LAT 数据集并扫描 roi/zmax
3. 读取事件并生成时间窗图
4. 按时间段构建 LAT plugin
5. 用 threeML 做最小化拟合并输出结果

设计目标是和 `lk2.py` 一样：结构清晰、逻辑集中、尽量少分支、少重复。
"""

from __future__ import annotations

import contextlib
import json
import multiprocessing as mp
import os
import shutil
import sys
import traceback
import uuid
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, Iterator, Optional

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.io import fits as pyfits

from threeML import (
    DataList,
    JointLikelihood,
    Model,
    PointSource,
    Powerlaw_flux,
    plot_spectra,
)
from threeML.io import update_logging_level
from threeML.utils.data_builders import TransientLATDataBuilder
from threeML.utils.data_download.Fermi_LAT.download_LAT_data import LAT_dataset

from GtBurst.dataHandling import _makeDatasetsOutOfLATdata

from .gbm_core import _build_lat_gcn_t95_segments
from .logging_utils import log
from .process_utils import wait_for_worker_result
from .runtime_env import ensure_analysis_runtime


_LAT_PIPELINE_TIME_PAD_S = 5.0
_LAT_WORKER_TIMEOUT_S = 3600.0

# roi/zmax 扫描时用来观测显著度的 on 区半径（度）。取 PSF 尺度的固定值，
# 对所有组合一致，否则组间不可比。
_SCAN_ON_RADIUS_DEG = 1.0


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
def _silence_stdio() -> Iterator[None]:
    """屏蔽标准输出/错误，避免底层库刷屏。"""
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


def _resolve_ft_paths(lat_dir: str, bn_name: str, extended_data_dir: str) -> tuple[str, str]:
    """优先使用 LAT 根目录下的 ``bnname`` 数据目录。"""
    target_dir = os.path.join(lat_dir, bn_name)
    os.makedirs(target_dir, exist_ok=True)

    ft1_local = os.path.join(target_dir, f"gll_ft1_tr_{bn_name}_v00.fit")
    ft2_local = os.path.join(target_dir, f"gll_ft2_tr_{bn_name}_v00.fit")
    if os.path.exists(ft1_local) and os.path.exists(ft2_local):
        return ft1_local, ft2_local

    source_candidates = [
        (
            os.path.join(lat_dir, f"gll_ft1_tr_{bn_name}_v00.fit"),
            os.path.join(lat_dir, f"gll_ft2_tr_{bn_name}_v00.fit"),
        ),
        (
            os.path.join(extended_data_dir, f"gll_ft1_tr_{bn_name}_v00.fit"),
            os.path.join(extended_data_dir, f"gll_ft2_tr_{bn_name}_v00.fit"),
        ),
    ]
    for src_ft1, src_ft2 in source_candidates:
        if os.path.exists(src_ft1) and os.path.exists(src_ft2):
            shutil.copy2(src_ft1, ft1_local)
            shutil.copy2(src_ft2, ft2_local)
            return ft1_local, ft2_local

    if not os.path.isdir(extended_data_dir):
        raise FileNotFoundError(f"Extended 目录不存在: {extended_data_dir}")

    names = os.listdir(extended_data_dir)
    ft1_candidates = [f for f in names if f.startswith("L") and f.endswith("FT1.fits")]
    ev_candidates = [f for f in names if f.startswith("L") and f.endswith("EV00.fits")]
    sc_candidates = [f for f in names if f.startswith("L") and f.endswith("SC00.fits")]

    if ft1_candidates and sc_candidates:
        shutil.copy2(os.path.join(extended_data_dir, ft1_candidates[0]), ft1_local)
        shutil.copy2(os.path.join(extended_data_dir, sc_candidates[0]), ft2_local)
        return ft1_local, ft2_local
    if ev_candidates and sc_candidates:
        shutil.copy2(os.path.join(extended_data_dir, ev_candidates[0]), ft1_local)
        shutil.copy2(os.path.join(extended_data_dir, sc_candidates[0]), ft2_local)
        return ft1_local, ft2_local

    raise FileNotFoundError(
        f"在 {extended_data_dir} 未找到可用的 FT1/FT2、FT1/SC00 或 EV00/SC00 文件"
    )


def _build_analysis_segments(
    t0_core: float,
    t1_core: float,
    t95: float,
    analysis_bin_start: Optional[float],
    analysis_bin_end: Optional[float],
) -> list[dict[str, Any]]:
    if analysis_bin_start is not None and analysis_bin_end is not None:
        return [
            {
                "tstart": float(analysis_bin_start),
                "tstop": float(analysis_bin_end),
                "tag": "analysis_bin",
            }
        ]
    return _build_lat_gcn_t95_segments(t0_core, t1_core, t95)


def _analyze_gbm_aligned_time_window(
    event_times_rel_s: np.ndarray,
    energies_mev: np.ndarray,
    t0_s: float,
    t1_s: float,
) -> Dict[str, Any]:
    tt = np.asarray(event_times_rel_s, dtype=float)
    ee = np.asarray(energies_mev, dtype=float)
    mask = (tt >= float(t0_s)) & (tt <= float(t1_s))
    n_tot = int(tt.size)
    n_in = int(mask.sum())

    stats: Dict[str, Any] = {
        "analysis_t0_s": float(t0_s),
        "analysis_t1_s": float(t1_s),
        "window_duration_s": float(t1_s - t0_s),
        "n_events_total_filtered_file": n_tot,
        "n_events_in_analyze_single_window": n_in,
        "fraction_events_in_window": float(n_in / n_tot) if n_tot > 0 else 0.0,
    }
    if n_in > 0:
        ew = ee[mask]
        stats.update(
            {
                "mean_energy_MeV_in_window": float(np.mean(ew)),
                "median_energy_MeV_in_window": float(np.median(ew)),
                "max_energy_MeV_in_window": float(np.max(ew)),
            }
        )
    else:
        stats.update(
            {
                "mean_energy_MeV_in_window": None,
                "median_energy_MeV_in_window": None,
                "max_energy_MeV_in_window": None,
            }
        )

    pd.Series(stats).to_csv("lat_gbm_window_time_analysis.csv")
    with open("lat_gbm_window_time_analysis.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)

    log(f"LAT 时间窗统计: [{t0_s:g}, {t1_s:g}] s → {n_in}/{n_tot}")
    return stats


def _make_lat_event_figure(
    *,
    event_times: np.ndarray,
    energies: np.ndarray,
    t0_lat: float,
    t1_lat: float,
    t0_analysis: float,
    t1_analysis: float,
    t95: float,
    output_path: str,
    zoom: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    et = np.asarray(event_times, dtype=float)
    ee = np.asarray(energies, dtype=float)
    if et.size:
        tstart_ev = float(np.min(et)) - 15.0
        tstop_ev = float(np.max(et)) + 15.0
    else:
        tstart_ev = t0_lat
        tstop_ev = t1_lat

    if zoom:
        pad = max(0.05 * max(t1_lat - t0_lat, 1e-6), 1.0)
        x0 = t0_lat - pad
        x1 = t1_lat + pad
        mask = (et >= x0) & (et <= x1)
        plot_times = et[mask]
        plot_energies = ee[mask]
        bins = np.arange(x0, x1 + 2.0, 2.0)
    else:
        x0, x1 = tstart_ev, tstop_ev
        plot_times = et
        plot_energies = ee
        bins = np.arange(tstart_ev, tstop_ev + 2.0, 2.0)

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    for ax in axs:
        ax.axvspan(t0_lat, t1_lat, color="tab:orange", alpha=0.12, zorder=0)
        ax.axvspan(t0_analysis, t1_analysis, color="tab:green", alpha=0.22, zorder=1)
    if t0_analysis < t95 < t1_analysis:
        for ax in axs:
            ax.axvline(t95, color="tab:red", ls="--", lw=1.0, alpha=0.8, zorder=2)

    if plot_times.size:
        axs[0].hist(plot_times, bins=bins, histtype="stepfilled", color="C0", alpha=0.25)
        axs[0].hist(plot_times, bins=bins, histtype="step", color="C0")
        axs[1].scatter(plot_times, plot_energies, c=plot_energies, norm="log", s=18, alpha=0.55)

    axs[0].set_ylabel("Events")
    axs[1].set_yscale("log")
    axs[1].set_ylabel("Energy [MeV]")
    axs[1].set_xlabel("Time - T0 [s]")
    axs[1].grid(True)

    if zoom:
        axs[0].set_xlim(x0, x1)
        axs[0].set_title(
            "LAT zoom window "
            f"[{t0_analysis:g}, {t1_analysis:g}] s, pipeline [{t0_lat:g}, {t1_lat:g}] s"
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150 if zoom else 100)
    plt.close(fig)


def _angular_separation_deg(
    ra0: float,
    dec0: float,
    ra: np.ndarray,
    dec: np.ndarray,
) -> np.ndarray:
    """事例到给定天球位置的角距（度）。"""
    lat0, lon0 = np.radians(float(dec0)), np.radians(float(ra0))
    lat1, lon1 = np.radians(np.asarray(dec, dtype=float)), np.radians(np.asarray(ra, dtype=float))
    cos_sep = np.sin(lat0) * np.sin(lat1) + np.cos(lat0) * np.cos(lat1) * np.cos(lon1 - lon0)
    return np.degrees(np.arccos(np.clip(cos_sep, -1.0, 1.0)))


def _li_ma_significance(n_on: int, n_off: int, alpha: float) -> float:
    """Li & Ma (1983) 式 17 的 on/off 显著度。

    超出为负时返回负值，便于排序时区分"没有超出"与"刚好为零"。
    """
    n_on, n_off = int(n_on), int(n_off)
    alpha = float(alpha)
    if alpha <= 0.0 or (n_on == 0 and n_off == 0):
        return 0.0

    total = n_on + n_off
    term_on = n_on * np.log(((1.0 + alpha) / alpha) * (n_on / total)) if n_on > 0 else 0.0
    term_off = n_off * np.log((1.0 + alpha) * (n_off / total)) if n_off > 0 else 0.0
    value = float(np.sqrt(max(2.0 * (term_on + term_off), 0.0)))
    return value if n_on >= alpha * n_off else -value


def _on_off_significance(
    filt_file: str,
    ra: float,
    dec: float,
    roi: float,
    on_radius: float = _SCAN_ON_RADIUS_DEG,
) -> dict[str, Any]:
    """在一次抽取的事例文件里做空间 on/off，估计源的显著度。

    on 区是以源为心、半径 ``on_radius`` 的圆，off 区是 ``on_radius`` 到 ``roi``
    之间的圆环，两者共用同一份 GTI 与曝光，因此 alpha 只是立体角之比、不含
    时间——这正是这里用空间 on/off 而不用时间 on/off 的原因：流水线前后只各
    留了 ``_LAT_PIPELINE_TIME_PAD_S`` 秒 padding，做不了时间本底。

    这个函数只用于观测和记录，不参与任何选择；任何异常都吞掉并返回空值，
    不能因为它把扫描本身弄挂。
    """
    empty = {"n_on": None, "n_off": None, "alpha": None, "excess": None, "sigma": None}
    if float(roi) <= float(on_radius):
        # 圆环退化，本底无从估计。
        return empty

    try:
        with pyfits.open(filt_file) as event_file:
            events = event_file["EVENTS"].data
            sep = _angular_separation_deg(ra, dec, events["RA"], events["DEC"])

        n_on = int(np.count_nonzero(sep <= float(on_radius)))
        n_off = int(np.count_nonzero((sep > float(on_radius)) & (sep <= float(roi))))

        # 立体角正比于 1 - cos(theta)
        omega_on = 1.0 - np.cos(np.radians(float(on_radius)))
        omega_off = np.cos(np.radians(float(on_radius))) - np.cos(np.radians(float(roi)))
        if omega_off <= 0.0:
            return empty
        alpha = float(omega_on / omega_off)

        return {
            "n_on": n_on,
            "n_off": n_off,
            "alpha": round(alpha, 6),
            "excess": round(n_on - alpha * n_off, 2),
            "sigma": round(_li_ma_significance(n_on, n_off, alpha), 2),
        }
    except Exception as exc:  # noqa: BLE001
        log(f"roi={roi} 的显著度估计失败（不影响扫描）: {exc}")
        return empty


def _scan_best_roi_zmax(lat_ds: LAT_dataset, irfs_sel: str, thetamax: float) -> tuple[int, int, list[dict[str, Any]]]:
    """扫描 roi/zmax 组合。

    选择判据仍然是 nEvents 最大，与历来一致。每组同时记录一份 on/off 显著度
    作为对照列，只进日志、不参与选择——用来观察"按事例数选"和"按显著度选"
    会不会选出不同的组合。
    """
    rows: list[dict[str, Any]] = []
    best_combo: Optional[tuple[int, int]] = None
    best_n = -1

    ra = float(getattr(lat_ds, "ra", float("nan")))
    dec = float(getattr(lat_ds, "dec", float("nan")))

    for zmax in range(100, 106):
        for roi in range(1, 13):
            with _silence_stdio():
                lat_ds.extract_events(roi, zmax, irfs_sel, thetamax, strategy="time")
            n_ev = int(lat_ds.nEvents)
            row: dict[str, Any] = {"zmax": zmax, "roi": roi, "nEvents": n_ev}
            row.update(
                _on_off_significance(getattr(lat_ds, "filt_file", ""), ra, dec, float(roi))
            )
            rows.append(row)
            if n_ev > best_n:
                best_n = n_ev
                best_combo = (roi, zmax)

    if best_combo is None:
        raise RuntimeError("LAT extract_events 扫描未得到有效组合")

    return best_combo[0], best_combo[1], rows


def _build_lat_plugins(
    *,
    lat_name: str,
    segments: list[dict[str, Any]],
    roi: int,
    zmax: int,
    irfs_sel: str,
    gtburst_data_repository: str,
    grb_name: str,
) -> Dict[str, Any]:
    plugins: Dict[str, Any] = {}
    for seg in segments:
        builder = TransientLATDataBuilder(
            grb_name,
            outfile=f"{grb_name}_{seg['tag']}",
            roi=float(roi),
            tstarts=f"{float(seg['tstart']):.3f}",
            tstops=f"{float(seg['tstop']):.3f}",
            irf=irfs_sel,
            zmax=float(zmax),
            galactic_model="template",
            particle_model="isotr template",
            datarepository=gtburst_data_repository,
        )
        builder.display(get=True)
        for lob in builder.run(include_previous_intervals=False):
            key = f"LAT_{float(lob.tstart):.3f}-{float(lob.tstop):.3f}"
            plugins[key] = lob.to_LATLike()
    if not plugins:
        raise RuntimeError("LAT 分析未生成任何 plugin")
    return plugins


def _fit_lat_plugins(
    *,
    plugins: Dict[str, Any],
    segments: list[dict[str, Any]],
    selection: Dict[str, Any],
) -> Dict[str, Any]:
    fit_results: Dict[str, Any] = {}
    for seg in segments:
        key = f"LAT_{float(seg['tstart']):.3f}-{float(seg['tstop']):.3f}"
        plugin = plugins.get(key)
        if plugin is None:
            continue

        model = Model(
            PointSource(
                "GRB",
                ra=float(selection["ra"]),
                dec=float(selection["dec"]),
                spectral_shape=Powerlaw_flux(),
            )
        )
        model.GRB.spectrum.main.Powerlaw_flux.a = 100.0 * u.MeV
        model.GRB.spectrum.main.Powerlaw_flux.b = 100000.0 * u.MeV
        model["GRB.spectrum.main.Powerlaw_flux.F"].bounds = (1e-7, 1e6)
        model["GRB.spectrum.main.Powerlaw_flux.F"].value = 1e-5
        model["GRB.spectrum.main.Powerlaw_flux.index"].value = -2.2
        model["GRB.spectrum.main.Powerlaw_flux.index"].bounds = (-4, 0)

        jl = JointLikelihood(model, DataList(plugin), verbose=False)
        jl.set_minimizer("minuit")
        jl.fit(compute_covariance=True)
        fit_results[key] = jl

    return fit_results


def _save_spectra_plots(fit_results: Dict[str, Any], segments: list[dict[str, Any]]) -> None:
    valid_jl = [jl for jl in fit_results.values() if jl is not None]
    if not valid_jl:
        return

    fig = plot_spectra(
        *[jl.results for jl in valid_jl],
        ene_min=100 * u.MeV,
        ene_max=100 * u.GeV,
        flux_unit="MeV2/(cm2 s MeV)",
        energy_unit="MeV",
        fit_cmap="viridis",
        contour_cmap="viridis",
    )
    fig.set_size_inches(10, 8)
    fig.savefig("spectra.png")
    import matplotlib.pyplot as plt

    plt.close(fig)

    try:
        seg_t0 = min(float(s["tstart"]) for s in segments)
        seg_t1 = max(float(s["tstop"]) for s in segments)
        xv: list[float] = []
        dxv: list[float] = []
        yv = {"F": [], "F_n": [], "F_p": [], "index": [], "index_n": [], "index_p": []}

        for seg in segments:
            key = f"LAT_{float(seg['tstart']):.3f}-{float(seg['tstop']):.3f}"
            jl = fit_results.get(key)
            xv.append((float(seg["tstart"]) + float(seg["tstop"])) / 2)
            dxv.append((float(seg["tstop"]) - float(seg["tstart"])) / 2)
            if jl is None:
                for arr in yv.values():
                    arr.append(np.nan)
                continue

            res = jl.results
            for name in ("F", "index"):
                mv = res.get_variates(f"GRB.spectrum.main.Powerlaw_flux.{name}")
                lo, hi = mv.equal_tail_interval()
                yv[name].append(mv.median)
                yv[f"{name}_n"].append(mv.median - lo)
                yv[f"{name}_p"].append(hi - mv.median)

        import matplotlib.pyplot as plt

        fig_v = plt.figure(figsize=(8, 10))
        for ii, name in enumerate(("F", "index"), start=1):
            plt.subplot(2, 1, ii)
            plt.errorbar(
                xv,
                yv[name],
                xerr=dxv,
                yerr=(yv[f"{name}_n"], yv[f"{name}_p"]),
                ls="",
                c="r" if name == "F" else "b",
            )
            if name == "F":
                plt.yscale("log")
            else:
                plt.ylim(-4, 0)
            plt.ylabel("Flux" if name == "F" else "index")
            plt.xlim(seg_t0, seg_t1)
        fig_v.savefig("variates.png")
        plt.close(fig_v)
    except Exception as exc:  # noqa: BLE001
        log(f"variates 图绘制跳过: {exc}")


def _result_summary_text(result_data: Dict[str, Any]) -> str:
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
    safe_keys = [
        "grb",
        "ra",
        "dec",
        "tstart",
        "tstop",
        "t95",
        "roi",
        "zmax",
        "irf",
        "photonIndex",
        "photonIndexError",
        "flux",
        "fluxError",
        "photonFlux",
        "photonFluxError",
        "global_highest_energy",
        "global_highest_prob",
        "global_highest_relative_time",
        "total_high_prob_photons",
    ]
    safe = {k: result_data.get(k, "") for k in safe_keys}
    return template.format(**safe)


def _extract_highest_photon_info(event_file: pyfits.HDUList, trigger_time: float) -> Dict[str, Any]:
    events = event_file["EVENTS"].data
    if len(events) == 0:
        return {}

    times = np.asarray(events["TIME"], dtype=float)
    energies = np.asarray(events["ENERGY"], dtype=float)
    rel_times = times - float(trigger_time)

    out: Dict[str, Any] = {
        "global_highest_energy": float(np.max(energies)),
        "global_highest_relative_time": float(rel_times[int(np.argmax(energies))]),
    }

    if "GRB" in events.columns.names:
        prob = np.asarray(events["GRB"], dtype=float)
        hi_mask = prob > 0.9
        out["total_high_prob_photons"] = int(np.sum(hi_mask))
        if np.any(hi_mask):
            idx = int(np.argmax(energies[hi_mask]))
            hi_events = events[hi_mask]
            hi_prob = prob[hi_mask]
            out["global_highest_prob"] = float(hi_prob[idx])
            out["global_highest_energy"] = float(hi_events["ENERGY"][idx])
            out["global_highest_relative_time"] = float(
                float(hi_events["TIME"][idx]) - float(trigger_time)
            )
        else:
            out["global_highest_prob"] = ""
    else:
        out["global_highest_prob"] = ""
        out["total_high_prob_photons"] = 0

    return out


def _run_lat_extended_three_ml_impl(
    *,
    bn_dir: str,
    bn_name: str,
    grb_name: str,
    selection: Dict[str, Any],
    result_parent: str,
    extended_data_dir: str,
    analysis_bin_start: Optional[float],
    analysis_bin_end: Optional[float],
    return_lat_plugin: bool,
) -> Dict[str, Any]:
    result_data: Dict[str, Any] = {}

    lat_dir = str(Path(bn_dir).expanduser().resolve())

    with _working_directory(lat_dir):
        gtburst_data_repository = lat_dir

        t0_core = float(selection["tstart"])
        t1_core = float(selection["tstop"])
        t0_lat = t0_core - _LAT_PIPELINE_TIME_PAD_S
        t1_lat = t1_core + _LAT_PIPELINE_TIME_PAD_S
        log(
            "LAT 时间轴：GCN [T0,T1] "
            f"[{t0_core:g}, {t1_core:g}] s → 扩展 ±{_LAT_PIPELINE_TIME_PAD_S:g} s 为 "
            f"[{t0_lat:g}, {t1_lat:g}] s"
        )

        ft1_file, ft2_file = _resolve_ft_paths(lat_dir, bn_name, extended_data_dir)
        dataset_dir = os.path.join(lat_dir, bn_name)
        _makeDatasetsOutOfLATdata(
            ft1_file,
            ft2_file,
            bn_name,
            t0_lat,
            t1_lat,
            selection["ra"],
            selection["dec"],
            selection["trigger_time"],
            dataset_dir,
        )

        lat_ds = LAT_dataset()
        lat_ds.make_LAT_dataset(
            selection["ra"],
            selection["dec"],
            12,
            selection["trigger_time"],
            t0_lat,
            t1_lat,
            selection.get("data_type", "Extended"),
            lat_dir,
            float(selection.get("Emin", 100.0)),
            float(selection.get("Emax", 100000.0)),
        )

        t05 = float(selection.get("t05", 0.0))
        t95 = round(float(selection["t90"]) + t05, 1) if float(selection.get("t90", 0.0)) > 0 else 0.0
        irfs_sel = selection.get("irfs", "p8_transient020e")
        thetamax = float(selection.get("thetamax", 180.0))

        roi, zmax, scan_rows = _scan_best_roi_zmax(lat_ds, irfs_sel, thetamax)
        with _silence_stdio():
            lat_ds.extract_events(roi, zmax, irfs_sel, thetamax, strategy="time")

        result_data.update({"roi": roi, "zmax": zmax, "irf": irfs_sel})
        scan_frame = pd.DataFrame(scan_rows)
        scan_preview = scan_frame.sort_values("nEvents", ascending=False).head().to_string(index=False)
        log("LAT roi/zmax 扫描前几名（按 nEvents，即实际生效的判据）:\n" + scan_preview)
        if "sigma" in scan_frame.columns and scan_frame["sigma"].notna().any():
            # 仅供对照：如果改按显著度选，会选出哪几组。
            sigma_preview = (
                scan_frame.dropna(subset=["sigma"])
                .sort_values("sigma", ascending=False)
                .head()
                .to_string(index=False)
            )
            log(
                f"LAT roi/zmax 扫描前几名（按 on/off 显著度，on 区半径 "
                f"{_SCAN_ON_RADIUS_DEG:g}°，仅作对照，未参与选择）:\n" + sigma_preview
            )

        with pyfits.open(lat_ds.filt_file) as event_file:
            events = event_file["EVENTS"].data
            event_times = np.asarray(events["TIME"], dtype=float) - float(selection["trigger_time"])
            energies = np.asarray(events["ENERGY"], dtype=float)
            result_data.update(_extract_highest_photon_info(event_file, selection["trigger_time"]))

        t0_analysis = float(analysis_bin_start) if analysis_bin_start is not None else t0_core
        t1_analysis = float(analysis_bin_end) if analysis_bin_end is not None else t1_core
        if analysis_bin_start is not None and analysis_bin_end is not None:
            log(f"LAT 使用 analyze_single 时间窗: [{t0_analysis:g}, {t1_analysis:g}] s")

        result_data.update(
            {
                "grb": grb_name,
                "ra": selection["ra"],
                "dec": selection["dec"],
                "gcn_t0_s": t0_core,
                "gcn_t1_s": t1_core,
                "analysis_bin_start_s": t0_analysis,
                "analysis_bin_end_s": t1_analysis,
                "lat_pipeline_t0_s": t0_lat,
                "lat_pipeline_t1_s": t1_lat,
                "lat_pipeline_pad_each_side_s": _LAT_PIPELINE_TIME_PAD_S,
                "T95": t95,
            }
        )

        result_data.update(_analyze_gbm_aligned_time_window(event_times, energies, t0_analysis, t1_analysis))

        analysis_segments = _build_analysis_segments(t0_core, t1_core, t95, analysis_bin_start, analysis_bin_end)
        if not analysis_segments:
            raise RuntimeError(f"GCN 时间窗无效: T0={t0_core:g}, T1={t1_core:g}, t95={t95:g}")
        result_data["analysis_segments"] = analysis_segments
        log("LAT 瞬态分析时段: " + ", ".join(f"[{s['tstart']:g},{s['tstop']:g}] ({s['tag']})" for s in analysis_segments))

        _make_lat_event_figure(
            event_times=event_times,
            energies=energies,
            t0_lat=t0_lat,
            t1_lat=t1_lat,
            t0_analysis=t0_analysis,
            t1_analysis=t1_analysis,
            t95=t95,
            output_path="events.png",
            zoom=False,
        )
        _make_lat_event_figure(
            event_times=event_times,
            energies=energies,
            t0_lat=t0_lat,
            t1_lat=t1_lat,
            t0_analysis=t0_analysis,
            t1_analysis=t1_analysis,
            t95=t95,
            output_path="events_analyze_single_window.png",
            zoom=True,
        )

        plugins = _build_lat_plugins(
            lat_name=bn_name,
            segments=analysis_segments,
            roi=roi,
            zmax=zmax,
            irfs_sel=irfs_sel,
            gtburst_data_repository=gtburst_data_repository,
            grb_name=lat_ds.grb_name,
        )
        primary_key = next(iter(plugins))
        result_data["lat_plugin_key"] = primary_key
        result_data["lat_plugin"] = plugins[primary_key]

        fit_results = {}
        if not return_lat_plugin:
            fit_results = _fit_lat_plugins(
                plugins=plugins,
                segments=analysis_segments,
                selection=selection,
            )
            _save_spectra_plots(fit_results, analysis_segments)

        valid_jl = [jl for jl in fit_results.values() if jl is not None]
        if valid_jl:
            res = valid_jl[0].results
            flux_var = res.get_variates("GRB.spectrum.main.Powerlaw_flux.F")
            idx_var = res.get_variates("GRB.spectrum.main.Powerlaw_flux.index")
            flux_lo, flux_hi = flux_var.equal_tail_interval()
            idx_lo, idx_hi = idx_var.equal_tail_interval()
            result_data.update(
                {
                    "flux": flux_var.median,
                    "fluxError": max(flux_var.median - flux_lo, flux_hi - flux_var.median),
                    "photonIndex": idx_var.median,
                    "photonIndexError": max(idx_var.median - idx_lo, idx_hi - idx_var.median),
                }
            )

        # 额外信息尽量保留，但不强制依赖这些字段
        result_data.setdefault("flux", "")
        result_data.setdefault("fluxError", "")
        result_data.setdefault("photonIndex", "")
        result_data.setdefault("photonIndexError", "")
        result_data.setdefault("photonFlux", "")
        result_data.setdefault("photonFluxError", "")
        result_data.setdefault("global_highest_energy", "")
        result_data.setdefault("global_highest_prob", "")
        result_data.setdefault("global_highest_relative_time", "")
        result_data.setdefault("total_high_prob_photons", "")

        Path(f"{grb_name}_fit_results.txt").write_text(
            _result_summary_text(result_data),
            encoding="utf-8",
        )

        log(f"LAT Extended threeML 流水线结束，输出目录: {lat_dir}")
        result_data["working_directory"] = lat_dir
        return result_data


def run_lat_extended_three_ml_pipeline(
    *,
    bn_dir: str,
    bn_name: str,
    grb_name: str,
    selection: Dict[str, Any],
    result_parent: str,
    extended_data_dir: str,
    fixed_num_time_bins: Optional[int] = None,
    analysis_bin_start: Optional[float] = None,
    analysis_bin_end: Optional[float] = None,
    return_lat_plugin: bool = False,
) -> Dict[str, Any]:
    """对外暴露的入口；用子进程隔离底层库状态。"""
    ensure_analysis_runtime()
    update_logging_level("INFO")

    start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    ctx = mp.get_context(start_method)
    queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(
        target=_run_lat_extended_three_ml_worker,
        args=(
            queue,
            bn_dir,
            bn_name,
            grb_name,
            selection,
            result_parent,
            extended_data_dir,
            fixed_num_time_bins,
            analysis_bin_start,
            analysis_bin_end,
            return_lat_plugin,
        ),
        name=f"lat-ext-{bn_name}-{uuid.uuid4().hex[:8]}",
    )
    proc.start()
    try:
        status, payload = wait_for_worker_result(proc, queue, _LAT_WORKER_TIMEOUT_S)
    except TimeoutError as exc:
        raise RuntimeError(f"LAT 子进程超时: {exc}") from exc

    if proc.exitcode != 0:
        if status == "ok":
            return payload
        raise RuntimeError(payload.get("error", f"LAT 子进程退出码 {proc.exitcode}"))
    if status == "error":
        raise RuntimeError(payload.get("error", "LAT 子进程失败"))
    return payload


def _run_lat_extended_three_ml_worker(
    queue: mp.Queue,
    bn_dir: str,
    bn_name: str,
    grb_name: str,
    selection: Dict[str, Any],
    result_parent: str,
    extended_data_dir: str,
    fixed_num_time_bins: Optional[int],
    analysis_bin_start: Optional[float],
    analysis_bin_end: Optional[float],
    return_lat_plugin: bool,
) -> None:
    try:
        _ = fixed_num_time_bins
        result = _run_lat_extended_three_ml_impl(
            bn_dir=bn_dir,
            bn_name=bn_name,
            grb_name=grb_name,
            selection=selection,
            result_parent=result_parent,
            extended_data_dir=extended_data_dir,
            analysis_bin_start=analysis_bin_start,
            analysis_bin_end=analysis_bin_end,
            return_lat_plugin=return_lat_plugin,
        )
        queue.put(("ok", result))
    except Exception as exc:  # noqa: BLE001
        queue.put(("error", {"error": f"{exc}\n{traceback.format_exc()}"}))


__all__ = [
    "run_lat_extended_three_ml_pipeline",
]
