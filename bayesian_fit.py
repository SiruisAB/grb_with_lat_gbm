# -*- coding: utf-8 -*-
"""单模型贝叶斯光谱拟合与 summary 条目构建。"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.io import fits as pyfits

from .logging_utils import log
from .publication_style import (
    save_publication_figure,
    style_corner_figure,
    style_diagnostic_figure,
)
from .reproducibility_export import (
    export_counts_figure_data,
    export_spectral_reproducibility_bundle,
)

logg = logging.getLogger("threeML")


def _rate_for_target_plot_bins(plugin, target_bins: int = 5) -> float:
    try:
        expected_rate = np.asarray(plugin.expected_model_rate, dtype=float)
        total_rate = float(np.nansum(expected_rate))
    except Exception:  # noqa: BLE001
        return -1.0
    if not np.isfinite(total_rate) or total_rate <= 0:
        return -1.0
    return float(np.nextafter(total_rate / max(1, int(target_bins)), 0.0))


def _counts_plot_min_rates(
    analysis,
    lat_plugin,
    bnname: str,
    lat_target_bins: int = 5,
) -> list[float]:
    gbm_min_rate = 1.0
    if bnname == "bn231222310":
        gbm_min_rate = 0.01 if lat_plugin is not None else 2.0
    rates: list[float] = []
    for plugin in analysis.data_list.values():
        if lat_plugin is not None and plugin is lat_plugin:
            rates.append(_rate_for_target_plot_bins(plugin, lat_target_bins))
        else:
            rates.append(gbm_min_rate)
    return rates


def _run_bayesian_analysis_for_model(
    model_str: str,
    grb_name: str,
    bnname: str,
    ra: float,
    dec: float,
    datalist,
    plugins,
    lat_plugin,
    dets: List[str],
    result_dir: str, #result_dir / model_str这里使用的是这样的
    bin_start: float,
    bin_end: float,
    duration: float,
    analysis_mode: str,
    requested_analysis_mode: Optional[str] = None,
    plot_style: Optional[dict] = None,
) -> Dict:
    from astromodels import Blackbody
    from grb_project.modelbuild import build_model
    from .separate_spectr import discrete_spectr
    from threeML import BayesianAnalysis, Model, PointSource, display_spectrum_model_counts, plot_spectra

    fluxdata_dir = os.path.join(result_dir, "fluxdata")
    os.makedirs(fluxdata_dir, exist_ok=True)

    spectral_model = build_model(model_str, analysis_mode=analysis_mode)
    grb = PointSource("GRB", ra=ra, dec=dec, spectral_shape=spectral_model)
    model = Model(grb)
    model.display(complete=True)

    bs = BayesianAnalysis(model, datalist)
    bs.set_sampler("dynesty_nested")
    bs.sampler.setup(n_live_points=700, bound="multi", sample="auto", dlogz=0.1)
    bs.sample(quiet=True)
    bs.restore_median_fit()

    log_marginal_likelihood = bs.sampler.log_marginal_likelihood
    suffix = "gbm_lat" if "lat" in analysis_mode.lower() else "gbm"
    # 绘图能量上限按"请求模式"统一：降级为 GBM-only 的 bin 图也画到 1e8 keV；
    # 文件名 suffix 与拟合先验仍按实际拟合（fit_mode）如实走。
    plot_mode = str(requested_analysis_mode or analysis_mode)

    corner_fig = bs.results.corner_plot()
    corner_fig_path = os.path.join(result_dir, f"bs_{bnname}_{model_str}_{bin_start}-{bin_end}_{suffix}_corner_plot.png")
    style_corner_figure(corner_fig, plot_style)
    save_publication_figure(corner_fig, corner_fig_path)
    plt.close(corner_fig)

    bs.results.display()

    result_fits_path = os.path.join(result_dir, f"my_results_{bin_start}-{bin_end}.fits")
    bs.results.write_to(result_fits_path, overwrite=True)
    with pyfits.open(result_fits_path) as results_data:
        parameter_values = results_data[1].data.field(1)
        parameter_value_errors = results_data[1].data.field(4)
        parameters = results_data[1].data.field(0)

    reproducibility_dir: Optional[str] = None
    try:
        reproducibility_dir = str(
            export_spectral_reproducibility_bundle(
                bs,
                result_dir=result_dir,
                grb_name=grb_name,
                bnname=bnname,
                model_name=model_str,
                bin_start=bin_start,
                bin_end=bin_end,
                analysis_mode=analysis_mode,
                detector_names=dets,
            )
        )
    except Exception as exc:  # noqa: BLE001
        logg.error("警告: 保存可复现光谱结果失败: %s", exc)

    try:
        lat_target_bins = int((plot_style or {}).get("lat_plot_bins", 5))
        min_rate = _counts_plot_min_rates(
            bs, lat_plugin, bnname, lat_target_bins=lat_target_bins
        )
        logg.info("Counts plot min_rate by dataset: %s", min_rate)
        spec_fig = display_spectrum_model_counts(bs, min_rate=min_rate)
        spec_fig_path = os.path.join(result_dir, f"bs_{bnname}_{model_str}_counts_{suffix}_spectrum_{bin_start}-{bin_end}.png")
        style_diagnostic_figure(spec_fig, plot_style)
        save_publication_figure(spec_fig, spec_fig_path)
        if reproducibility_dir is not None:
            export_counts_figure_data(spec_fig, reproducibility_dir)
        plt.close(spec_fig)
    except Exception as exc:  # noqa: BLE001
        logg.error("警告: 绘制频谱图失败: %s", exc)

    try:
        sed_ene_max = 100 * u.GeV if "lat" in plot_mode.lower() else 100 * u.MeV
        fig_sed = plot_spectra(
            bs.results,
            ene_min=1 * u.keV,
            ene_max=sed_ene_max,
            flux_unit="erg/(cm2 s )",
            use_components=True,
        )
        ax = fig_sed.get_axes()[0]
        ax.set_ylim(1e-10, 1e-5)
        sed_path = os.path.join(result_dir, f"bs_{bnname}_{model_str}_{suffix}_spectrum_{bin_start}-{bin_end}_total.png")
        style_diagnostic_figure(fig_sed, plot_style)
        save_publication_figure(fig_sed, sed_path)
        plt.close(fig_sed)
    except Exception as exc:  # noqa: BLE001
        log(f"警告: 绘制SED图失败: {exc}")

    try:
        discrete_spectr(
            fluence_plugins=plugins,
            lat=lat_plugin,
            model_str=model_str,
            parameter_values=parameter_values,
            bs=bs,
            result_dir=result_dir,
            bnname=bnname,
            gbm_detectors=dets,
            bin_start=bin_start,
            bin_end=bin_end,
            analysis_mode=plot_mode,
            output_dir=fluxdata_dir,
            plot_style=plot_style,
        )
    except Exception as exc:  # noqa: BLE001
        logg.error("警告: 绘制分离谱失败: %s", exc)

    stat_frame = bs.results.get_statistic_measure_frame()
    aic = stat_frame["statistical measures"]["AIC"]
    bic = stat_frame["statistical measures"]["BIC"]

    emin = 8 * u.keV
    emax = 1e8 * u.keV if "lat" in analysis_mode.lower() else 40000 * u.keV
    flux_total = bs.results.get_flux(ene_min=emin, ene_max=emax)
    fluence = flux_total["flux"][0] * (bin_end - bin_start)

    param_dict: Dict[str, float] = {}
    for i, param_name in enumerate(parameters):
        clean_param_name = param_name.replace(":", "_").replace("+", "_").replace(" ", "_")
        param_dict[f"{clean_param_name}_value"] = parameter_values[i]
        param_dict[f"{clean_param_name}_error"] = parameter_value_errors[i]

    def _compute_bb_energy_flux_erg_cm2_s(k: float, kT: float, emin_keV: float, emax_keV: float, n: int = 4096) -> float:
        try:
            bb = Blackbody()
            bb.K = float(k)
            bb.kT = float(kT)
            es = np.logspace(np.log10(emin_keV), np.log10(emax_keV), n)
            dnde = bb(es)
            return float(np.trapz(es * dnde, es)) * 1.602176634e-9
        except Exception:
            return float("nan")

    def _get_first_value(keys: List[str]) -> Optional[float]:
        for k in keys:
            if k in param_dict and isinstance(param_dict[k], (int, float)):
                return float(param_dict[k])
        return None

    f_bb = float("nan")
    if model_str == "blackbody":
        bb_k = _get_first_value(["GRB.spectrum.main.Blackbody.K_value"])
        bb_kT = _get_first_value(["GRB.spectrum.main.Blackbody.kT_value"])
        if bb_k is not None and bb_kT is not None:
            f_bb = _compute_bb_energy_flux_erg_cm2_s(bb_k, bb_kT, float(emin.to_value(u.keV)), float(emax.to_value(u.keV)))
    elif model_str in {"band+bb", "comp+bb"}:
        bb_k = _get_first_value(["GRB.spectrum.main.composite.K_2_value"])
        bb_kT = _get_first_value(["GRB.spectrum.main.composite.kT_2_value"])
        if bb_k is not None and bb_kT is not None:
            f_bb = _compute_bb_energy_flux_erg_cm2_s(bb_k, bb_kT, float(emin.to_value(u.keV)), float(emax.to_value(u.keV)))

    summary_entry: Dict[str, object] = {
        "grb_name": grb_name,
        "bnname": bnname,
        "model": model_str,
        "analysis_mode": analysis_mode,
        "AIC": aic,
        "BIC": bic,
        "Flux(erg/cm2/s)": flux_total["flux"][0].value,
        "FTot(erg/cm2/s)": flux_total["flux"][0].value,
        "Fluence(erg/cm2)": fluence.value,
        "FBB(erg/cm2/s)": f_bb,
        "log_marginal_likelihood": log_marginal_likelihood,
        "duration": duration,
        "num_time_bins": 1,
        "bin_start_time": bin_start,
        "bin_end_time": bin_end,
        "bin_duration": bin_end - bin_start,
        "reproducibility_dir": reproducibility_dir,
    }
    summary_entry.update(param_dict)
    return summary_entry
