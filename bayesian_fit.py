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
from astromodels import Blackbody
from threeML import (
    BayesianAnalysis,
    DataList,
    Model,
    OGIPLike,
    PointSource,
    display_spectrum_model_counts,
    plot_spectra,
)
from modelbuild import build_model
from separate_spectr import discrete_spectr

from .logging_utils import log

logg = logging.getLogger("threeML")


def _run_bayesian_analysis_for_model(
    model_str: str,
    grb_name: str,
    bnname: str,
    ra: float,
    dec: float,
    datalist: DataList,
    plugins: List[OGIPLike],
    lat_plugin: Optional[OGIPLike],
    dets: List[str],
    result_dir: str,
    bin_start: float,
    bin_end: float,
    duration: float,
    analysis_mode: str,
) -> Dict:
    spectral_model = build_model(model_str)
    grb = PointSource("GRB", ra=ra, dec=dec, spectral_shape=spectral_model)
    model = Model(grb)
    model.display(complete=True)

    bs = BayesianAnalysis(model, datalist)

    bs.set_sampler("dynesty_nested")
    bs.sampler.setup(
        n_live_points=700,
        bound="multi",
        sample="rslice",
        dlogz=0.1,
    )

    bs.sample(quiet=True)
    bs.restore_median_fit()

    result = bs.sampler.results
    samples = result.samples  # noqa: F841
    log_marginal_likelihood = bs.sampler.log_marginal_likelihood

    suffix = "gbm_lat" if "lat" in analysis_mode.lower() else "gbm"

    corner_fig = bs.results.corner_plot()
    corner_fig_path = os.path.join(
        result_dir,
        f"bs_{bnname}_{model_str}_{bin_start}-{bin_end}_{suffix}_corner_plot.png",
    )
    plt.savefig(corner_fig_path)
    plt.close(corner_fig)

    bs.results.display()

    result_fits_path = os.path.join(
        result_dir,
        f"my_results_{bin_start}-{bin_end}.fits",
    )
    bs.results.write_to(result_fits_path, overwrite=True)

    results_data = pyfits.open(result_fits_path)
    parameter_values = results_data[1].data.field(1)
    parameter_value_errors = results_data[1].data.field(4)
    parameters = results_data[1].data.field(0)

    logg.info("开始绘制频谱图")
    try:
        if bnname == "bn231222310":
            spec_fig = display_spectrum_model_counts(bs, min_rate=2)
        else:
            if "lat" in analysis_mode.lower():
                min_rate = [0.01, 1.0, 1.0, 1.0]
            else:
                min_rate = [5.0, 5.0, 5.0]

            spec_fig = display_spectrum_model_counts(bs, min_rate=min_rate)
        spec_fig_path = os.path.join(
            result_dir,
            f"bs_{bnname}_{model_str}_counts_{suffix}_spectrum_{bin_start}-{bin_end}.png",
        )
        spec_fig.savefig(spec_fig_path)
        plt.close(spec_fig)
    except Exception as exc:  # noqa: BLE001
        logg.error("警告: 绘制频谱图失败: %s", exc)

    logg.info("开始绘制SED")
    components_to_use = ["", "total"]
    try:
        sed_ene_max = 100 * u.GeV if "lat" in analysis_mode.lower() else 100 * u.MeV
        fig_sed = plot_spectra(
            bs.results,
            ene_min=1 * u.keV,
            ene_max=sed_ene_max,
            flux_unit="erg/(cm2 s )",
            use_components=True,
        )
        ax = fig_sed.get_axes()[0]
        ax.set_ylim(1e-10, 1e-5)
        sed_path = os.path.join(
            result_dir,
            f"bs_{bnname}_{model_str}_{suffix}_spectrum_{bin_start}-{bin_end}_{components_to_use[0]}.png",
        )
        fig_sed.savefig(sed_path)
        plt.close(fig_sed)
    except Exception as exc:  # noqa: BLE001
        log(f"警告: 绘制SED图失败: {exc}")

    logg.info("开始绘制分离谱")
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
            analysis_mode=analysis_mode,
        )
    except Exception as exc:  # noqa: BLE001
        logg.error("警告: 绘制分离谱失败: %s", exc)

    logg.info("结束绘制")

    logg.info("开始计算统计量")
    stat_frame = bs.results.get_statistic_measure_frame()
    aic = stat_frame["statistical measures"]["AIC"]
    bic = stat_frame["statistical measures"]["BIC"]

    emin = 8 * u.keV
    if "lat" in analysis_mode.lower():
        emax = 1e8 * u.keV
    else:
        emax = 40000 * u.keV
    flux_total = bs.results.get_flux(ene_min=emin, ene_max=emax)
    fluence = flux_total["flux"][0] * (bin_end - bin_start)

    logg.info("开始保存参数信息与误差")
    param_dict: Dict[str, float] = {}
    for i, param_name in enumerate(parameters):
        clean_param_name = (
            param_name.replace(":", "_")
            .replace("+", "_")
            .replace(" ", "_")
        )
        param_dict[f"{clean_param_name}_value"] = parameter_values[i]
        param_dict[f"{clean_param_name}_error"] = parameter_value_errors[i]
    logg.info("结束保存参数信息与误差")

    def _compute_bb_energy_flux_erg_cm2_s(
        k: float,
        kT: float,
        emin_keV: float,
        emax_keV: float,
        n: int = 4096,
    ) -> float:
        try:
            bb = Blackbody()
            bb.K = float(k)
            bb.kT = float(kT)

            es = np.logspace(np.log10(emin_keV), np.log10(emax_keV), n)
            dnde = bb(es)
            integrand = es * dnde
            keV_flux = np.trapz(integrand, es)
            return float(keV_flux) * 1.602176634e-9
        except Exception:
            return float("nan")

    def _get_first_value(keys: List[str]) -> Optional[float]:
        for k in keys:
            if k in param_dict and isinstance(param_dict[k], (int, float)):
                return float(param_dict[k])
        return None

    logg.info("开始计算 FBB")
    f_bb = float("nan")
    if model_str == "blackbody":
        bb_k = _get_first_value(["GRB.spectrum.main.Blackbody.K_value"])
        bb_kT = _get_first_value(["GRB.spectrum.main.Blackbody.kT_value"])
        if bb_k is not None and bb_kT is not None:
            f_bb = _compute_bb_energy_flux_erg_cm2_s(
                bb_k,
                bb_kT,
                float(emin.to_value(u.keV)),
                float(emax.to_value(u.keV)),
            )
    elif model_str in {"band+bb", "comp+bb"}:
        bb_k = _get_first_value(["GRB.spectrum.main.composite.K_2_value"])
        bb_kT = _get_first_value(["GRB.spectrum.main.composite.kT_2_value"])
        if bb_k is not None and bb_kT is not None:
            f_bb = _compute_bb_energy_flux_erg_cm2_s(
                bb_k,
                bb_kT,
                float(emin.to_value(u.keV)),
                float(emax.to_value(u.keV)),
            )
    logg.info("结束计算 FBB")
    logg.info("开始构建 summary_entry")
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
    }
    logg.info("结束构建 summary_entry")
    summary_entry.update(param_dict)
    logg.info("开始更新 summary_entry")

    return summary_entry
