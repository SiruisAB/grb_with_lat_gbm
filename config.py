# -*- coding: utf-8 -*-
"""GBM/LAT 联合分析配置与运行覆盖参数。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class GRBProjectConfig:
    data_dir: str = "/home/mxr/lee/gbmtest/GBM_data"
    gbm_download_dir: str = "/home/mxr/lee/gbmtest/GBM_data"
    lat_filtered_csv: str = "/home/mxr/lee/data/GRB_FermiLAT_filtered.csv"
    gbm_download_year_from: int = 2022
    gcn_archive_url: str = "https://gcn.nasa.gov/circulars/archive.json.tar.gz"
    gcn_archive_tar: str = "/home/mxr/lee/data/archive.json.tar.gz"
    gcn_archive_dir: str = "/home/mxr/lee/data/archive.json"
    lat_gcn_output_csv: str = "/home/mxr/lee/data/GRB_FermiLAT_Time_Extended.csv"
    lat_download_root: str = "/home/mxr/lee/data/fermilat/Extended_data_ex"
    catalog_xls: str = "/home/mxr/lee/data/GBMcatolog.xls"
    merged_xls: str = "/home/mxr/lee/newflietest/merged_grb_fit_results.xlsx"
    fermilat_grb_xls: str = "/home/mxr/lee/fermilat-grb.xls"
    result_root: str = "/home/mxr/lee/gbmtest/results_last"
    summary_csv_name: str = "summary_results.csv"
    analysis_mode: str = "gbm+lat"
    fixed_num_time_bins: Optional[int] = None
    bnname: Optional[str] = None
    grbname: Optional[str] = None
    t0: Optional[float] = None
    t1: Optional[float] = None
    trigger_met: Optional[float] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    lat_three_ml_full: Optional[bool] = None
    plot_joint_lightcurve: Optional[bool] = None
    gbm_start: Optional[float] = None
    gbm_stop: Optional[float] = None
    gbm_display_pad_before_s: Optional[float] = None
    gbm_display_pad_after_s: Optional[float] = None
    lightcurve_include_lat: Optional[bool] = None
    lightcurve_lat_prob_threshold: Optional[float] = None
    lightcurve_nai_bands_kev: Optional[Sequence[tuple[float, float]]] = None
    lightcurve_bgo_band_kev: Optional[tuple[float, float]] = None
    lightcurve_active_interval: Optional[str] = None
    lightcurve_background_intervals: Optional[Sequence[str]] = None
    models: Optional[Sequence[str]] = None
    parallel_models: bool = True
    model_workers: int = 2
    # 单个模型拟合的 wall-clock 上限（秒），<=0 表示不限时。
    model_fit_timeout_s: float = 5400.0
    plot_style: Optional[dict] = None
    special_yaml: Optional[str] = None
    special_burst_name: Optional[str] = None
    model_preset: Optional[str] = None


@dataclass
class GRBRunOverrides:
    grbname: Optional[str] = None
    t0: Optional[float] = None
    t1: Optional[float] = None
    trigger_met: Optional[float] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    lat_three_ml_full: Optional[bool] = None
    plot_joint_lightcurve: Optional[bool] = None
    gbm_start: Optional[float] = None
    gbm_stop: Optional[float] = None
    gbm_display_pad_before_s: Optional[float] = None
    gbm_display_pad_after_s: Optional[float] = None
    lightcurve_include_lat: Optional[bool] = None
    lightcurve_lat_prob_threshold: Optional[float] = None
    lightcurve_nai_bands_kev: Optional[Sequence[tuple[float, float]]] = None
    lightcurve_bgo_band_kev: Optional[tuple[float, float]] = None
    lightcurve_active_interval: Optional[str] = None
    lightcurve_background_intervals: Optional[Sequence[str]] = None
    models: Optional[Sequence[str]] = None
    parallel_models: Optional[bool] = None
    model_workers: Optional[int] = None
    model_fit_timeout_s: Optional[float] = None
    plot_style: Optional[dict] = None
    special_yaml: Optional[str] = None
    special_burst_name: Optional[str] = None

    def is_empty(self) -> bool:
        return all(
            getattr(self, k) is None
            for k in (
                "grbname",
                "t0",
                "t1",
                "trigger_met",
                "ra",
                "dec",
                "lat_three_ml_full",
                "plot_joint_lightcurve",
                "gbm_start",
                "gbm_stop",
                "gbm_display_pad_before_s",
                "gbm_display_pad_after_s",
                "lightcurve_include_lat",
                "lightcurve_lat_prob_threshold",
                "lightcurve_nai_bands_kev",
                "lightcurve_bgo_band_kev",
                "lightcurve_active_interval",
                "lightcurve_background_intervals",
                "models",
                "parallel_models",
                "model_workers",
                "model_fit_timeout_s",
                "plot_style",
                "special_yaml",
                "special_burst_name",
            )
        )


def run_overrides_from_config(cfg: GRBProjectConfig) -> Optional[GRBRunOverrides]:
    ov = GRBRunOverrides(
        grbname=cfg.grbname,
        t0=cfg.t0,
        t1=cfg.t1,
        trigger_met=cfg.trigger_met,
        ra=cfg.ra,
        dec=cfg.dec,
        lat_three_ml_full=cfg.lat_three_ml_full,
        plot_joint_lightcurve=cfg.plot_joint_lightcurve,
        gbm_start=cfg.gbm_start,
        gbm_stop=cfg.gbm_stop,
        gbm_display_pad_before_s=cfg.gbm_display_pad_before_s,
        gbm_display_pad_after_s=cfg.gbm_display_pad_after_s,
        lightcurve_include_lat=cfg.lightcurve_include_lat,
        lightcurve_lat_prob_threshold=cfg.lightcurve_lat_prob_threshold,
        lightcurve_nai_bands_kev=cfg.lightcurve_nai_bands_kev,
        lightcurve_bgo_band_kev=cfg.lightcurve_bgo_band_kev,
        lightcurve_active_interval=cfg.lightcurve_active_interval,
        lightcurve_background_intervals=cfg.lightcurve_background_intervals,
        models=cfg.models,
        parallel_models=cfg.parallel_models,
        model_workers=cfg.model_workers,
        plot_style=cfg.plot_style,
        special_yaml=cfg.special_yaml,
        special_burst_name=cfg.special_burst_name,
    )
    return None if ov.is_empty() else ov
