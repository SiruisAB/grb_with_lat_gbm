# -*- coding: utf-8 -*-
"""单次 GRB 全流程：LAT（可选）、GBM 插件、多模型拟合。"""

from __future__ import annotations

import logging
import os
import re
import traceback
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from threeML import DataList, OGIPLike

from .bayesian_fit import _run_bayesian_analysis_for_model
from .config import GRBRunOverrides
from .gbm_core import (
    _build_background_interval_string,
    _build_gbm_plugin_for_detector,
    _determine_time_bins,
    _determine_time_interval_and_position,
    _select_gbm_detectors,
)
from .io_utils import (
    _append_text_report,
    copy_extended_lat_to_bn_workspace,
    ensure_dir,
)
from .lat_processing import process_lat_data
from .logging_utils import log
from .session import session

logg = logging.getLogger("threeML")


def analyze_grb(
    idx: int,
    model_str: str,
    df_filtered: pd.DataFrame,
    df_catalog: pd.DataFrame,
    summary_list: List[Dict],
    fixed_num_time_bins: Optional[int] = None,
    analysis_mode: str = "gbm+lat",
    run_overrides: Optional[GRBRunOverrides] = None,
) -> None:
    if "trigger_name" in df_filtered.columns:
        bnname = df_filtered["trigger_name"].iloc[idx]
    elif "bnname" in df_filtered.columns:
        bnname = df_filtered["bnname"].iloc[idx]
    else:
        raise ValueError("数据中没有找到trigger_name或bnname列")

    df_lat = pd.read_excel(
        session.fermilat_grb_xls,
        sheet_name="GCN",
        index_col="trigname",
    )
    row_lat = df_lat.loc[bnname]
    if isinstance(row_lat, pd.DataFrame):
        row_lat = row_lat.iloc[0]
    grb_name = re.sub(r"\s+", "", row_lat["gcn_name"])
    if run_overrides is not None and run_overrides.grbname is not None:
        grb_name = re.sub(r"\s+", "", str(run_overrides.grbname))

    grb_dir = os.path.join(session.data_dir, bnname)
    result_grb_root = ensure_dir(
        os.path.join(session.result_root, grb_name)
    )
    bn_workspace = ensure_dir(os.path.join(result_grb_root, bnname))

    if "lat" in analysis_mode.lower() and getattr(
        session, "copy_extended_lat_to_bn_dir", True
    ):
        copy_extended_lat_to_bn_workspace(
            grb_name,
            bn_workspace,
            getattr(session, "extended_lat_data_root", None),
        )

    os.chdir(result_grb_root)

    t0, t1, ra, dec = _determine_time_interval_and_position(
        bnname,
        df_catalog,
    )

    if bnname == "bn231222310":
        t0, t1 = float(0.1), float(85)
        logg.info("此GRB为230910110，探测器有效时间间隔为0-85s")
    if bnname == "bn221023862":
        t0, t1 = float(8), float(30)
    if bnname == "bn231129799":
        t0, t1 = float(0.1), float(8)

    if run_overrides is not None:
        if run_overrides.t0 is not None:
            t0 = float(run_overrides.t0)
        if run_overrides.t1 is not None:
            t1 = float(run_overrides.t1)
        if run_overrides.ra is not None:
            ra = float(run_overrides.ra)
        if run_overrides.dec is not None:
            dec = float(run_overrides.dec)

    run_lat_three_ml = getattr(
        session, "lat_extended_three_ml_pipeline", False
    )
    if run_overrides is not None and getattr(
        run_overrides, "lat_three_ml_full", None
    ) is not None:
        run_lat_three_ml = bool(run_overrides.lat_three_ml_full)

    if "lat" in analysis_mode.lower() and run_lat_three_ml:
        try:
            row_lat_sel = df_lat.loc[bnname]
            if isinstance(row_lat_sel, pd.DataFrame):
                row_lat_sel = row_lat_sel.iloc[0]
            trig_met = float(row_lat_sel["trigger_met"])
            row_cat_early = df_catalog.loc[bnname]
            t05_sel = (
                float(row_cat_early["t90_start"])
                if float(row_cat_early["t90_start"]) >= 0
                else 0.0
            )
            t90_sel = float(row_cat_early["t90"])
            sel_irfs = getattr(session, "default_lat_irfs", None)
            if sel_irfs is None:
                sel_irfs = "p8_transient010e"
            selection_lat = {
                "tstart": float(t0),
                "tstop": float(t1),
                "ra": float(ra),
                "dec": float(dec),
                "trigger_time": trig_met,
                "data_type": "Extended",
                "Emin": 100.0,
                "Emax": 100000.0,
                "irfs": sel_irfs,
                "t05": t05_sel,
                "t90": t90_sel,
            }
            from .lat_extended_three_ml import (
                run_lat_extended_three_ml_pipeline,
            )

            ext_root = getattr(
                session, "extended_lat_data_root", ""
            )
            extended_src = os.path.join(
                os.path.expanduser(ext_root), grb_name
            )
            run_lat_extended_three_ml_pipeline(
                bn_dir=bn_workspace,
                bn_name=bnname,
                grb_name=grb_name,
                selection=selection_lat,
                result_parent=result_grb_root,
                extended_data_dir=extended_src,
            )
        except Exception as exc:  # noqa: BLE001
            log(f"{bnname}: LAT Extended threeML 流水线失败: {exc}")
            traceback.print_exc()
            _append_text_report(
                session.result_root,
                "lat_three_ml_pipeline_errors.txt",
                f"{pd.Timestamp.now()}  {bnname}: {exc}\n{traceback.format_exc()}\n",
            )

    time_bins, num_time_bins, duration = _determine_time_bins(
        t0,
        t1,
        fixed_num_time_bins=fixed_num_time_bins,
    )

    time_bins_list = [time_bins]

    if bnname == "bn231129799":
        time_bins_list = [
            0.10,
            0.26,
            0.61,
            0.85,
            2.25,
            2.73,
            2.90,
            4.43,
            5.07,
            5.75,
            6.18,
            6.83,
            7.21,
            8.00,
        ]
        num_time_bins = len(time_bins_list) - 1
        duration = time_bins_list[-1] - time_bins_list[0]

    if bnname == "bn250313607":
        time_bins_list = [
            np.array([1.09, 3, 7, 10, 15, 25]),
            np.array([260, 270, 276, 285, 299]),
        ]
        num_time_bins = sum(len(tb) - 1 for tb in time_bins_list)
        duration = time_bins_list[-1][-1] - time_bins_list[0][0]

    lat_plugin: Optional[OGIPLike] = None
    if "lat" in analysis_mode.lower():
        lat_plugin = process_lat_data(
            bnname,
            t0,
            t1,
            ensure_dir(os.path.join(session.result_root)),
        )
        if lat_plugin is None:
            log(f"{bnname}: LAT 数据处理失败，跳过该 GRB 的分析。")
            _append_text_report(
                session.result_root,
                "lat_processing_skipped.txt",
                f"{pd.Timestamp.now()}  {bnname}: LAT 数据处理失败，跳过该 GRB\n",
            )
            return

    row_catalog = df_catalog.loc[bnname]
    background_interval = _build_background_interval_string(
        row_catalog,
        bnname,
    )

    dets, *_ = _select_gbm_detectors(grb_dir)
    log(f"{bnname}: 探测器 {dets}")

    try:
        from .lightcurves import (
            detectors_for_lightcurve,
            parse_background_interval_tuple,
            plot_gbm_lat_lightcurve_figure,
        )

        nai_pair, bgo_id = detectors_for_lightcurve(dets)
        bkg_parts = parse_background_interval_tuple(background_interval)
        gbm_lc_start = float(t0) - 5.0
        gbm_lc_stop = float(t1) + 5.0
        active_lc = f"{float(t0):.6g}-{float(t1):.6g}"
        include_lat_panel = (
            "lat" in analysis_mode.lower() and lat_plugin is not None
        )
        out_lc = os.path.join(
            session.result_root,
            grb_name,
            f"{bnname}_lightcurve.png",
        )
        plot_gbm_lat_lightcurve_figure(
            bnname,
            grb_name=grb_name,
            trigger_met=None,
            gbm_start=gbm_lc_start,
            gbm_stop=gbm_lc_stop,
            nai_detector_ids=nai_pair,
            bgo_detector_id=bgo_id,
            active_interval=active_lc,
            background_intervals=bkg_parts,
            include_lat=include_lat_panel,
            out_path=out_lc,
        )
        log(f"{bnname}: 光变曲线已保存 {out_lc}")
    except Exception as exc:  # noqa: BLE001
        log(f"{bnname}: 光变曲线绘制跳过 ({exc})")

    bin_counter = 0
    for tb in time_bins_list:
        for i in range(len(tb) - 1):
            bin_start = tb[i]
            bin_end = tb[i + 1]
            bin_counter += 1
            log(
                f"正在处理时间bin {bin_counter}/{num_time_bins}: "
                f"[{bin_start:.2f}, {bin_end:.2f}]"
            )
            source_interval = f"{bin_start:.2f}-{bin_end:.2f}"

            plugins: List[OGIPLike] = []
            time_series = {}
            for det in dets:
                plugin = _build_gbm_plugin_for_detector(
                    det=det,
                    grb_dir=grb_dir,
                    background_interval=background_interval,
                    bin_start=bin_start,
                    bin_end=bin_end,
                    source_interval=source_interval,
                    time_series=time_series,
                )
                if plugin is not None:
                    plugins.append(plugin)

            if not plugins:
                log(f"警告: {bnname} 在该时间bin内未能成功构建任何 GBM 插件，跳过。")
                continue

            if lat_plugin is not None:
                datalist = DataList(lat_plugin, *plugins)
            else:
                datalist = DataList(*plugins)

            for model_str in [
                "band",
                "blackbody",
                "band+bb",
            ]:
                result_dir = ensure_dir(
                    os.path.join(session.result_root, grb_name, model_str)
                )

                log(
                    f"正在运行 {bnname} {model_str} "
                    f"在时间bin {bin_start:.2f}-{bin_end:.2f} 的贝叶斯拟合"
                )
                summary_entry = _run_bayesian_analysis_for_model(
                    model_str=model_str,
                    grb_name=grb_name,
                    bnname=bnname,
                    ra=ra,
                    dec=dec,
                    datalist=datalist,
                    plugins=plugins,
                    lat_plugin=lat_plugin,
                    dets=dets,
                    result_dir=result_dir,
                    bin_start=bin_start,
                    bin_end=bin_end,
                    duration=duration,
                    analysis_mode=analysis_mode,
                )
                summary_list.append(summary_entry)
                log(
                    f"正在运行 {bnname} {model_str} "
                    f"在时间bin {bin_start:.2f}-{bin_end:.2f} 的贝叶斯拟合结束"
                )
            log(
                f"{bnname} 时间bin [{bin_start:.2f}-{bin_end:.2f}] 所有模型拟合完成"
            )
