# -*- coding: utf-8 -*-
"""对外项目封装入口。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from .config import GRBProjectConfig, GRBRunOverrides, run_overrides_from_config
from .session import set_result_root, session


def apply_project_config(cfg: GRBProjectConfig) -> None:
    session.data_dir = cfg.data_dir
    session.catalog_xls = cfg.catalog_xls
    session.merged_xls = cfg.merged_xls
    session.fermilat_grb_xls = cfg.fermilat_grb_xls
    session.lat_extended_three_ml_pipeline = bool(cfg.lat_three_ml_full) if cfg.lat_three_ml_full is not None else session.lat_extended_three_ml_pipeline
    session.default_lat_irfs = getattr(cfg, "default_lat_irfs", session.default_lat_irfs)
    session.extended_lat_data_root = getattr(cfg, "extended_lat_data_root", session.extended_lat_data_root)
    session.copy_extended_lat_to_bn_dir = getattr(cfg, "copy_extended_lat_to_bn_dir", session.copy_extended_lat_to_bn_dir)
    set_result_root(cfg.result_root, cfg.summary_csv_name)


def _mode_includes(analysis_mode: object, token: str) -> bool:
    return token.lower() in str(analysis_mode or "").lower()


def _as_text(value: object, default: str = "") -> str:
    text = str(value).strip() if value is not None else ""
    return text or default


def _catalog_frame_by_bnname(catalog: pd.DataFrame) -> pd.DataFrame:
    if "bnname" not in catalog.columns:
        return catalog
    try:
        return catalog.set_index("bnname", drop=False)
    except Exception:
        return catalog


def _lookup_catalog_row(catalog: pd.DataFrame, target: str) -> pd.Series:
    target = str(target).strip()
    if target in catalog.index:
        row = catalog.loc[target]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return row
    if "bnname" in catalog.columns:
        matches = catalog[catalog["bnname"].astype(str) == target]
        if not matches.empty:
            return matches.iloc[0]
    raise KeyError(f"目录表中找不到目标 {target!r}")


def _lookup_lat_catalog_row(bnname: str) -> Optional[pd.Series]:
    try:
        df = pd.read_excel(session.fermilat_grb_xls, sheet_name="GCN", index_col="trigname")
    except Exception:
        return None
    try:
        row = df.loc[bnname]
    except KeyError:
        return None
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return row


def _normalize_time_segments(time_segments: object) -> list[dict]:
    normalized: list[dict] = []
    if not isinstance(time_segments, Sequence) or isinstance(time_segments, (str, bytes)):
        return normalized
    for idx, item in enumerate(time_segments):
        if not isinstance(item, dict):
            continue
        if "tstart" in item and "tstop" in item:
            tstart = float(item["tstart"])
            tstop = float(item["tstop"])
            normalized.append(
                {
                    "name": item.get("name"),
                    "tag": _as_text(item.get("tag"), _as_text(item.get("name"), f"seg{idx + 1}")),
                    "tstart": tstart,
                    "tstop": tstop,
                }
            )
            continue
        if "start" in item and "stop" in item:
            tstart = float(item["start"])
            tstop = float(item["stop"])
            normalized.append(
                {
                    "name": item.get("name"),
                    "tag": _as_text(item.get("tag"), _as_text(item.get("name"), f"seg{idx + 1}")),
                    "tstart": tstart,
                    "tstop": tstop,
                }
            )
    return normalized


def _build_result_metadata(
    *,
    grb_name: str,
    bnname: str,
    analysis_mode: str,
    result_dir: Path,
    t0: float,
    t1: float,
    ra: float,
    dec: float,
    background_interval: str,
    active_interval: str,
    time_segments: Sequence[dict],
    models: Optional[Sequence[str]],
    special_yaml: Optional[str],
    special_burst_name: Optional[str],
    lat_three_ml_full: Optional[bool],
    plot_joint_lightcurve: Optional[bool] = None,
    gbm_start: Optional[float] = None,
    gbm_stop: Optional[float] = None,
    gbm_display_pad_before_s: Optional[float] = None,
    gbm_display_pad_after_s: Optional[float] = None,
    lightcurve_include_lat: Optional[bool] = None,
    lightcurve_lat_prob_threshold: Optional[float] = None,
    lightcurve_nai_bands_kev: Optional[Sequence[tuple[float, float]]] = None,
    lightcurve_bgo_band_kev: Optional[tuple[float, float]] = None,
    lightcurve_active_interval: Optional[str] = None,
    lightcurve_background_intervals: Optional[Sequence[str]] = None,
    lightcurve_path: Optional[str] = None,
    lightcurve_request: Optional[dict] = None,
) -> dict:
    payload = {
        "grb_name": grb_name,
        "bnname": bnname,
        "analysis_mode": analysis_mode,
        "result_dir": str(result_dir),
        "t0": float(t0),
        "t1": float(t1),
        "duration": float(t1 - t0),
        "ra": float(ra),
        "dec": float(dec),
        "background_interval": background_interval,
        "active_interval": active_interval,
        "num_time_segments": len(time_segments),
        "time_segments": _normalize_time_segments(time_segments),
        "models": list(models) if models is not None else None,
        "special_yaml": special_yaml,
        "special_burst_name": special_burst_name,
        "lat_three_ml_full": lat_three_ml_full,
        "plot_joint_lightcurve": plot_joint_lightcurve,
        "gbm_start": gbm_start,
        "gbm_stop": gbm_stop,
        "gbm_display_pad_before_s": gbm_display_pad_before_s,
        "gbm_display_pad_after_s": gbm_display_pad_after_s,
        "lightcurve_include_lat": lightcurve_include_lat,
        "lightcurve_lat_prob_threshold": lightcurve_lat_prob_threshold,
        "lightcurve_nai_bands_kev": [list(v) for v in lightcurve_nai_bands_kev] if lightcurve_nai_bands_kev is not None else None,
        "lightcurve_bgo_band_kev": list(lightcurve_bgo_band_kev) if lightcurve_bgo_band_kev is not None else None,
        "lightcurve_active_interval": lightcurve_active_interval,
        "lightcurve_background_intervals": list(lightcurve_background_intervals) if lightcurve_background_intervals is not None else None,
        "lightcurve_path": lightcurve_path,
        "lightcurve_request": lightcurve_request,
    }
    (result_dir / "run_metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def _default_model_list(project_config: Optional[GRBProjectConfig], run_overrides: Optional[GRBRunOverrides]) -> list[str]:
    override_models = getattr(run_overrides, "models", None)
    if override_models:
        return [str(m).strip() for m in override_models if str(m).strip()]
    if project_config is not None and project_config.models:
        return [str(m).strip() for m in project_config.models if str(m).strip()]
    if project_config is not None and getattr(project_config, "model_preset", None):
        preset = str(project_config.model_preset).strip().lower()
        if preset == "gbm_only":
            return ["band", "comp", "blackbody"]
    return ["band", "comp", "blackbody"]


def _resolve_gbm_source_dir(bnname: str, grb_name: str) -> str:
    candidates = [
        Path(session.data_dir) / grb_name / bnname,
        Path(session.data_dir) / bnname,
        Path(session.data_dir) / grb_name,
        Path(session.data_dir),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[-1])


def _resolve_lat_extended_dir(grb_name: str) -> str:
    return str(Path(session.extended_lat_data_root).expanduser() / grb_name)


def _result_subdirs(run_root: Path, grb_name: str) -> tuple[Path, Path, Path]:
    burst_dir = run_root / grb_name
    return burst_dir, burst_dir / "gbm", burst_dir / "lat"


def run_joint_lightcurve(
    *,
    bnname: str,
    grb_name: str,
    result_dir: Path,
    lat_dir: Path,
    trigger_met: Optional[float],
    data_dir: str,
    active_interval: str,
    background_interval: str,
    fixed_num_time_bins: Optional[int],
    gbm_start: Optional[float],
    gbm_stop: Optional[float],
    gbm_display_pad_before_s: Optional[float],
    gbm_display_pad_after_s: Optional[float],
    plot_joint_lightcurve: bool,
    include_lat: bool,
    lat_prob_threshold: float,
    nai_bands_kev: Sequence[tuple[float, float]],
    bgo_band_kev: tuple[float, float],
    special_time_segments: Optional[Sequence[dict]] = None,
    out_path: Optional[Path] = None,
) -> Path:
    from .lightcurves import parse_background_interval_tuple, plot_gbm_lat_lightcurve_figure

    result_dir.mkdir(parents=True, exist_ok=True)
    bg_intervals_lc = (
        parse_background_interval_tuple(background_interval)
        if background_interval
        else ("-24--5", "350-400")
    )
    lc_out = Path(out_path or (result_dir / f"{grb_name}_lightcurve.png"))
    plot_gbm_lat_lightcurve_figure(
        bnname,
        grb_name=grb_name,
        trigger_met=trigger_met,
        data_dir=data_dir,
        lat_prob_bn_dir=lat_dir / bnname,
        include_lat=include_lat,
        lat_prob_threshold=lat_prob_threshold,
        gbm_start=gbm_start,
        gbm_stop=gbm_stop,
        active_interval=active_interval or "0-100",
        background_intervals=bg_intervals_lc,
        fixed_num_time_bins=fixed_num_time_bins,
        special_time_segments=special_time_segments,
        display_window_pad_before_s=gbm_display_pad_before_s if gbm_display_pad_before_s is not None else 1.0,
        display_window_pad_after_s=gbm_display_pad_after_s,
        nai_bands_kev=nai_bands_kev,
        bgo_band_kev=bgo_band_kev,
        out_path=lc_out,
    )
    return lc_out


def _build_lat_selection(
    *,
    bnname: str,
    grb_name: str,
    ra: float,
    dec: float,
    t0: float,
    t1: float,
    catalog_row: pd.Series,
    lat_row: Optional[pd.Series],
) -> dict:
    trigger_met = float(lat_row.get("trigger_met", 0.0)) if lat_row is not None else 0.0
    selection = {
        "ra": float(ra),
        "dec": float(dec),
        "trigger_time": trigger_met,
        "tstart": float(t0),
        "tstop": float(t1),
        "t90": float(catalog_row.get("t90", t1 - t0)),
        "t05": float(catalog_row.get("t05", 0.0)),
        "data_type": "Extended",
        "Emin": 100.0,
        "Emax": 100000.0,
        "irfs": getattr(session, "default_lat_irfs", "p8_transient010e"),
        "bnname": bnname,
        "grb_name": grb_name,
    }
    if lat_row is not None and "ra,dec" in lat_row.index:
        try:
            ra_str, dec_str = str(lat_row["ra,dec"]).split(",")
            selection["ra"] = float(ra_str)
            selection["dec"] = float(dec_str)
        except Exception:
            pass
    return selection


def _run_lat_analysis(
    *,
    bnname: str,
    grb_name: str,
    result_dir: Path,
    lat_dir: Path,
    analysis_mode: str,
    t0: float,
    t1: float,
    ra: float,
    dec: float,
    catalog_row: pd.Series,
    project_config: Optional[GRBProjectConfig],
    run_overrides: Optional[GRBRunOverrides],
    fixed_num_time_bins: Optional[int],
) -> dict:
    from .lat_processing import process_lat_data
    from .logging_utils import log

    if not _mode_includes(analysis_mode, "lat"):
        return {"lat_plugin": None, "analysis_segments": []}

    lat_row = _lookup_lat_catalog_row(bnname)
    if lat_row is None:
        log(f"{bnname}: 未找到 LAT 目录行，跳过 LAT 联合拟合")
        return {"lat_plugin": None, "analysis_segments": []}

    use_full_lat = bool(getattr(run_overrides, "lat_three_ml_full", None))
    if getattr(project_config, "lat_three_ml_full", None) is not None:
        use_full_lat = bool(project_config.lat_three_ml_full)
    use_full_lat = use_full_lat or bool(session.lat_extended_three_ml_pipeline)

    selection = _build_lat_selection(
        bnname=bnname,
        grb_name=grb_name,
        ra=ra,
        dec=dec,
        t0=t0,
        t1=t1,
        catalog_row=catalog_row,
        lat_row=lat_row,
    )

    lat_result: dict = {"lat_plugin": None, "analysis_segments": []}
    if use_full_lat:
        from .io_utils import copy_extended_lat_to_bn_dir

        if session.copy_extended_lat_to_bn_dir:
            try:
                copy_extended_lat_to_bn_dir(
                    grb_name,
                    str(lat_dir / bnname),
                    session.extended_lat_data_root,
                )
            except Exception as exc:  # noqa: BLE001
                log(f"{bnname}: 复制 Extended LAT 数据失败，继续尝试直接读取: {exc}")

        try:
            from .lat_extended_three_ml import run_lat_extended_three_ml_pipeline

            lat_result = run_lat_extended_three_ml_pipeline(
                bn_dir=str(lat_dir),
                bn_name=bnname,
                grb_name=grb_name,
                selection=selection,
                result_parent=str(lat_dir),
                extended_data_dir=_resolve_lat_extended_dir(grb_name),
                fixed_num_time_bins=fixed_num_time_bins,
                analysis_bin_start=t0,
                analysis_bin_end=t1,
                return_lat_plugin=True,
            )
            lat_result.setdefault("analysis_segments", [])
            lat_result.setdefault("lat_plugin", None)
            return lat_result
        except Exception as exc:  # noqa: BLE001
            log(f"{bnname}: LAT 完整流水线失败，回退到基础 LAT 处理: {exc}")

    try:
        lat_plugin = process_lat_data(bnname, t0, t1, str(lat_dir))
        lat_result = {
            "lat_plugin": lat_plugin,
            "analysis_segments": [{"tstart": float(t0), "tstop": float(t1), "tag": "analysis_full"}],
        }
    except Exception as exc:  # noqa: BLE001
        log(f"{bnname}: LAT 基础处理失败: {exc}")
        lat_result = {"lat_plugin": None, "analysis_segments": []}
    return lat_result


def _run_gbm_analysis(
    *,
    grb_name: str,
    bnname: str,
    result_dir: Path,
    gbm_dir: Path,
    source_dir: str,
    analysis_mode: str,
    background_interval: str,
    source_interval: str,
    time_segments: Sequence[dict],
    models: Sequence[str],
    ra: float,
    dec: float,
    catalog_row: pd.Series,
    project_config: Optional[GRBProjectConfig],
    run_overrides: Optional[GRBRunOverrides],
    fixed_num_time_bins: Optional[int],
) -> list[dict]:
    from .bayesian_fit import _run_bayesian_analysis_for_model
    from .gbm_core import _build_gbm_plugin_for_detector
    from .gbm_detector_selection import select_gbm_detectors
    from .logging_utils import log

    try:
        from threeML import DataList
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"threeML 不可用，无法执行 GBM 拟合: {exc}") from exc

    dets, *_geom = select_gbm_detectors(source_dir)
    if not dets:
        raise RuntimeError(f"{bnname}: 未选择到可用的 GBM 探测器")

    summary_rows: list[dict] = []
    for seg_idx, seg in enumerate(time_segments, start=1):
        bin_start = float(seg["tstart"])
        bin_end = float(seg["tstop"])
        seg_tag = _as_text(seg.get("tag"), f"seg{seg_idx}")
        time_series: dict = {}
        plugins: list = []
        for det in dets:
            plugin = _build_gbm_plugin_for_detector(
                det=det,
                grb_dir=source_dir,
                background_interval=background_interval,
                bin_start=bin_start,
                bin_end=bin_end,
                source_interval=source_interval,
                time_series=time_series,
                output_dir=str(gbm_dir),
            )
            if plugin is not None:
                plugins.append(plugin)
        if not plugins:
            raise RuntimeError(f"{bnname}: 时间段 {bin_start:g}-{bin_end:g} 没有成功构建 GBM 插件")

        lat_info = _run_lat_analysis(
            bnname=bnname,
            grb_name=grb_name,
            result_dir=result_dir,
            lat_dir=result_dir / "lat",
            analysis_mode=analysis_mode,
            t0=bin_start,
            t1=bin_end,
            ra=ra,
            dec=dec,
            catalog_row=catalog_row,
            project_config=project_config,
            run_overrides=run_overrides,
            fixed_num_time_bins=fixed_num_time_bins,
        )
        lat_plugin = lat_info.get("lat_plugin")
        if lat_plugin is not None:
            log(
                f"{bnname}: 本 bin [{bin_start:g}, {bin_end:g}] 已接入 LAT plugin，"
                "将执行 GBM+LAT 联合拟合"
            )
        else:
            log(
                f"{bnname}: 本 bin [{bin_start:g}, {bin_end:g}] 未接入 LAT plugin，"
                "当前仅执行 GBM 拟合"
            )
        datalist = DataList(lat_plugin, *plugins) if lat_plugin is not None else DataList(*plugins)
        duration = float(bin_end - bin_start)
        for model_str in models:
            model_dir = result_dir / model_str
            model_dir.mkdir(parents=True, exist_ok=True)
            try:
                row = _run_bayesian_analysis_for_model(
                    model_str=model_str,
                    grb_name=grb_name,
                    bnname=bnname,
                    ra=float(ra),
                    dec=float(dec),
                    datalist=datalist,
                    plugins=plugins,
                    lat_plugin=lat_plugin,
                    dets=list(dets),
                    result_dir=str(model_dir),
                    bin_start=bin_start,
                    bin_end=bin_end,
                    duration=duration,
                    analysis_mode=analysis_mode,
                )
                row.update(
                    {
                        "result_dir": str(model_dir),
                        "source_dir": source_dir,
                        "segment_tag": seg_tag,
                        "detectors": ",".join(dets),
                        "analysis_status": "completed",
                        "analysis_note": "GBM+LAT 联合拟合已恢复",
                    }
                )
                summary_rows.append(row)
            except Exception as exc:  # noqa: BLE001
                log(f"{bnname}: 模型 {model_str} 在 {bin_start:g}-{bin_end:g} 拟合失败: {exc}")
                summary_rows.append(
                    {
                        "grb_name": grb_name,
                        "bnname": bnname,
                        "model": model_str,
                        "analysis_mode": analysis_mode,
                        "AIC": None,
                        "BIC": None,
                        "Flux(erg/cm2/s)": None,
                        "FTot(erg/cm2/s)": None,
                        "Fluence(erg/cm2)": None,
                        "FBB(erg/cm2/s)": None,
                        "log_marginal_likelihood": None,
                        "duration": duration,
                        "num_time_bins": len(time_segments),
                        "bin_start_time": bin_start,
                        "bin_end_time": bin_end,
                        "bin_duration": duration,
                        "result_dir": str(model_dir),
                        "source_dir": source_dir,
                        "segment_tag": seg_tag,
                        "detectors": ",".join(dets),
                        "analysis_status": "failed",
                        "analysis_note": str(exc),
                    }
                )
    return summary_rows


def run_single_analysis(
    target_grbs=None,
    analysis_mode=None,
    result_root=None,
    summary_csv_name=None,
    fixed_num_time_bins=None,
    run_overrides: Optional[GRBRunOverrides] = None,
) -> pd.DataFrame:
    from .gbm_core import (
        _build_background_interval_string,
        _build_lat_analysis_segments,
        _determine_time_interval_and_position,
        resolve_active_interval_from_special_and_catalog,
    )
    from .io_utils import read_catalog
    from .logging_utils import log
    from .special_bursts import _load_special_burst_config
    from .summary_export import (
        _append_time_bin_info,
        _compute_time_bin_analysis,
        _plot_model_comparison,
        _save_all_models_per_grb,
        _save_time_bin_analysis,
    )

    catalog = read_catalog()
    catalog_by_bn = _catalog_frame_by_bnname(catalog)

    run_root = Path(result_root or session.result_root).expanduser()
    run_root.mkdir(parents=True, exist_ok=True)
    summary_name = summary_csv_name or session.summary_csv_name
    set_result_root(str(run_root), summary_name)

    mode = analysis_mode or getattr(session.extra.get("project_config"), "analysis_mode", "gbm+lat")
    project_config = session.extra.get("project_config")

    if target_grbs is None:
        if "bnname" in catalog.columns:
            target_list = [str(v) for v in catalog["bnname"].dropna().astype(str).tolist()]
        else:
            target_list = [str(v) for v in catalog.index.astype(str).tolist()]
    else:
        target_list = [str(v) for v in target_grbs]

    summary_rows: list[dict] = []
    models = _default_model_list(project_config, run_overrides)
    for target in target_list:
        try:
            catalog_row = _lookup_catalog_row(catalog_by_bn if target in catalog_by_bn.index else catalog, target)
        except KeyError as exc:
            log(str(exc))
            continue

        bnname = _as_text(catalog_row.get("bnname"), target)
        if not bnname:
            bnname = target
        grb_name_default = _as_text(catalog_row.get("gcn_name"), bnname).replace(" ", "")
        special_yaml = getattr(run_overrides, "special_yaml", None) or getattr(project_config, "special_yaml", None)
        special_burst_name = getattr(run_overrides, "special_burst_name", None) or getattr(project_config, "special_burst_name", None)
        special_cfg = _load_special_burst_config(special_yaml, bnname, grb_name_default)

        if special_burst_name and _as_text(special_cfg.get("name")) not in {special_burst_name, ""}:
            special_cfg = {}

        if special_cfg.get("name"):
            grb_name = _as_text(special_cfg.get("name"), grb_name_default).replace(" ", "")
        else:
            grb_name = grb_name_default

        interval_frame = catalog_by_bn if bnname in catalog_by_bn.index else catalog
        try:
            t0, t1, ra, dec = _determine_time_interval_and_position(bnname, interval_frame)
        except Exception:
            t90_start = float(catalog_row.get("t90_start", 0.0))
            t90 = float(catalog_row.get("t90", 0.0))
            t0 = 0.0 if t90_start < 0 else t90_start
            t1 = t90_start + t90
            ra = float(catalog_row.get("ra", 0.0))
            dec = float(catalog_row.get("dec", 0.0))

        if run_overrides is not None and run_overrides.t0 is not None:
            t0 = float(run_overrides.t0)
        if run_overrides is not None and run_overrides.t1 is not None:
            t1 = float(run_overrides.t1)
        if run_overrides is not None and run_overrides.ra is not None:
            ra = float(run_overrides.ra)
        if run_overrides is not None and run_overrides.dec is not None:
            dec = float(run_overrides.dec)
        if project_config is not None and getattr(project_config, "grbname", None):
            grb_name = _as_text(getattr(project_config, "grbname"), grb_name)

        active_interval = resolve_active_interval_from_special_and_catalog(
            special_cfg.get("time_segments"),
            catalog_row,
            str(special_cfg.get("active_interval", "")),
        )
        background_interval = _as_text(
            special_cfg.get("background_interval"),
            _build_background_interval_string(catalog_row, bnname)
            if {"back_interval_low_start", "back_interval_low_stop", "back_interval_high_start", "back_interval_high_stop"}.issubset(catalog_row.index)
            else "",
        )
        raw_segments = special_cfg.get("time_segments")
        if raw_segments:
            time_segments = _normalize_time_segments(raw_segments)
        else:
            time_segments = _build_lat_analysis_segments(
                bnname,
                t0,
                t1,
                fixed_num_time_bins=fixed_num_time_bins,
            )
        if not time_segments:
            time_segments = [{"tstart": float(t0), "tstop": float(t1), "tag": "analysis_full"}]

        result_dir, gbm_dir, lat_dir = _result_subdirs(run_root, grb_name)
        result_dir.mkdir(parents=True, exist_ok=True)
        gbm_dir.mkdir(parents=True, exist_ok=True)
        lat_dir.mkdir(parents=True, exist_ok=True)
        metadata = _build_result_metadata(
            grb_name=grb_name,
            bnname=bnname,
            analysis_mode=mode,
            result_dir=result_dir,
            t0=t0,
            t1=t1,
            ra=ra,
            dec=dec,
            background_interval=background_interval,
            active_interval=active_interval,
            time_segments=time_segments,
            models=models,
            special_yaml=special_yaml,
            special_burst_name=special_burst_name,
            lat_three_ml_full=bool(getattr(run_overrides, "lat_three_ml_full", None) or getattr(project_config, "lat_three_ml_full", None) or session.lat_extended_three_ml_pipeline),
            plot_joint_lightcurve=bool(getattr(run_overrides, "plot_joint_lightcurve", None) or getattr(project_config, "plot_joint_lightcurve", None)),
            gbm_start=getattr(run_overrides, "gbm_start", None) if run_overrides is not None else getattr(project_config, "gbm_start", None),
            gbm_stop=getattr(run_overrides, "gbm_stop", None) if run_overrides is not None else getattr(project_config, "gbm_stop", None),
            gbm_display_pad_before_s=getattr(run_overrides, "gbm_display_pad_before_s", None) if run_overrides is not None else getattr(project_config, "gbm_display_pad_before_s", None),
            gbm_display_pad_after_s=getattr(run_overrides, "gbm_display_pad_after_s", None) if run_overrides is not None else getattr(project_config, "gbm_display_pad_after_s", None),
        )

        source_dir = _resolve_gbm_source_dir(bnname, grb_name)
        lat_info = _run_lat_analysis(
            bnname=bnname,
            grb_name=grb_name,
            result_dir=result_dir,
            lat_dir=lat_dir,
            analysis_mode=mode,
            t0=t0,
            t1=t1,
            ra=ra,
            dec=dec,
            catalog_row=catalog_row,
            project_config=project_config,
            run_overrides=run_overrides,
            fixed_num_time_bins=fixed_num_time_bins,
        )
        lat_plugin = lat_info.get("lat_plugin")

        if _mode_includes(mode, "gbm") or _mode_includes(mode, "lat"):
            try:
                gbm_rows = _run_gbm_analysis(
                    grb_name=grb_name,
                    bnname=bnname,
                    result_dir=result_dir,
                    gbm_dir=gbm_dir,
                    source_dir=source_dir,
                    analysis_mode=mode,
                    background_interval=background_interval,
                    source_interval=active_interval,
                    time_segments=time_segments,
                    models=models,
                    ra=ra,
                    dec=dec,
                    catalog_row=catalog_row,
                    project_config=project_config,
                    run_overrides=run_overrides,
                    fixed_num_time_bins=fixed_num_time_bins,
                )
                summary_rows.extend(gbm_rows)
            except Exception as exc:  # noqa: BLE001
                log(f"{bnname}: GBM 联合拟合失败: {exc}")
                summary_rows.append(
                    {
                        **metadata,
                        "result_root": str(run_root),
                        "summary_csv_name": summary_name,
                        "analysis_status": "failed",
                        "analysis_note": str(exc),
                        "source_dir": source_dir,
                    }
                )
        else:
            summary_rows.append(
                {
                    **metadata,
                    "result_root": str(run_root),
                    "summary_csv_name": summary_name,
                    "analysis_status": "skipped",
                    "analysis_note": f"analysis_mode={mode} 未请求 GBM/LAT 拟合",
                    "source_dir": source_dir,
                }
            )
        # 光变曲线（勾选 plot_joint_lightcurve 时在拟合完成后绘制）
        want_lc = bool(
            getattr(run_overrides, "plot_joint_lightcurve", None)
            or getattr(project_config, "plot_joint_lightcurve", None)
        )
        if want_lc:
            try:
                from .lightcurves import (
                    parse_background_interval_tuple,
                    plot_gbm_lat_lightcurve_figure,
                )

                lat_row_lc = _lookup_lat_catalog_row(bnname)
                trigger_met_lc = (
                    float(lat_row_lc.get("trigger_met", 0.0))
                    if lat_row_lc is not None
                    else None
                )
                special_segs_lc = special_cfg.get("time_segments") or None
                lc_out = result_dir / f"{grb_name}_lightcurve.png"
                gbm_start_lc = getattr(run_overrides, "gbm_start", None) or getattr(
                    project_config, "gbm_start", None
                )
                gbm_stop_lc = getattr(run_overrides, "gbm_stop", None) or getattr(
                    project_config, "gbm_stop", None
                )
                gbm_pad_before = getattr(run_overrides, "gbm_display_pad_before_s", None) or getattr(
                    project_config, "gbm_display_pad_before_s", None
                )
                gbm_pad_after = getattr(run_overrides, "gbm_display_pad_after_s", None) or getattr(
                    project_config, "gbm_display_pad_after_s", None
                )
                include_lat_lc = bool(getattr(run_overrides, "lightcurve_include_lat", None) if run_overrides is not None and getattr(run_overrides, "lightcurve_include_lat", None) is not None else getattr(project_config, "lightcurve_include_lat", None) if project_config is not None and getattr(project_config, "lightcurve_include_lat", None) is not None else True)
                lat_prob_threshold_lc = float(getattr(run_overrides, "lightcurve_lat_prob_threshold", None) if run_overrides is not None and getattr(run_overrides, "lightcurve_lat_prob_threshold", None) is not None else getattr(project_config, "lightcurve_lat_prob_threshold", None) if project_config is not None and getattr(project_config, "lightcurve_lat_prob_threshold", None) is not None else 0.9)
                nai_bands_lc = getattr(run_overrides, "lightcurve_nai_bands_kev", None) if run_overrides is not None and getattr(run_overrides, "lightcurve_nai_bands_kev", None) is not None else getattr(project_config, "lightcurve_nai_bands_kev", None) if project_config is not None and getattr(project_config, "lightcurve_nai_bands_kev", None) is not None else ((8.0, 50.0), (50.0, 300.0))
                bgo_band_lc = getattr(run_overrides, "lightcurve_bgo_band_kev", None) if run_overrides is not None and getattr(run_overrides, "lightcurve_bgo_band_kev", None) is not None else getattr(project_config, "lightcurve_bgo_band_kev", None) if project_config is not None and getattr(project_config, "lightcurve_bgo_band_kev", None) is not None else (300.0, 38000.0)
                lc_out = run_joint_lightcurve(
                    bnname=bnname,
                    grb_name=grb_name,
                    result_dir=result_dir,
                    lat_dir=lat_dir,
                    trigger_met=trigger_met_lc,
                    data_dir=session.data_dir,
                    active_interval=active_interval or "0-100",
                    background_interval=background_interval,
                    fixed_num_time_bins=fixed_num_time_bins,
                    gbm_start=gbm_start_lc,
                    gbm_stop=gbm_stop_lc,
                    gbm_display_pad_before_s=gbm_pad_before,
                    gbm_display_pad_after_s=gbm_pad_after,
                    plot_joint_lightcurve=bool(getattr(run_overrides, "plot_joint_lightcurve", None) or getattr(project_config, "plot_joint_lightcurve", None)),
                    include_lat=include_lat_lc,
                    lat_prob_threshold=lat_prob_threshold_lc,
                    nai_bands_kev=tuple(tuple(map(float, b)) for b in nai_bands_lc),
                    bgo_band_kev=(float(bgo_band_lc[0]), float(bgo_band_lc[1])),
                    special_time_segments=special_segs_lc,
                    out_path=lc_out,
                )
                log(f"{bnname}: 光变曲线已保存 → {lc_out}")
            except Exception as exc:  # noqa: BLE001
                log(f"{bnname}: 光变曲线绘制失败（不影响拟合结果）: {exc}")

        log(f"{bnname}: 已完成单次分析目录 {result_dir}")

    df_summary = pd.DataFrame(summary_rows)
    summary_path = run_root / summary_name
    if not df_summary.empty:
        _append_time_bin_info(df_summary)
        _save_all_models_per_grb(df_summary)
        time_analysis = _compute_time_bin_analysis(df_summary)
        _save_time_bin_analysis(time_analysis)
        _plot_model_comparison(df_summary)
    df_summary.to_csv(summary_path, index=False)
    session.extra["last_run_summary"] = df_summary.to_dict(orient="records")
    session.extra["last_summary_csv"] = str(summary_path)
    log(f"单次分析摘要已写入 {summary_path}")
    return df_summary


class GRBProject:
    def __init__(self, config: Optional[GRBProjectConfig] = None, **config_fields: object) -> None:
        self.config = config or GRBProjectConfig()
        for key, val in config_fields.items():
            if not hasattr(self.config, key):
                raise TypeError(f"GRBProjectConfig 无字段 {key!r}")
            setattr(self.config, key, val)

    @property
    def bnname(self) -> Optional[str]:
        return self.config.bnname

    @bnname.setter
    def bnname(self, value: Optional[str]) -> None:
        self.config.bnname = value

    @property
    def grbname(self) -> Optional[str]:
        return self.config.grbname

    @grbname.setter
    def grbname(self, value: Optional[str]) -> None:
        self.config.grbname = value

    @property
    def t0(self) -> Optional[float]:
        return self.config.t0

    @t0.setter
    def t0(self, value: Optional[float]) -> None:
        self.config.t0 = value

    @property
    def t1(self) -> Optional[float]:
        return self.config.t1

    @t1.setter
    def t1(self, value: Optional[float]) -> None:
        self.config.t1 = value

    @property
    def ra(self) -> Optional[float]:
        return self.config.ra

    @ra.setter
    def ra(self, value: Optional[float]) -> None:
        self.config.ra = value

    @property
    def dec(self) -> Optional[float]:
        return self.config.dec

    @dec.setter
    def dec(self, value: Optional[float]) -> None:
        self.config.dec = value

    def run(
        self,
        target_grbs: Optional[Sequence[str]] = None,
        analysis_mode: Optional[str] = None,
        result_root: Optional[str] = None,
        summary_csv_name: Optional[str] = None,
        session_log: bool = False,
        run_overrides: Optional[GRBRunOverrides] = None,
    ) -> None:
        apply_project_config(self.config)
        mode = analysis_mode if analysis_mode is not None else self.config.analysis_mode
        run_root = result_root or self.config.result_root
        summary_name = summary_csv_name or self.config.summary_csv_name
        set_result_root(run_root, summary_name)

        tg: Optional[list[str]]
        if target_grbs is not None:
            tg = list(target_grbs)
        elif self.config.bnname:
            tg = [self.config.bnname]
        else:
            tg = None

        ov = run_overrides if run_overrides is not None else run_overrides_from_config(self.config)
        session.extra["project_config"] = self.config
        from .pipeline import main
        main(
            tg,
            mode,
            result_root=run_root,
            summary_csv_name=summary_name,
            session_log=session_log,
            fixed_num_time_bins=self.config.fixed_num_time_bins,
            run_overrides=ov,
        )
