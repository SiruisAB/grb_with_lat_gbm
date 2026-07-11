# -*- coding: utf-8 -*-
"""GRB 工作台 Streamlit 入口，提供单次分析与光变曲线两个页面。"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

_RUNTIME_ROOT = Path(os.environ.get("GBMTEST_RUNTIME_DIR", "/tmp/gbmtest-runtime"))
_ASTROMODELS_CONFIG_DIR = _RUNTIME_ROOT / "astromodels"
_THREEML_CONFIG_DIR = _RUNTIME_ROOT / "threeml"
for _path in (
    _RUNTIME_ROOT,
    _RUNTIME_ROOT / "mplconfig",
    _RUNTIME_ROOT / "cache",
    _ASTROMODELS_CONFIG_DIR,
    _THREEML_CONFIG_DIR,
):
    _path.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(_RUNTIME_ROOT / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(_RUNTIME_ROOT / "cache"))
os.environ.setdefault("ASTROMODELS_CONFIG", str(_ASTROMODELS_CONFIG_DIR))
os.environ.setdefault("THREEML_CONFIG", str(_THREEML_CONFIG_DIR))

_ASTROMODELS_CONFIG_FILE = _ASTROMODELS_CONFIG_DIR / "defaults.yml"
if not _ASTROMODELS_CONFIG_FILE.exists():
    _ASTROMODELS_CONFIG_FILE.write_text(
        "logging:\n  path: /tmp/gbmtest-runtime/astromodels/log\n",
        encoding="utf-8",
    )

_THREEML_CONFIG_FILE = _THREEML_CONFIG_DIR / "defaults.yml"
if not _THREEML_CONFIG_FILE.exists():
    _THREEML_CONFIG_FILE.write_text(
        "logging:\n  path: /tmp/gbmtest-runtime/threeml/log\n",
        encoding="utf-8",
    )

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grb_project.config import GRBProjectConfig, GRBRunOverrides
from grb_project.project import GRBProject, _build_result_metadata, run_joint_lightcurve
from grb_project.session import set_result_root
from grb_project.special_bursts import _normalize_time_segments, load_special_burst_config

DEFAULT_NAI_BANDS = ((8.0, 50.0), (50.0, 300.0))
DEFAULT_BGO_BAND = (300.0, 38000.0)
DEFAULT_BACKGROUND_LOW = "-24--5"
DEFAULT_BACKGROUND_HIGH = "350-400"
SPECIAL_BURSTS_YAML = Path(__file__).with_name("special_bursts.yaml")


def _load_catalog(cfg: GRBProjectConfig) -> pd.DataFrame:
    df = pd.read_excel(cfg.catalog_xls, sheet_name="fermigbrst")
    if "trigger_name" in df.columns:
        return df.set_index("trigger_name", drop=False)
    if "bnname" in df.columns:
        return df.set_index("bnname", drop=False)
    raise RuntimeError("目录表缺少 trigger_name / bnname 列")


def _load_fermilat_catalog(cfg: GRBProjectConfig) -> pd.DataFrame:
    df = pd.read_excel(cfg.fermilat_grb_xls, sheet_name="GCN", index_col="trigname")
    df.index = df.index.astype(str)
    return df


def _default_target(df: pd.DataFrame) -> str:
    first = str(df.index.astype(str).tolist()[0]).strip()
    if not first:
        raise RuntimeError("目录表第一个目标为空")
    return first


def _safe_text(value: object, default: str = "") -> str:
    text = str(value).strip() if value is not None else ""
    return text or default


def _catalog_year_from_label(label: str) -> Optional[int]:
    text = str(label).strip()
    if not text.startswith("bn") or len(text) < 4:
        return None
    year_part = text[2:4]
    if not year_part.isdigit():
        return None
    return 2000 + int(year_part)


def _recent_catalog_rows(df_catalog: pd.DataFrame) -> pd.DataFrame:
    years = df_catalog.index.astype(str).map(_catalog_year_from_label)
    recent = df_catalog.loc[years >= 2022].copy()
    return recent if not recent.empty else df_catalog.copy()


def _parse_models(text: str) -> list[str]:
    models = [item.strip() for item in str(text).split(",") if item.strip()]
    return models or ["band", "comp", "blackbody"]


def _parse_band_pairs(text: str, default: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    items: list[tuple[float, float]] = []
    for raw in str(text).split(","):
        token = raw.strip()
        if not token:
            continue
        parts = [part.strip() for part in token.replace("～", "-").split("-") if part.strip()]
        if len(parts) != 2:
            continue
        try:
            lo = float(parts[0])
            hi = float(parts[1])
        except Exception:
            continue
        if hi > lo:
            items.append((lo, hi))
    return items or list(default)


def _parse_background_intervals(text: str, default: Sequence[str]) -> tuple[str, str]:
    cleaned = [item.strip() for item in str(text).split(",") if item.strip()]
    if len(cleaned) >= 2:
        return cleaned[0], cleaned[1]
    if len(cleaned) == 1:
        return cleaned[0], default[1]
    return (default[0], default[1])


def _make_catalog_summary(df_recent: pd.DataFrame) -> pd.DataFrame:
    preview_cols = [
        col
        for col in ("trigger_name", "bnname", "gcn_name", "t90_start", "t90", "ra", "dec")
        if col in df_recent.columns
    ]
    return df_recent.loc[:, preview_cols] if preview_cols else df_recent


def _build_project(base_cfg: GRBProjectConfig, *, analysis_mode: str, result_root: str, summary_csv_name: str) -> GRBProject:
    project = GRBProject(base_cfg)
    project.config.analysis_mode = analysis_mode
    project.config.result_root = result_root
    project.config.summary_csv_name = summary_csv_name
    return project


def _interval_text(value: object, fallback: str) -> str:
    text = _safe_text(value, fallback)
    return text or fallback


def _smart_active_interval(catalog_row: pd.Series, lat_row: Optional[pd.Series], default_t0: float, default_t1: float) -> str:
    for candidate_row in (lat_row, catalog_row):
        if candidate_row is None:
            continue
        for key in ("active_interval", "source_interval", "analysis_interval", "time_interval"):
            text = _safe_text(candidate_row.get(key), "")
            if text:
                return text
        t05 = candidate_row.get("t05")
        t95 = candidate_row.get("t95")
        try:
            if t05 is not None and t95 is not None:
                t05f = float(t05)
                t95f = float(t95)
                if t95f > t05f:
                    return f"{t05f:g}-{t95f:g}"
        except Exception:
            pass
    return f"{default_t0:g}-{default_t1:g}"


def _background_defaults(catalog_row: pd.Series, lat_row: Optional[pd.Series], target: str) -> tuple[str, str]:
    for candidate_row in (lat_row, catalog_row):
        if candidate_row is None:
            continue
        combined = _safe_text(candidate_row.get("background_intervals"), "")
        if combined:
            parsed = [item.strip() for item in combined.split(",") if item.strip()]
            if len(parsed) >= 2:
                return parsed[0], parsed[1]
        background_interval = _safe_text(candidate_row.get("background_interval"), "")
        if background_interval:
            parsed = [item.strip() for item in background_interval.split(",") if item.strip()]
            if len(parsed) >= 2:
                return parsed[0], parsed[1]
        try:
            special_cfg = load_special_burst_config(SPECIAL_BURSTS_YAML, target, _safe_text(lat_row.get("gcn_name") if lat_row is not None else catalog_row.get("gcn_name", target), target))
        except Exception:
            special_cfg = {}
        if special_cfg.get("background_interval"):
            parsed = [item.strip() for item in str(special_cfg.get("background_interval")).split(",") if item.strip()]
            if len(parsed) >= 2:
                return parsed[0], parsed[1]
    low_start = _safe_text(catalog_row.get("back_interval_low_start"), DEFAULT_BACKGROUND_LOW.split("-")[0])
    low_stop = _safe_text(catalog_row.get("back_interval_low_stop"), DEFAULT_BACKGROUND_LOW.split("-")[-1])
    high_start = _safe_text(catalog_row.get("back_interval_high_start"), DEFAULT_BACKGROUND_HIGH.split("-")[0])
    high_stop = _safe_text(catalog_row.get("back_interval_high_stop"), DEFAULT_BACKGROUND_HIGH.split("-")[-1])
    low = f"{low_start}-{low_stop}"
    high = f"{high_start}-{high_stop}"
    if low == "-":
        low = DEFAULT_BACKGROUND_LOW
    if high == "-":
        high = DEFAULT_BACKGROUND_HIGH
    return low, high


def _target_defaults(catalog_row: pd.Series, lat_row: Optional[pd.Series], target: str) -> dict[str, object]:
    grb_name = _safe_text(lat_row.get("gcn_name") if lat_row is not None else catalog_row.get("gcn_name", target), target).replace(" ", "")
    trigger_met = float(lat_row.get("trigger_met", 0.0)) if lat_row is not None else 0.0
    t90_start = float(catalog_row.get("t90_start", 0.0))
    t90 = float(catalog_row.get("t90", 8.0))
    ra = float(catalog_row.get("ra", 0.0))
    dec = float(catalog_row.get("dec", 0.0))
    background_low, background_high = _background_defaults(catalog_row, lat_row, target)
    default_t0 = 0.0 if t90_start < 0 else t90_start
    default_t1 = t90_start + t90
    active_interval = _smart_active_interval(catalog_row, lat_row, default_t0, default_t1)
    try:
        special_cfg = load_special_burst_config(SPECIAL_BURSTS_YAML, target, grb_name)
    except Exception:
        special_cfg = {}
    if special_cfg.get("active_interval"):
        active_interval = str(special_cfg.get("active_interval")).strip() or active_interval
    if special_cfg.get("background_interval"):
        parsed = [item.strip() for item in str(special_cfg.get("background_interval")).split(",") if item.strip()]
        if len(parsed) >= 2:
            background_low, background_high = parsed[0], parsed[1]
    return {
        "grb_name": grb_name,
        "trigger_met": trigger_met,
        "t0": default_t0,
        "t1": default_t1,
        "t90_start": t90_start,
        "t90": t90,
        "ra": ra,
        "dec": dec,
        "background_low": background_low,
        "background_high": background_high,
        "active_interval": active_interval,
        "gbm_start": default_t0,
        "gbm_stop": default_t1,
    }


def _sync_target_defaults(st, *, prefix: str, target: str, catalog_row: pd.Series, lat_row: Optional[pd.Series]) -> dict[str, object]:
    defaults = _target_defaults(catalog_row, lat_row, target)
    state_key = f"{prefix}_selected_target"
    if st.session_state.get(state_key) != target:
        st.session_state[state_key] = target
        for field, value in defaults.items():
            st.session_state[f"{prefix}_{field}"] = value
    return defaults


def _show_result_summary(st, *, result_path: Path, metadata: Optional[dict] = None) -> None:
    st.success(f"结果已保存到：{result_path}")
    if metadata is not None:
        with st.expander("本次使用的参数", expanded=False):
            st.json(metadata)


def _analysis_succeeded(summary: pd.DataFrame) -> bool:
    return (
        not summary.empty
        and "analysis_status" in summary.columns
        and summary["analysis_status"].astype(str).eq("completed").any()
    )


def _single_result_dir(result_root: str, grb_name: str) -> Path:
    return Path(result_root).expanduser() / grb_name


def _show_target_context(st, *, defaults: dict[str, object]) -> None:
    cols = st.columns(4)
    cols[0].metric("GRB 名称", str(defaults["grb_name"]))
    cols[1].metric("trigger_met", f"{float(defaults['trigger_met']):.3f}")
    cols[2].metric("T90 起点", f"{float(defaults['t90_start']):.2f}")
    cols[3].metric("T90", f"{float(defaults['t90']):.2f}")
    cols = st.columns(2)
    cols[0].write(f"**RA**：{float(defaults['ra']):.6f}")
    cols[1].write(f"**DEC**：{float(defaults['dec']):.6f}")


def _run_single_analysis_page(
    *,
    st,
    base_cfg: GRBProjectConfig,
    df_catalog: pd.DataFrame,
    fermilat_catalog: pd.DataFrame,
    target_options: Sequence[str],
    default_target: str,
) -> None:
    st.markdown("#### 单次分析")
    st.caption("结果会直接写到 `result_root/GRB名/`，并复用现有 `project.py` 分析流程。")

    target = st.selectbox(
        "目标 GRB",
        options=target_options,
        index=target_options.index(default_target),
        key="single_target",
    )
    catalog_row = df_catalog.loc[target]
    if isinstance(catalog_row, pd.DataFrame):
        catalog_row = catalog_row.iloc[0]
    lat_row = fermilat_catalog.loc[target] if target in fermilat_catalog.index else None
    if isinstance(lat_row, pd.DataFrame):
        lat_row = lat_row.iloc[0]
    defaults = _sync_target_defaults(st, prefix="single", target=target, catalog_row=catalog_row, lat_row=lat_row)

    _show_target_context(st, defaults=defaults)

    analysis_mode = st.selectbox(
        "analysis_mode",
        options=["gbm", "lat", "gbm+lat"],
        index=2,
        key="single_analysis_mode",
    )
    data_dir = st.text_input("GBM 数据根目录 data_dir", value=base_cfg.data_dir, key="single_data_dir")
    result_root = st.text_input("结果根目录 result_root", value=base_cfg.result_root, key="single_result_root")
    summary_csv_name = st.text_input("summary_csv_name", value=base_cfg.summary_csv_name, key="single_summary_csv_name")
    session_log = st.checkbox("写入 session.log", value=False, key="single_session_log")
    fixed_num_time_bins = st.number_input("fixed_num_time_bins", min_value=0, value=0, step=1, key="single_fixed_num_time_bins")

    col1, col2 = st.columns(2)
    with col1:
        grb_name = st.text_input("grb_name", value=str(defaults["grb_name"]), key="single_grb_name")
        trigger_met = st.number_input("trigger_met", value=float(defaults["trigger_met"]), step=0.1, key="single_trigger_met")
        t0 = st.number_input("t0", value=float(st.session_state.get("single_t0", defaults["gbm_start"])), step=0.5, key="single_t0")
        t1 = st.number_input("t1", value=float(st.session_state.get("single_t1", defaults["gbm_stop"])), step=0.5, key="single_t1")
    with col2:
        ra = st.number_input("ra", value=float(defaults["ra"]), step=0.1, format="%.6f", key="single_ra")
        dec = st.number_input("dec", value=float(defaults["dec"]), step=0.1, format="%.6f", key="single_dec")
        gbm_start = st.number_input("gbm_start", value=float(defaults["gbm_start"]), step=0.5, key="single_gbm_start")
        gbm_stop = st.number_input("gbm_stop", value=float(defaults["gbm_stop"]), step=0.5, key="single_gbm_stop")

    active_interval = st.text_input(
        "active_interval",
        value=str(st.session_state.get("single_active_interval", defaults["active_interval"])),
        key="single_active_interval",
    )
    bg_low_col, bg_high_col = st.columns(2)
    with bg_low_col:
        background_low = st.text_input(
            "background_low",
            value=str(st.session_state.get("single_background_low", defaults["background_low"])),
            key="single_background_low",
        )
    with bg_high_col:
        background_high = st.text_input(
            "background_high",
            value=str(st.session_state.get("single_background_high", defaults["background_high"])),
            key="single_background_high",
        )
    gbm_pad_before = st.number_input("display_window_pad_before_s", value=1.0, step=0.5, key="single_gbm_pad_before_s")
    gbm_pad_after = st.number_input("display_window_pad_after_s", value=1.0, step=0.5, key="single_gbm_pad_after_s")

    col1, col2 = st.columns(2)
    with col1:
        lat_three_ml_full = st.checkbox("lat_three_ml_full", value=False, key="single_lat_three_ml_full")
        plot_joint_lightcurve = st.checkbox("plot_joint_lightcurve", value=True, key="single_plot_joint_lightcurve")
    with col2:
        models_text = st.text_input("models（逗号分隔）", value="band,comp,blackbody", key="single_models_text")
        special_yaml = st.text_input("special_yaml", value="", key="single_special_yaml")
        special_burst_name = st.text_input("special_burst_name", value="", key="single_special_burst_name")

    if st.button("开始运行", type="primary", key="single_run_button"):
        run_overrides = GRBRunOverrides(
            grbname=grb_name.strip() or str(defaults["grb_name"]),
            t0=float(t0),
            t1=float(t1),
            trigger_met=float(trigger_met),
            ra=float(ra),
            dec=float(dec),
            lat_three_ml_full=bool(lat_three_ml_full),
            plot_joint_lightcurve=bool(plot_joint_lightcurve),
            gbm_start=float(gbm_start),
            gbm_stop=float(gbm_stop),
            gbm_display_pad_before_s=float(gbm_pad_before),
            gbm_display_pad_after_s=float(gbm_pad_after),
            lightcurve_active_interval=str(active_interval),
            lightcurve_background_intervals=(str(background_low), str(background_high)),
            models=_parse_models(models_text),
            special_yaml=special_yaml.strip() or None,
            special_burst_name=special_burst_name.strip() or None,
        )
        result_root_value = str(Path(result_root).expanduser())
        project = _build_project(base_cfg, analysis_mode=analysis_mode, result_root=result_root_value, summary_csv_name=summary_csv_name)
        project.config.data_dir = str(Path(data_dir).expanduser())
        project.config.fixed_num_time_bins = int(fixed_num_time_bins) or None
        project.config.bnname = target
        project.config.grbname = grb_name.strip() or str(defaults["grb_name"])
        project.config.t0 = run_overrides.t0
        project.config.t1 = run_overrides.t1
        project.config.trigger_met = run_overrides.trigger_met
        project.config.ra = run_overrides.ra
        project.config.dec = run_overrides.dec
        project.config.lat_three_ml_full = run_overrides.lat_three_ml_full
        project.config.plot_joint_lightcurve = run_overrides.plot_joint_lightcurve
        project.config.gbm_start = run_overrides.gbm_start
        project.config.gbm_stop = run_overrides.gbm_stop
        project.config.gbm_display_pad_before_s = run_overrides.gbm_display_pad_before_s
        project.config.gbm_display_pad_after_s = run_overrides.gbm_display_pad_after_s
        project.config.models = run_overrides.models
        project.config.special_yaml = run_overrides.special_yaml
        project.config.special_burst_name = run_overrides.special_burst_name
        project.config.lightcurve_active_interval = str(active_interval)
        project.config.lightcurve_background_intervals = (str(background_low), str(background_high))
        set_result_root(result_root_value, summary_csv_name)

        with st.spinner("正在运行单次分析..."):
            try:
                summary = project.run(
                    target_grbs=[target],
                    analysis_mode=analysis_mode,
                    result_root=result_root_value,
                    summary_csv_name=summary_csv_name,
                    session_log=session_log,
                    run_overrides=run_overrides,
                )
                final_result_dir = _single_result_dir(
                    result_root_value,
                    grb_name.strip() or str(defaults["grb_name"]),
                )
                if _analysis_succeeded(summary):
                    _show_result_summary(st, result_path=final_result_dir)
                else:
                    st.error(f"分析未成功完成，请查看摘要和日志：{final_result_dir}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"运行失败：{exc}")


def _run_lightcurve_page(
    *,
    st,
    base_cfg: GRBProjectConfig,
    df_catalog: pd.DataFrame,
    fermilat_catalog: pd.DataFrame,
    target_options: Sequence[str],
    default_target: str,
) -> None:
    st.markdown("#### 光变曲线")
    st.caption("此页直接调用后端同一套绘图逻辑，保存路径与元数据会写回结果目录。")

    target = st.selectbox(
        "目标 GRB",
        options=target_options,
        index=target_options.index(default_target),
        key="lc_target",
    )
    catalog_row = df_catalog.loc[target]
    if isinstance(catalog_row, pd.DataFrame):
        catalog_row = catalog_row.iloc[0]
    lat_row = fermilat_catalog.loc[target] if target in fermilat_catalog.index else None
    if isinstance(lat_row, pd.DataFrame):
        lat_row = lat_row.iloc[0]
    defaults = _sync_target_defaults(st, prefix="lc", target=target, catalog_row=catalog_row, lat_row=lat_row)

    _show_target_context(st, defaults=defaults)

    result_root = st.text_input("结果根目录 result_root", value=base_cfg.result_root, key="lc_result_root")
    data_dir = st.text_input("GBM 数据根目录 data_dir", value=base_cfg.data_dir, key="lc_data_dir")
    grb_name = st.text_input("grb_name", value=str(defaults["grb_name"]), key="lc_grb_name")
    trigger_met = st.number_input("trigger_met", value=float(defaults["trigger_met"]), step=0.1, key="lc_trigger_met")
    plot_joint_lightcurve = st.checkbox("plot_joint_lightcurve", value=True, key="lc_plot_joint_lightcurve")
    include_lat = st.checkbox("include_lat", value=True, key="lc_include_lat")

    col1, col2 = st.columns(2)
    with col1:
        gbm_start = st.number_input("gbm_start", value=float(defaults["gbm_start"]), step=0.5, key="lc_gbm_start")
        gbm_stop = st.number_input("gbm_stop", value=float(defaults["gbm_stop"]), step=0.5, key="lc_gbm_stop")
        active_interval = st.text_input("active_interval", value=str(st.session_state.get("lc_active_interval", defaults["active_interval"])), key="lc_active_interval")
        gbm_pad_before = st.number_input("display_window_pad_before_s", value=1.0, step=0.5, key="lc_gbm_pad_before_s")
        gbm_pad_after = st.number_input("display_window_pad_after_s", value=1.0, step=0.5, key="lc_gbm_pad_after_s")
    with col2:
        background_low = st.text_input("background_low", value=str(st.session_state.get("lc_background_low", defaults["background_low"])), key="lc_background_low")
        background_high = st.text_input("background_high", value=str(st.session_state.get("lc_background_high", defaults["background_high"])), key="lc_background_high")
        fixed_num_time_bins = st.number_input("fixed_num_time_bins", min_value=0, value=0, step=1, key="lc_fixed_num_time_bins")
        lat_prob_threshold = st.number_input("lat_prob_threshold", min_value=0.0, max_value=1.0, value=0.9, step=0.01, key="lc_lat_prob_threshold")
        nai_bands_text = st.text_input("nai_bands_kev", value="8-50,50-300", key="lc_nai_bands")
        bgo_band_text = st.text_input("bgo_band_kev", value="300-38000", key="lc_bgo_band")

    special_yaml = st.text_input("special_yaml", value="", key="lc_special_yaml")
    special_burst_name = st.text_input("special_burst_name", value="", key="lc_special_burst_name")

    if st.button("生成光变曲线", type="primary", key="lc_run_button"):
        if not plot_joint_lightcurve:
            st.info("已关闭 plot_joint_lightcurve，未生成光变曲线。")
            return
        result_root_value = str(Path(result_root).expanduser())
        project = _build_project(base_cfg, analysis_mode="gbm+lat", result_root=result_root_value, summary_csv_name=base_cfg.summary_csv_name)
        project.config.data_dir = str(Path(data_dir).expanduser())
        project.config.bnname = target
        project.config.grbname = grb_name.strip() or str(defaults["grb_name"])
        project.config.plot_joint_lightcurve = plot_joint_lightcurve
        project.config.gbm_start = float(gbm_start)
        project.config.gbm_stop = float(gbm_stop)
        project.config.gbm_display_pad_before_s = float(gbm_pad_before)
        project.config.gbm_display_pad_after_s = float(gbm_pad_after)
        project.config.lightcurve_include_lat = bool(include_lat)
        project.config.lightcurve_lat_prob_threshold = float(lat_prob_threshold)
        project.config.lightcurve_nai_bands_kev = _parse_band_pairs(nai_bands_text, DEFAULT_NAI_BANDS)
        project.config.lightcurve_bgo_band_kev = tuple(_parse_band_pairs(bgo_band_text, [DEFAULT_BGO_BAND])[0])
        project.config.lightcurve_active_interval = str(active_interval)
        project.config.lightcurve_background_intervals = (str(background_low), str(background_high))
        project.config.special_yaml = special_yaml.strip() or None
        project.config.special_burst_name = special_burst_name.strip() or None
        set_result_root(result_root_value, base_cfg.summary_csv_name)

        result_dir = Path(result_root_value) / (grb_name.strip() or str(defaults["grb_name"]))
        lat_dir = result_dir / "lat"
        (lat_dir / target).mkdir(parents=True, exist_ok=True)
        nai_bands = _parse_band_pairs(nai_bands_text, DEFAULT_NAI_BANDS)
        bgo_band = _parse_band_pairs(bgo_band_text, [DEFAULT_BGO_BAND])[0]
        background_tuple = (str(background_low), str(background_high))
        special_segments = None
        if special_yaml.strip():
            try:
                special_cfg = load_special_burst_config(special_yaml.strip() or None, target, project.config.grbname)
                special_segments = _normalize_time_segments(special_cfg.get("time_segments")) or None
                if not active_interval.strip() and special_cfg.get("active_interval"):
                    active_interval = str(special_cfg.get("active_interval")).strip()
                if special_cfg.get("background_interval"):
                    parsed_bg = _parse_background_intervals(str(special_cfg.get("background_interval")), background_tuple)
                    background_tuple = parsed_bg
            except Exception:
                special_segments = None

        with st.spinner("正在生成光变曲线..."):
            try:
                lc_out = run_joint_lightcurve(
                    bnname=target,
                    grb_name=grb_name.strip() or str(defaults["grb_name"]),
                    result_dir=result_dir,
                    lat_dir=lat_dir,
                    trigger_met=float(trigger_met),
                    data_dir=str(Path(data_dir).expanduser()),
                    active_interval=str(active_interval),
                    background_interval=",".join(background_tuple),
                    fixed_num_time_bins=int(fixed_num_time_bins) or None,
                    gbm_start=float(gbm_start),
                    gbm_stop=float(gbm_stop),
                    gbm_display_pad_before_s=float(gbm_pad_before),
                    gbm_display_pad_after_s=float(gbm_pad_after),
                    plot_joint_lightcurve=bool(plot_joint_lightcurve),
                    include_lat=bool(include_lat),
                    lat_prob_threshold=float(lat_prob_threshold),
                    nai_bands_kev=tuple(nai_bands),
                    bgo_band_kev=(float(bgo_band[0]), float(bgo_band[1])),
                    special_time_segments=special_segments,
                    out_path=Path(result_root_value)
                    / (grb_name.strip() or str(defaults["grb_name"]))
                    / f"{grb_name.strip() or str(defaults['grb_name'])}_lightcurve.png",
                )
                metadata = _build_result_metadata(
                    grb_name=grb_name.strip() or str(defaults["grb_name"]),
                    bnname=target,
                    analysis_mode="gbm+lat",
                    result_dir=result_dir,
                    t0=float(gbm_start),
                    t1=float(gbm_stop),
                    ra=float(defaults["ra"]),
                    dec=float(defaults["dec"]),
                    background_interval=",".join(background_tuple),
                    active_interval=str(active_interval),
                    time_segments=special_segments or [{"tstart": float(gbm_start), "tstop": float(gbm_stop), "tag": "lightcurve"}],
                    models=None,
                    special_yaml=special_yaml.strip() or None,
                    special_burst_name=special_burst_name.strip() or None,
                    lat_three_ml_full=False,
                    plot_joint_lightcurve=bool(plot_joint_lightcurve),
                    gbm_start=float(gbm_start),
                    gbm_stop=float(gbm_stop),
                    gbm_display_pad_before_s=float(gbm_pad_before),
                    gbm_display_pad_after_s=float(gbm_pad_after),
                    lightcurve_include_lat=bool(include_lat),
                    lightcurve_lat_prob_threshold=float(lat_prob_threshold),
                    lightcurve_nai_bands_kev=tuple(nai_bands),
                    lightcurve_bgo_band_kev=(float(bgo_band[0]), float(bgo_band[1])),
                    lightcurve_active_interval=str(active_interval),
                    lightcurve_background_intervals=background_tuple,
                    lightcurve_path=str(lc_out),
                    lightcurve_request={
                        "bnname": target,
                        "grb_name": grb_name.strip() or str(defaults["grb_name"]),
                        "trigger_met": float(trigger_met),
                        "data_dir": str(Path(data_dir).expanduser()),
                        "lat_prob_bn_dir": str(lat_dir / target),
                        "gbm_start": float(gbm_start),
                        "gbm_stop": float(gbm_stop),
                        "active_interval": str(active_interval),
                        "background_intervals": list(background_tuple),
                        "fixed_num_time_bins": int(fixed_num_time_bins) or None,
                        "special_time_segments": special_segments,
                        "display_window_pad_before_s": float(gbm_pad_before),
                        "display_window_pad_after_s": float(gbm_pad_after),
                        "include_lat": bool(include_lat),
                        "lat_prob_threshold": float(lat_prob_threshold),
                        "nai_bands_kev": [list(v) for v in nai_bands],
                        "bgo_band_kev": list(bgo_band),
                    },
                )
                _show_result_summary(st, result_path=Path(lc_out).resolve(), metadata=metadata)
                st.image(str(lc_out), caption=f"{grb_name.strip() or str(defaults['grb_name'])} 光变曲线", use_container_width=True)
            except Exception as exc:  # noqa: BLE001
                st.error(f"光变曲线生成失败：{exc}")


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="GRB 工作台", layout="wide")
    st.title("GRB 工作台")
    st.caption("单页内切换单次分析与光变曲线页面，前后端共用同一套参数与结果目录约定。")

    base_cfg = GRBProjectConfig()
    try:
        df_catalog = _load_catalog(base_cfg)
        fermilat_catalog = _load_fermilat_catalog(base_cfg)
        df_recent = _recent_catalog_rows(df_catalog)
        default_target = _default_target(df_recent)
    except Exception as exc:  # noqa: BLE001
        st.error(f"无法加载目录表：{exc}")
        return

    target_options = df_recent.index.astype(str).tolist()
    st.sidebar.header("页面切换")
    page = st.sidebar.radio("页面", options=["单次分析", "光变曲线"], index=0)
    st.sidebar.markdown("---")
    st.sidebar.caption("第二页直接复用 `project.py` 与 `lightcurves.py` 的同一套后端约定。")

    st.subheader("目录表概况")
    with st.expander("查看最近样本", expanded=False):
        left, right = st.columns([1.2, 1])
        with left:
            st.caption(f"目录表总行数：{len(df_catalog)}")
            st.caption(f"2022 年后样本数：{len(df_recent)}")
        with right:
            st.caption(f"默认目标：{default_target}")
        st.dataframe(_make_catalog_summary(df_recent).head(10), use_container_width=True)

    if page == "单次分析":
        _run_single_analysis_page(
            st=st,
            base_cfg=base_cfg,
            df_catalog=df_catalog,
            fermilat_catalog=fermilat_catalog,
            target_options=target_options,
            default_target=default_target,
        )
    else:
        _run_lightcurve_page(
            st=st,
            base_cfg=base_cfg,
            df_catalog=df_catalog,
            fermilat_catalog=fermilat_catalog,
            target_options=target_options,
            default_target=default_target,
        )


if __name__ == "__main__":
    main()
