# -*- coding: utf-8 -*-
"""GRB 工作台 Streamlit 入口，提供单次分析与光变曲线两个页面。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
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
from grb_project import gbm_download, lat_download, lat_gcn_extract
from grb_project.io_utils import load_lat_catalog
from grb_project.project import GRBProject, _build_result_metadata, run_joint_lightcurve
from grb_project.session import set_result_root
from grb_project.special_bursts import (
    _normalize_time_segments,
    format_time_segments_text,
    load_special_burst_config,
    parse_time_segments_text,
    upsert_special_burst_config,
)

DEFAULT_NAI_BANDS = ((8.0, 50.0), (50.0, 300.0))
DEFAULT_BGO_BAND = (300.0, 38000.0)
DEFAULT_BACKGROUND_LOW = "-24--5"
DEFAULT_BACKGROUND_HIGH = "350-400"
SPECIAL_BURSTS_YAML = Path(__file__).with_name("special_bursts.yaml")
SUPPORTED_MODELS = (
    "band",
    "band_wide",
    "comp",
    "blackbody",
    "NDP",
    "pl",
    "SBPL",
    "2SBPL",
    "2SBPL_syn",
    "band+bb",
    "mbb",
    "band+mbb",
    "mbb+pl",
    "band+pl",
    "comp+bb",
    "comp+pl",
    "pl+bb",
    "band+bb+pl",
    "band+gauss",
)


def _show_lightcurve_plot(st, png_path: Path, *, caption: str) -> Path:
    html_path = Path(png_path).with_suffix(".html")
    if html_path.is_file():
        st.components.v1.html(html_path.read_text(encoding="utf-8"), height=900, scrolling=True)
        return html_path
    st.image(str(png_path), caption=caption, use_container_width=True)
    return Path(png_path)


def _load_catalog(cfg: GRBProjectConfig) -> pd.DataFrame:
    df = pd.read_excel(cfg.catalog_xls, sheet_name="fermigbrst")
    if "trigger_name" in df.columns:
        return df.set_index("trigger_name", drop=False)
    if "bnname" in df.columns:
        return df.set_index("bnname", drop=False)
    raise RuntimeError("目录表缺少 trigger_name / bnname 列")


def _load_fermilat_catalog(cfg: GRBProjectConfig) -> pd.DataFrame:
    return load_lat_catalog(cfg.fermilat_grb_xls)


def _default_target(df: pd.DataFrame) -> str:
    first = str(df.index.astype(str).tolist()[0]).strip()
    if not first:
        raise RuntimeError("目录表第一个目标为空")
    return first


def _safe_text(value: object, default: str = "") -> str:
    # 目录表里缺值是 NaN，而 str(float("nan")) == "nan" 是非空字符串，直接 `or default`
    # 兜不住，会把字面量 "nan" 当成真值一路传下去（GRB260208A 的本底窗就是这样变成
    # "nan-nan,nan-nan" 的）。所以 NaN/NaT 必须先判掉。
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        # pd.isna 对数组/列表会返回数组或抛歧义错误，这类值走下面的字符串逻辑
        pass
    text = str(value).strip()
    return text or default


def _catalog_year_from_label(label: str) -> Optional[int]:
    text = str(label).strip()
    if not text.startswith("bn") or len(text) < 4:
        return None
    year_part = text[2:4]
    if not year_part.isdigit():
        return None
    return 2000 + int(year_part)


def _joint_gbm_lat_rows(df_catalog: pd.DataFrame, fermilat_catalog: pd.DataFrame) -> pd.DataFrame:
    lat_targets = set(fermilat_catalog.index.astype(str))
    return df_catalog.loc[df_catalog.index.astype(str).isin(lat_targets)].copy()


def _filter_joint_catalog_rows(
    df_catalog: pd.DataFrame,
    fermilat_catalog: pd.DataFrame,
    year_from: Optional[int] = None,
) -> pd.DataFrame:
    joint = _joint_gbm_lat_rows(df_catalog, fermilat_catalog)
    if year_from is None:
        return joint
    years = joint.index.astype(str).map(_catalog_year_from_label)
    return joint.loc[years >= int(year_from)].copy()


def _validate_selected_models(selected_models: Sequence[str]) -> list[str]:
    models = [str(item).strip() for item in selected_models if str(item).strip()]
    if not models:
        raise ValueError("至少选择一个模型")
    return models


def _parse_download_list_text(text: str) -> list[str]:
    targets: list[str] = []
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        targets.append(gbm_download.normalize_bnname(line))
    return list(dict.fromkeys(targets))


def _run_gbm_download_request(
    *,
    mode: str,
    bnname: str,
    list_text: str,
    save_dir: str,
    csv_path: str,
    gbm_xls: str,
    year_from: int,
    year_to: Optional[int],
) -> list[str]:
    if mode == "单个 GRB":
        targets = [gbm_download.normalize_bnname(bnname)]
    elif mode == "GRB 列表":
        targets = _parse_download_list_text(list_text)
    else:
        targets = gbm_download.get_joint_target_list(
            csv_path=Path(csv_path).expanduser(),
            gbm_xls=Path(gbm_xls).expanduser(),
            year_from=int(year_from),
            year_to=year_to,
        )

    if not targets:
        raise ValueError("没有可下载的目标 GRB")
    gbm_download.download_target_list(targets, save_dir=Path(save_dir).expanduser())
    return targets


def _run_lat_gcn_refresh_request(
    *, archive_url: str, archive_tar: str, archive_dir: str, output_csv: str
):
    return lat_gcn_extract.refresh_gcn_lat_data(
        archive_url=archive_url,
        archive_tar_path=Path(archive_tar).expanduser(),
        archive_dir=Path(archive_dir).expanduser(),
        output_csv=Path(output_csv).expanduser(),
    )


def _run_lat_download_request(
    *,
    bnnames: Optional[Sequence[str]],
    year: Optional[int],
    month_from: int,
    catalog_xls: str,
    data_root: str,
):
    return lat_download.run_download(
        bnnames=bnnames,
        year=year,
        month_from=month_from,
        catalog_xls=Path(catalog_xls).expanduser(),
        data_root=Path(data_root).expanduser(),
    )


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


def _parse_background_intervals(text: str, default: Sequence[str]) -> tuple[str, ...]:
    """把逗号分隔的本底窗字符串拆成若干段，全部保留。

    special_bursts.yaml 中已有暴使用三段本底（例如
    ``-24--5,100-150,350-400``）。谱拟合那条路径原样透传整串，只有网页
    的光变分支会截断，截断后画出来的本底与实际拟合所用的本底不是同一个。
    """
    cleaned = [item.strip() for item in str(text).split(",") if item.strip()]
    if len(cleaned) >= 2:
        return tuple(cleaned)
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
    # 四个端点任缺其一就退回整段默认窗。原先按端点拆默认值是错的：
    # "-24--5".split("-") == ['', '24', '', '5']，[0]/[-1] 拼出来是 "-5" 而不是 "-24--5"。
    low_start = _safe_text(catalog_row.get("back_interval_low_start"))
    low_stop = _safe_text(catalog_row.get("back_interval_low_stop"))
    high_start = _safe_text(catalog_row.get("back_interval_high_start"))
    high_stop = _safe_text(catalog_row.get("back_interval_high_stop"))
    low = f"{low_start}-{low_stop}" if low_start and low_stop else DEFAULT_BACKGROUND_LOW
    high = f"{high_start}-{high_stop}" if high_start and high_stop else DEFAULT_BACKGROUND_HIGH
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


def _sync_special_editor_defaults(
    st,
    *,
    prefix: str,
    target: str,
    grb_name: str,
    active_interval: str,
    background_low: str,
    background_high: str,
    special_cfg: dict,
    segments_text: str,
) -> None:
    state_key = f"{prefix}_special_editor_selected_target"
    if st.session_state.get(state_key) == target:
        return
    st.session_state[state_key] = target
    st.session_state[f"{prefix}_special_editor_name"] = str(special_cfg.get("name") or grb_name).strip()
    st.session_state[f"{prefix}_special_editor_bnname"] = str(special_cfg.get("bnname") or target).strip()
    st.session_state[f"{prefix}_special_editor_active_interval"] = str(
        special_cfg.get("active_interval") or active_interval
    ).strip()
    st.session_state[f"{prefix}_special_editor_background_interval"] = str(
        special_cfg.get("background_interval") or f"{background_low},{background_high}"
    ).strip()
    st.session_state[f"{prefix}_special_editor_description"] = str(
        special_cfg.get("description") or "特殊时间分段与背景窗"
    )
    st.session_state[f"{prefix}_special_editor_segments_text"] = segments_text


def _render_special_burst_editor(
    st,
    *,
    prefix: str,
    target: str,
    grb_name: str,
    active_interval: str,
    background_low: str,
    background_high: str,
    expanded: bool = False,
) -> None:
    try:
        special_cfg = load_special_burst_config(str(SPECIAL_BURSTS_YAML), target, grb_name)
    except Exception:
        special_cfg = {}
    existing_segments = special_cfg.get("time_segments") or []
    existing_text = format_time_segments_text(existing_segments) if existing_segments else ""
    _sync_special_editor_defaults(
        st,
        prefix=prefix,
        target=target,
        grb_name=grb_name,
        active_interval=active_interval,
        background_low=background_low,
        background_high=background_high,
        special_cfg=special_cfg,
        segments_text=existing_text,
    )

    with st.expander("特殊时间分 bin 写入 special_bursts.yaml", expanded=expanded):
        st.caption("保存后会按 bnname 覆盖已有条目；其他特殊暴配置会保留。")
        yaml_path = st.text_input(
            "YAML 文件",
            value=str(SPECIAL_BURSTS_YAML),
            key=f"{prefix}_special_editor_yaml_path",
        )
        cols = st.columns(2)
        with cols[0]:
            yaml_name = st.text_input(
                "name",
                value=str(special_cfg.get("name") or grb_name).strip(),
                key=f"{prefix}_special_editor_name",
            )
            yaml_active = st.text_input(
                "active_interval",
                value=str(special_cfg.get("active_interval") or active_interval).strip(),
                key=f"{prefix}_special_editor_active_interval",
            )
        with cols[1]:
            yaml_bnname = st.text_input(
                "bnname",
                value=str(special_cfg.get("bnname") or target).strip(),
                key=f"{prefix}_special_editor_bnname",
            )
            yaml_background = st.text_input(
                "background_interval",
                value=str(special_cfg.get("background_interval") or f"{background_low},{background_high}").strip(),
                key=f"{prefix}_special_editor_background_interval",
            )
        yaml_description = st.text_input(
            "description",
            value=str(special_cfg.get("description") or "特殊时间分段与背景窗"),
            key=f"{prefix}_special_editor_description",
        )
        segments_text = st.text_area(
            "time_segments（每行：name start stop，或 start stop 自动命名）",
            value=existing_text,
            height=160,
            key=f"{prefix}_special_editor_segments_text",
        )
        if st.button("保存特殊时间分 bin", key=f"{prefix}_special_editor_save"):
            try:
                segments = parse_time_segments_text(segments_text)
                saved = upsert_special_burst_config(
                    yaml_path,
                    name=yaml_name,
                    bnname=yaml_bnname,
                    active_interval=yaml_active,
                    background_interval=yaml_background,
                    time_segments=segments,
                    description=yaml_description,
                )
                st.session_state[f"{prefix}_special_yaml"] = str(Path(yaml_path).expanduser())
                st.session_state[f"{prefix}_special_burst_name"] = str(saved["name"])
                st.success(f"已写入 {Path(yaml_path).expanduser()}：{saved['name']} / {saved['bnname']}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"保存失败：{exc}")


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
    _render_special_burst_editor(
        st,
        prefix="single",
        target=target,
        grb_name=grb_name.strip() or str(defaults["grb_name"]),
        active_interval=str(active_interval),
        background_low=str(background_low),
        background_high=str(background_high),
    )
    gbm_pad_before = st.number_input("display_window_pad_before_s", value=1.0, step=0.5, key="single_gbm_pad_before_s")
    gbm_pad_after = st.number_input("display_window_pad_after_s", value=1.0, step=0.5, key="single_gbm_pad_after_s")

    col1, col2 = st.columns(2)
    with col1:
        lat_three_ml_full = st.checkbox("lat_three_ml_full", value=False, key="single_lat_three_ml_full")
        plot_joint_lightcurve = st.checkbox("plot_joint_lightcurve", value=True, key="single_plot_joint_lightcurve")
    with col2:
        selected_models = st.multiselect(
            "models",
            options=SUPPORTED_MODELS,
            default=["band"],
            key="single_models",
        )
        parallel_models = st.checkbox(
            "parallel_models",
            value=True,
            key="single_parallel_models",
        )
        model_workers = st.number_input(
            "model_workers",
            min_value=1,
            max_value=4,
            value=2,
            step=1,
            key="single_model_workers",
            disabled=not parallel_models,
        )
        special_yaml = st.text_input("special_yaml", value="", key="single_special_yaml")
        special_burst_name = st.text_input("special_burst_name", value="", key="single_special_burst_name")

    with st.expander("绘图版式", expanded=False):
        canvas_left, canvas_mid, canvas_right = st.columns(3)
        with canvas_left:
            diagnostic_width = st.number_input(
                "Counts/SED 宽度 (in)", min_value=2.5, max_value=12.0,
                value=5.0, step=0.1, key="single_diagnostic_width"
            )
            diagnostic_height = st.number_input(
                "Counts/SED 高度 (in)", min_value=2.5, max_value=12.0,
                value=3.8, step=0.1, key="single_diagnostic_height"
            )
        with canvas_mid:
            spectrum_width = st.number_input(
                "分离谱宽度 (in)", min_value=2.5, max_value=12.0,
                value=12.0, step=0.1, key="single_spectrum_width_reference_v2"
            )
            spectrum_height = st.number_input(
                "分离谱高度 (in)", min_value=2.5, max_value=12.0,
                value=8.0, step=0.1, key="single_spectrum_height_reference_v2"
            )
            spectrum_font_mode_label = st.selectbox(
                "分离谱字号模式",
                options=("参考原图字号", "按论文最终字号缩放"),
                index=0,
                key="single_spectrum_font_mode_v2",
            )
        with canvas_right:
            corner_width = st.number_input(
                "Corner 宽度 (in)", min_value=3.0, max_value=14.0,
                value=5.0, step=0.1, key="single_corner_width"
            )
            corner_height = st.number_input(
                "Corner 高度 (in)", min_value=3.0, max_value=14.0,
                value=5.0, step=0.1, key="single_corner_height"
            )

        font_left, font_mid, font_right = st.columns(3)
        with font_left:
            axis_label_pt = st.number_input(
                "最终轴标题字号 (pt)", min_value=5.0, max_value=16.0,
                value=8.5, step=0.5, key="single_axis_label_pt"
            )
        with font_mid:
            tick_label_pt = st.number_input(
                "最终刻度字号 (pt)", min_value=5.0, max_value=16.0,
                value=8.0, step=0.5, key="single_tick_label_pt"
            )
        with font_right:
            legend_pt = st.number_input(
                "最终图例字号 (pt)", min_value=5.0, max_value=16.0,
                value=8.0, step=0.5, key="single_legend_pt"
            )

        legend_counts, legend_spectrum, legend_counts_cols, legend_spectrum_cols, legend_bins = st.columns(5)
        legend_position_labels = {
            "顶部图例带": "top",
            "右上": "upper right",
            "左上": "upper left",
            "右下": "lower right",
            "左下": "lower left",
        }
        with legend_counts:
            diagnostic_legend_position_label = st.selectbox(
                "Counts/SED 图例位置",
                options=list(legend_position_labels),
                index=0,
                key="single_diagnostic_legend_position_v2",
            )
        with legend_spectrum:
            spectrum_legend_position_label = st.selectbox(
                "分离谱图例位置",
                options=list(legend_position_labels),
                index=2,
                key="single_spectrum_legend_position_v2",
            )
            spectrum_legend_frame = st.checkbox(
                "分离谱图例边框",
                value=True,
                key="single_spectrum_legend_frame_v2",
            )
        with legend_counts_cols:
            diagnostic_legend_columns = st.number_input(
                "Counts 图例列数", min_value=1, max_value=6, value=4, step=1,
                key="single_diagnostic_legend_columns"
            )
        with legend_spectrum_cols:
            spectrum_legend_columns = st.number_input(
                "分离谱图例列数", min_value=1, max_value=6, value=2, step=1,
                key="single_spectrum_legend_columns_reference_v2"
            )
        with legend_bins:
            lat_plot_bins = st.number_input(
                "LAT 展示 bin 数", min_value=2, max_value=20, value=5, step=1,
                key="single_lat_plot_bins"
            )
            show_lat_upper_limits = st.checkbox(
                "LAT 95% 上限", value=True,
                key="single_show_lat_upper_limits",
                help="绘制 LAT 95% upper-limit 箭头"
            )

    plot_style = {
        "diagnostic_width_in": float(diagnostic_width),
        "diagnostic_height_in": float(diagnostic_height),
        "spectrum_width_in": float(spectrum_width),
        "spectrum_height_in": float(spectrum_height),
        "corner_width_in": float(corner_width),
        "corner_height_in": float(corner_height),
        "axis_label_pt": float(axis_label_pt),
        "tick_label_pt": float(tick_label_pt),
        "legend_pt": float(legend_pt),
        "diagnostic_legend_position": legend_position_labels[diagnostic_legend_position_label],
        "spectrum_legend_position": legend_position_labels[spectrum_legend_position_label],
        "spectrum_legend_frame": bool(spectrum_legend_frame),
        "spectrum_font_mode": (
            "reference" if spectrum_font_mode_label == "参考原图字号" else "manuscript"
        ),
        "diagnostic_legend_columns": int(diagnostic_legend_columns),
        "spectrum_legend_columns": int(spectrum_legend_columns),
        "lat_plot_bins": int(lat_plot_bins),
        "show_lat_upper_limits": bool(show_lat_upper_limits),
    }

    if st.button("开始运行", type="primary", key="single_run_button"):
        try:
            models = _validate_selected_models(selected_models)
        except ValueError as exc:
            st.error(str(exc))
            return
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
            models=models,
            parallel_models=bool(parallel_models),
            model_workers=int(model_workers),
            plot_style=plot_style,
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
        project.config.parallel_models = bool(run_overrides.parallel_models)
        project.config.model_workers = int(run_overrides.model_workers or 1)
        project.config.plot_style = run_overrides.plot_style
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

    _render_special_burst_editor(
        st,
        prefix="lc",
        target=target,
        grb_name=grb_name.strip() or str(defaults["grb_name"]),
        active_interval=str(active_interval),
        background_low=str(background_low),
        background_high=str(background_high),
    )

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
        lat_dir.mkdir(parents=True, exist_ok=True)
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
                        "lat_prob_bn_dir": str(lat_dir),
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
                _show_lightcurve_plot(
                    st,
                    Path(lc_out),
                    caption=f"{grb_name.strip() or str(defaults['grb_name'])} 光变曲线",
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"光变曲线生成失败：{exc}")


def _run_gbm_download_page(*, st, base_cfg: GRBProjectConfig) -> None:
    st.markdown("#### GBM 数据下载")
    st.caption("从 HEASARC Fermi GBM 归档下载 burst 数据，保存目录会作为后续分析的 GBM 数据根目录使用。")

    mode = st.radio(
        "下载模式",
        options=["单个 GRB", "GRB 列表", "联合筛选"],
        horizontal=True,
        key="download_mode",
    )
    save_dir = st.text_input(
        "保存目录",
        value=str(getattr(base_cfg, "gbm_download_dir", base_cfg.data_dir)),
        key="download_save_dir",
    )

    bnname = ""
    list_text = ""
    csv_path = str(getattr(base_cfg, "lat_filtered_csv", "/home/mxr/lee/data/GRB_FermiLAT_filtered.csv"))
    gbm_xls = base_cfg.catalog_xls
    year_from = int(getattr(base_cfg, "gbm_download_year_from", 2022))
    year_to_value = 0

    if mode == "单个 GRB":
        bnname = st.text_input("bnname", value="bn231129799", key="download_bnname")
    elif mode == "GRB 列表":
        list_text = st.text_area(
            "GRB 列表",
            value="bn231129799\n",
            help="每行一个 bnname，空行和 # 开头的行会被忽略。",
            key="download_list_text",
        )
    else:
        csv_path = st.text_input("LAT 过滤 CSV", value=csv_path, key="download_csv_path")
        gbm_xls = st.text_input("GBM catalog Excel", value=gbm_xls, key="download_gbm_xls")
        cols = st.columns(2)
        with cols[0]:
            year_from = st.number_input("最早年份", min_value=2008, max_value=2100, value=year_from, step=1, key="download_year_from")
        with cols[1]:
            year_to_value = st.number_input("最晚年份（0 表示不限制）", min_value=0, max_value=2100, value=0, step=1, key="download_year_to")

    if st.button("开始下载", type="primary", key="download_run_button"):
        year_to = int(year_to_value) if int(year_to_value) > 0 else None
        with st.spinner("正在下载 GBM 数据..."):
            try:
                targets = _run_gbm_download_request(
                    mode=mode,
                    bnname=bnname,
                    list_text=list_text,
                    save_dir=save_dir,
                    csv_path=csv_path,
                    gbm_xls=gbm_xls,
                    year_from=int(year_from),
                    year_to=year_to,
                )
                st.success(f"下载任务完成：{len(targets)} 个 GRB，保存到 {Path(save_dir).expanduser()}")
                with st.expander("目标列表", expanded=False):
                    st.write(targets)
            except Exception as exc:  # noqa: BLE001
                st.error(f"下载失败：{exc}")

    st.markdown("---")
    st.markdown("#### GCN LAT 数据更新")
    st.caption("每次都从 GCN 官方归档重新下载 archive.json.tar.gz，成功后覆盖本地 archive 和 LAT CSV。")
    archive_url = st.text_input("GCN archive URL", value=base_cfg.gcn_archive_url, key="gcn_archive_url")
    archive_tar = st.text_input("archive 压缩包", value=base_cfg.gcn_archive_tar, key="gcn_archive_tar")
    archive_dir = st.text_input("archive 解压目录", value=base_cfg.gcn_archive_dir, key="gcn_archive_dir")
    output_csv = st.text_input("LAT 输出 CSV", value=base_cfg.lat_gcn_output_csv, key="gcn_lat_output_csv")
    if st.button("更新 GCN LAT 数据", type="primary", key="gcn_lat_refresh_button"):
        with st.spinner("正在下载最新 GCN archive 并提取 LAT 数据..."):
            try:
                result = _run_lat_gcn_refresh_request(
                    archive_url=archive_url,
                    archive_tar=archive_tar,
                    archive_dir=archive_dir,
                    output_csv=output_csv,
                )
                st.success(f"GCN LAT 数据更新完成：{result.record_count} 条，输出到 {result.output_csv}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"GCN LAT 数据更新失败：{exc}")

    st.markdown("---")
    st.markdown("#### LAT Extended 数据下载")
    st.caption("使用 threeML 的 Fermi-LAT 下载接口，时间范围按 GCN 观测窗前 1000 秒、后 2000 秒扩展。")
    lat_mode = st.radio(
        "LAT 下载模式",
        options=["单个 GRB", "按年月批量"],
        horizontal=True,
        key="lat_download_mode",
    )
    lat_catalog = st.text_input(
        "LAT catalog Excel", value=base_cfg.fermilat_grb_xls, key="lat_download_catalog"
    )
    lat_data_root = st.text_input(
        "LAT 保存目录", value=base_cfg.lat_download_root, key="lat_download_root"
    )
    lat_bnnames = None
    lat_year = None
    lat_month = 1
    if lat_mode == "单个 GRB":
        lat_bnname = st.text_input("LAT bnname", value="bn250313607", key="lat_download_bnname")
        lat_bnnames = [lat_bnname]
    else:
        lat_cols = st.columns(2)
        with lat_cols[0]:
            lat_year = int(
                st.number_input(
                    "LAT 年份", min_value=2008, max_value=2100, value=2025, step=1, key="lat_download_year"
                )
            )
        with lat_cols[1]:
            lat_month = int(
                st.number_input(
                    "起始月份", min_value=1, max_value=12, value=1, step=1, key="lat_download_month"
                )
            )
    if st.button("开始下载 LAT 数据", type="primary", key="lat_download_run_button"):
        with st.spinner("正在请求并下载 LAT Extended 数据..."):
            try:
                results = _run_lat_download_request(
                    bnnames=lat_bnnames,
                    year=lat_year,
                    month_from=lat_month,
                    catalog_xls=lat_catalog,
                    data_root=lat_data_root,
                )
                st.success(f"LAT 下载完成：{len(results)} 个 GRB，保存到 {Path(lat_data_root).expanduser()}")
                with st.expander("下载结果", expanded=False):
                    st.write([result.target.bnname for result in results])
            except Exception as exc:  # noqa: BLE001
                st.error(f"LAT 下载失败：{exc}")


def _run_special_burst_page(
    *,
    st,
    df_catalog: pd.DataFrame,
    fermilat_catalog: pd.DataFrame,
    target_options: Sequence[str],
    default_target: str,
) -> None:
    st.markdown("#### 特殊暴分 bin")
    st.caption("把某个暴的特殊时间分段和背景窗写入项目内 special_bursts.yaml。")

    target = st.selectbox(
        "bnname / 目标 GRB",
        options=target_options,
        index=target_options.index(default_target),
        key="special_page_target",
    )
    catalog_row = df_catalog.loc[target]
    if isinstance(catalog_row, pd.DataFrame):
        catalog_row = catalog_row.iloc[0]
    lat_row = fermilat_catalog.loc[target] if target in fermilat_catalog.index else None
    if isinstance(lat_row, pd.DataFrame):
        lat_row = lat_row.iloc[0]
    defaults = _sync_target_defaults(
        st,
        prefix="special_page",
        target=target,
        catalog_row=catalog_row,
        lat_row=lat_row,
    )

    _show_target_context(st, defaults=defaults)

    cols = st.columns(2)
    with cols[0]:
        grb_name = st.text_input(
            "name / GRBname",
            value=str(defaults["grb_name"]),
            key="special_page_grb_name",
        )
        active_interval = st.text_input(
            "active_interval",
            value=str(defaults["active_interval"]),
            key="special_page_active_interval",
        )
    with cols[1]:
        background_low = st.text_input(
            "background_low",
            value=str(defaults["background_low"]),
            key="special_page_background_low",
        )
        background_high = st.text_input(
            "background_high",
            value=str(defaults["background_high"]),
            key="special_page_background_high",
        )
    _render_special_burst_editor(
        st,
        prefix="special_page",
        target=target,
        grb_name=grb_name.strip(),
        active_interval=active_interval.strip(),
        background_low=background_low.strip(),
        background_high=background_high.strip(),
        expanded=True,
    )


def _joint_batch_run_state_path() -> Path:
    return Path(__file__).resolve().parent.parent / "joint_batch_run.json"


def _read_joint_batch_run_state() -> Optional[dict]:
    path = _joint_batch_run_state_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _joint_batch_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _build_joint_batch_command(
    *,
    analysis_mode: str,
    result_root: str,
    summary_csv_name: str,
    models: Sequence[str],
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    min_lat_ts: Optional[float] = None,
    include_lle_only: bool = False,
    only: Optional[Sequence[str]] = None,
) -> list:
    """构造后台批量运行的 select --run 命令（纯函数，便于测试）。"""
    cmd = [
        sys.executable,
        "-m",
        "grb_project",
        "select",
        "--analysis-mode",
        str(analysis_mode),
        "--run",
        "--result-root",
        str(Path(result_root).expanduser()),
        "--summary-csv-name",
        str(summary_csv_name),
        "--session-log",
        "--models",
        *[str(m) for m in models],
    ]
    if year_from is not None:
        cmd += ["--year-from", str(int(year_from))]
    if year_to is not None:
        cmd += ["--year-to", str(int(year_to))]
    if min_lat_ts is not None:
        cmd += ["--min-lat-ts", f"{float(min_lat_ts):g}"]
    if include_lle_only:
        cmd.append("--include-lle-only")
    if only:
        cmd += ["--only", *[str(b) for b in only]]
    return cmd


def _launch_joint_batch_run(
    *, target_count: int, log_tail_hint: str = "", **command_kwargs
) -> dict:
    """以独立后台进程启动批量联合分析，日志写结果目录，状态写 gbmtest 根。"""
    cmd = _build_joint_batch_command(**command_kwargs)
    cwd = Path(__file__).resolve().parent.parent
    result_root = Path(command_kwargs["result_root"]).expanduser()
    result_root.mkdir(parents=True, exist_ok=True)
    log_path = result_root / "joint_batch_run.log"
    with log_path.open("ab") as log_fh:
        log_fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} 启动 =====\n".encode("utf-8"))
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    state = {
        "pid": proc.pid,
        "cmd": cmd,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "log": str(log_path),
        "result_root": str(result_root),
        "targets": int(target_count),
    }
    _joint_batch_run_state_path().write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return state


def _run_joint_selection_page(*, st, base_cfg: GRBProjectConfig) -> None:
    from grb_project.joint_selection import JointSelectionCriteria, select_joint_targets

    st.subheader("联合目标选择")
    st.caption(
        "以 lat_download_targets.csv 交叉表（GBM 目录 × 2FLGC × LLE × GCN）加磁盘实况为准，"
        "一键筛出可做 GBM+LAT 联合分析的暴；排除原因逐项可查。"
    )
    col_from, col_to, col_ts, col_lle = st.columns(4)
    year_from = int(col_from.number_input("起始年份（含）", min_value=2008, max_value=2100, value=2008, step=1))
    year_to = int(col_to.number_input("截止年份（含）", min_value=2008, max_value=2100, value=2026, step=1))
    min_ts_raw = float(
        col_ts.number_input("最小 LAT TS（0 = 不限）", min_value=0.0, max_value=100000.0, value=0.0, step=1.0)
    )
    include_lle = bool(col_lle.checkbox("包含 LLE-only 暴"))

    criteria = JointSelectionCriteria(
        year_from=year_from,
        year_to=year_to,
        min_lat_ts=(min_ts_raw if min_ts_raw > 0 else None),
        include_lle_only=include_lle,
        analysis_mode="gbm+lat",
    )
    try:
        result = select_joint_targets(criteria)
    except Exception as exc:  # noqa: BLE001
        st.error(f"选择失败：{exc}")
        return

    metric_targets, metric_excluded, metric_total = st.columns(3)
    metric_targets.metric("可选目标", len(result.targets))
    metric_excluded.metric("被排除", result.excluded_total())
    metric_total.metric("表内总行数", len(result.targets) + result.excluded_total())

    df = result.to_frame()
    if not df.empty:
        st.bar_chart(df["year"].value_counts().sort_index())
    st.dataframe(df, use_container_width=True)

    with st.expander("排除原因明细", expanded=False):
        if result.excluded:
            for reason in sorted(result.excluded):
                names = result.excluded[reason]
                preview = "、".join(names[:8]) + ("…" if len(names) > 8 else "")
                st.write(f"- **{reason}**：{len(names)} 个（{preview}）")
        else:
            st.write("无排除记录。")

    st.download_button(
        "导出选中目标 CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"joint_targets_{year_from}_{year_to}.csv",
        mime="text/csv",
        disabled=df.empty,
    )
    st.markdown("---")
    state = _read_joint_batch_run_state()
    running = bool(state and _joint_batch_pid_alive(int(state["pid"])))
    if running:
        st.warning(
            f"已有批量任务在运行（PID {state['pid']}，{state.get('targets', '?')} 个目标，"
            f"启动于 {state.get('started', '?')}）；结束后才能再次启动。"
        )

    with st.expander("批量运行设置", expanded=not running):
        run_result_root = str(
            st.text_input(
                "结果目录",
                value="/home/mxr/lee/gbmtest/results_joint_batch",
                key="joint_run_result_root",
            )
        ).strip()
        run_summary_csv = str(
            st.text_input(
                "汇总 CSV 文件名", value="summary_results.csv", key="joint_run_summary_csv"
            )
        ).strip() or "summary_results.csv"
        run_mode = str(
            st.selectbox("分析模式", ("gbm+lat", "lat"), index=0, key="joint_run_mode")
        )
        run_models = st.multiselect(
            "模型（至少选一个）", list(SUPPORTED_MODELS), default=["band"], key="joint_run_models"
        )
        scope = st.radio("目标范围", ("全部选中目标", "手动挑选子集"), key="joint_run_scope")
        chosen = None
        if scope == "手动挑选子集":
            chosen = st.multiselect("要运行的暴（bn 名）", result.bnnames, key="joint_run_targets")
        st.caption(
            "点击启动后会在服务器后台以独立进程运行批量联合分析"
            "（python -m grb_project select --run），页面可以关闭；"
            "日志写入结果目录下 joint_batch_run.log。"
        )
        launched = st.button("启动批量联合分析", disabled=running, key="joint_run_launch")

    if launched:
        try:
            models_validated = _validate_selected_models(run_models)
            if chosen is not None and not chosen:
                raise ValueError("手动挑选子集时至少选择一个暴")
            if not run_result_root:
                raise ValueError("结果目录不能为空")
            target_count = len(chosen) if chosen is not None else len(result.targets)
            info = _launch_joint_batch_run(
                analysis_mode=run_mode,
                result_root=run_result_root,
                summary_csv_name=run_summary_csv,
                models=models_validated,
                year_from=year_from,
                year_to=year_to,
                min_lat_ts=criteria.min_lat_ts,
                include_lle_only=include_lle,
                only=chosen,
                target_count=target_count,
            )
            st.success(
                f"已在后台启动批量分析：PID {info['pid']}，{info['targets']} 个目标，"
                f"日志 {info['log']}"
            )
            state = info
        except Exception as exc:  # noqa: BLE001
            st.error(f"启动失败：{exc}")

    if state:
        alive = _joint_batch_pid_alive(int(state["pid"]))
        st.markdown("---")
        st.subheader("最近一次批量运行")
        st.write(
            f"状态：{'运行中' if alive else '已结束'}　PID：{state['pid']}　"
            f"目标：{state.get('targets', '?')} 个　启动于：{state.get('started', '?')}"
        )
        st.code(" ".join(state.get("cmd", [])), language="bash")
        log_path = Path(state["log"]) if state.get("log") else None
        if log_path is not None and log_path.exists():
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            with st.expander(f"日志尾部（{log_path}）", expanded=alive):
                st.code("\n".join(lines[-15:]) or "（暂无输出）")


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="GRB 工作台", layout="wide")
    st.title("GRB 工作台")
    st.caption("单页内切换单次分析与光变曲线页面，前后端共用同一套参数与结果目录约定。")

    base_cfg = GRBProjectConfig()
    st.sidebar.header("页面切换")
    page = st.sidebar.radio(
        "页面",
        options=["单次分析", "光变曲线", "特殊暴分 bin", "GBM 数据下载", "联合目标选择"],
        index=0,
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("分析、光变曲线和下载页面共用项目内后端流程。")

    if page == "GBM 数据下载":
        _run_gbm_download_page(st=st, base_cfg=base_cfg)
        return
    if page == "联合目标选择":
        _run_joint_selection_page(st=st, base_cfg=base_cfg)
        return
    try:
        df_catalog = _load_catalog(base_cfg)
        fermilat_catalog = _load_fermilat_catalog(base_cfg)
        year_mode = st.selectbox(
            "联合分析样本年份",
            options=("所有年份", "指定起始年份"),
            index=0,
            key="joint_catalog_year_mode",
        )
        year_from = None
        if year_mode == "指定起始年份":
            year_from = int(
                st.number_input(
                    "起始年份",
                    min_value=2008,
                    max_value=2100,
                    value=2022,
                    step=1,
                    key="joint_catalog_year_from",
                )
            )
        df_joint = _filter_joint_catalog_rows(df_catalog, fermilat_catalog, year_from=year_from)
        if df_joint.empty:
            st.error("当前年份范围内没有同时存在于 GBM 和 LAT catalog 的暴")
            return
        default_target = _default_target(df_joint)
    except Exception as exc:  # noqa: BLE001
        st.error(f"无法加载目录表：{exc}")
        return

    target_options = df_joint.index.astype(str).tolist()

    st.subheader("目录表概况")
    with st.expander("查看可联合分析样本", expanded=False):
        left, right = st.columns([1.2, 1])
        with left:
            st.caption(f"目录表总行数：{len(df_catalog)}")
            range_label = "所有年份" if year_from is None else f"{year_from} 年起"
            st.caption(f"{range_label} GBM/LAT 共同样本数：{len(df_joint)}")
        with right:
            st.caption(f"默认目标：{default_target}")
        st.dataframe(_make_catalog_summary(df_joint), use_container_width=True)

    if page == "特殊暴分 bin":
        _run_special_burst_page(
            st=st,
            df_catalog=df_catalog,
            fermilat_catalog=fermilat_catalog,
            target_options=target_options,
            default_target=default_target,
        )
    elif page == "单次分析":
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
