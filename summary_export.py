# -*- coding: utf-8 -*-
"""汇总表后处理：时间 bin 辅助列、时间 bin 分析、TeX/JSON/YAML 导出。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .logging_utils import log
from .session import session


def _append_time_bin_info(df_summary: pd.DataFrame) -> None:
    if df_summary.empty or "bin_start_time" not in df_summary.columns:
        return

    df_summary["time_bin_identifier"] = df_summary.apply(
        lambda row: f"{row['bnname']}_bin_{row['bin_start_time']:.2f}_"
        f"{row['bin_end_time']:.2f}",
        axis=1,
    )
    df_summary["bin_relative_position"] = (
        (df_summary["bin_start_time"] + df_summary["bin_end_time"])
        / 2.0
        / df_summary["duration"]
    )
    df_summary["bin_duration_ratio"] = (
        df_summary["bin_duration"] / df_summary["duration"]
    )


def _compute_time_bin_analysis(df_summary: pd.DataFrame) -> List[Dict]:
    time_analysis: List[Dict] = []

    if df_summary.empty:
        return time_analysis

    if "bin_start_time" in df_summary.columns:
        time_bin_analysis = (
            df_summary.groupby(["bnname", "bin_start_time", "bin_end_time"])
            .agg(
                {
                    "AIC": ["mean", "std"],
                    "BIC": ["mean", "std"],
                    "log_marginal_likelihood": ["mean", "std"],
                    "Flux(erg/cm2/s)": ["mean", "std"],
                    "model": lambda x: ", ".join(x.unique()),
                }
            )
            .round(4)
        )
        time_bin_analysis.columns = [
            "_".join(col).strip() for col in time_bin_analysis.columns
        ]
        time_bin_analysis.reset_index(inplace=True)
        time_analysis.extend(time_bin_analysis.to_dict("records"))

    return time_analysis


def _save_time_bin_analysis(time_analysis: List[Dict]) -> None:
    if time_analysis:
        time_analysis_df = pd.DataFrame(time_analysis)
        time_analysis_csv = os.path.join(
            session.result_root,
            "time_bin_analysis_results.csv",
        )
        time_analysis_df.to_csv(time_analysis_csv, index=False)
        log(f"时间段分析结果已保存至 {time_analysis_csv}")


def _time_bin_csv_tag(bin_start: object, bin_end: object) -> str:
    """与 separate_spectr 中谱图/LAT 文件后缀一致的时间段标签，如 0.1-1。"""
    def _fmt_time(t: object) -> str:
        try:
            return f"{float(t):.2f}".rstrip("0").rstrip(".")
        except (TypeError, ValueError):
            return str(t)

    if isinstance(bin_start, (int, float)) and isinstance(bin_end, (int, float)):
        return f"{_fmt_time(bin_start)}-{_fmt_time(bin_end)}"
    return ""


def _iter_time_bin_groups(
    df_grb: pd.DataFrame,
) -> List[Tuple[str, float, float, pd.DataFrame]]:
    """按时间段拆分 DataFrame，返回 (标签, t_start, t_end, 子表) 列表。"""
    if df_grb.empty:
        return []

    has_bin_times = (
        "bin_start_time" in df_grb.columns and "bin_end_time" in df_grb.columns
    )
    if not has_bin_times:
        return []

    groups: List[Tuple[str, float, float, pd.DataFrame]] = []
    grouped = df_grb.groupby(["bin_start_time", "bin_end_time"], sort=True)
    for (t_start, t_end), df_bin in grouped:
        tag = _time_bin_csv_tag(t_start, t_end)
        if not tag:
            continue
        groups.append((tag, float(t_start), float(t_end), df_bin))
    return groups


def _save_allmodel_csv_per_time_bins(
    df_grb: pd.DataFrame,
    grb_dir: str,
    base_name: str,
) -> List[str]:
    """为每个时间段写出 {GRB}_allmodel_{t0}-{t1}.csv。"""
    written: List[str] = []
    for tag, _t0, _t1, df_bin in _iter_time_bin_groups(df_grb):
        bin_csv = os.path.join(grb_dir, f"{base_name}_allmodel_{tag}.csv")
        df_bin.to_csv(bin_csv, index=False)
        written.append(bin_csv)
    if written:
        log(
            f"GRB {base_name} 已按时间段写出 {len(written)} 个 allmodel CSV "
            f"（{written[0]} 等）"
        )
    return written


def _save_all_models_per_grb(df_summary: pd.DataFrame) -> None:
    if "grb_name" not in df_summary.columns:
        return

    for grb_name_unique in df_summary["grb_name"].unique():
        actual_grb_name = None
        for item in os.listdir(session.result_root):
            item_path = os.path.join(session.result_root, item)
            if not os.path.isdir(item_path):
                continue
            if grb_name_unique in os.listdir(item_path):
                actual_grb_name = item
                break

        if actual_grb_name:
            df_grb = df_summary[df_summary["grb_name"] == grb_name_unique]
            grb_dir = os.path.join(session.result_root, str(actual_grb_name))
        else:
            df_grb = df_summary[df_summary["grb_name"] == grb_name_unique]
            grb_dir = os.path.join(session.result_root, str(grb_name_unique))

        base_name = os.path.basename(grb_dir)
        safescv = os.path.join(grb_dir, f"{base_name}_allmodel.csv")
        df_grb.to_csv(safescv, mode="w", index=False)
        log(f"GRB {base_name} 的所有模型结果已保存至 {safescv}")

        _save_allmodel_csv_per_time_bins(df_grb=df_grb, grb_dir=grb_dir, base_name=base_name)

        _export_grb_additional_formats(
            df_grb=df_grb,
            csv_path=safescv,
            grb_dir=grb_dir,
        )


def _export_grb_additional_formats(
    df_grb: pd.DataFrame,
    csv_path: str,
    grb_dir: str,
) -> None:
    grb_path = Path(grb_dir)
    base_name = grb_path.name

    flat_rows = df_grb.to_dict(orient="records")
    json_flat_path = grb_path / f"{base_name}_allmodel.json"
    yaml_flat_path = grb_path / f"{base_name}_allmodel.yaml"

    with json_flat_path.open("w", encoding="utf-8") as f:
        json.dump(flat_rows, f, ensure_ascii=False, indent=2)
    yaml_flat_path.write_text(
        json_flat_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    nested: Dict[str, Dict[str, Dict[str, object]]] = {}
    for row in flat_rows:
        grb_name = row.get("grb_name")
        bnname = row.get("bnname")
        model = row.get("model")
        if not grb_name or not bnname or not model:
            continue

        time_bin_identifier = row.get("time_bin_identifier")
        if not time_bin_identifier:
            t0 = row.get("bin_start_time")
            t1 = row.get("bin_end_time")
            if isinstance(t0, (int, float)) and isinstance(t1, (int, float)):
                time_bin_identifier = f"{bnname}_bin_{t0:.2f}_{t1:.2f}"
            else:
                time_bin_identifier = f"{bnname}_bin_unknown"

        top_key = f"{grb_name}__{bnname}__{time_bin_identifier}"
        if top_key not in nested:
            nested[top_key] = {}
        if model not in nested[top_key]:
            nested[top_key][model] = {}

        payload = nested[top_key][model]
        for k, v in row.items():
            if k in {"grb_name", "bnname", "model"}:
                continue
            if k.startswith("cons"):
                continue
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            payload[k] = v

    json_nested_path = grb_path / f"{base_name}_allmodel_nested.json"
    yaml_nested_path = grb_path / f"{base_name}_allmodel_nested.yaml"
    with json_nested_path.open("w", encoding="utf-8") as f:
        json.dump(nested, f, ensure_ascii=False, indent=2)
    yaml_nested_path.write_text(
        json_nested_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    COMMON_KEYS = {
        "analysis_mode",
        "num_time_bins",
        "bin_start_time",
        "bin_end_time",
        "bin_duration",
        "bin_relative_position",
        "bin_duration_ratio",
    }

    compact: Dict[str, Dict[str, object]] = {}
    for top_key, models in nested.items():
        parts = top_key.split("__")
        if len(parts) >= 3:
            grb_name_key, bnname_key, time_bin_identifier = (
                parts[0],
                parts[1],
                "__".join(parts[2:]),
            )
        elif len(parts) == 2:
            grb_name_key, bnname_key = parts
            time_bin_identifier = f"{bnname_key}_bin_unknown"
        else:
            grb_name_key, bnname_key = top_key, ""
            time_bin_identifier = "bin_unknown"

        meta: Dict[str, object] = {"bnname": bnname_key}
        if isinstance(models, dict) and models:
            _first_model_name, first_payload = next(iter(models.items()))
            if isinstance(first_payload, dict):
                for k in COMMON_KEYS:
                    if k in first_payload:
                        meta[k] = first_payload[k]

        cleaned_models: Dict[str, Dict[str, object]] = {}
        for model_name, payload in models.items():
            if not isinstance(payload, dict):
                cleaned_models[model_name] = payload  # type: ignore[assignment]
                continue
            cleaned = {k: v for k, v in payload.items()}
            for ck in COMMON_KEYS:
                cleaned.pop(ck, None)
            cleaned.pop("time_bin_identifier", None)
            cleaned_models[model_name] = cleaned

        if grb_name_key not in compact:
            compact[grb_name_key] = {"bins": {}}

        entry = compact[grb_name_key]
        bins = entry.get("bins", {})
        if not isinstance(bins, dict):
            bins = {}
            entry["bins"] = bins

        bin_entry = bins.get(time_bin_identifier, {})
        if not isinstance(bin_entry, dict):
            bin_entry = {}
        if meta:
            bin_entry["meta"] = meta
        for model_name, cleaned_payload in cleaned_models.items():
            bin_entry[model_name] = cleaned_payload
        bins[time_bin_identifier] = bin_entry

    json_nested_compact = grb_path / f"{base_name}_allmodel_nested_compact.json"
    yaml_nested_compact = grb_path / f"{base_name}_allmodel_nested_compact.yaml"
    with json_nested_compact.open("w", encoding="utf-8") as f:
        json.dump(compact, f, ensure_ascii=False, indent=2)
    yaml_nested_compact.write_text(
        json_nested_compact.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    tex_path = grb_path / f"{base_name}_allmodel_table.tex"

    lines: List[str] = []

    def _fmt(v: object, sci: bool = False) -> str:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if sci:
                return f"{v:.3e}"
            else:
                return f"{v:.2f}"
        if v == "":
            return ""
        return str(v)

    for grb_name_key, entry in compact.items():
        if not isinstance(entry, dict):
            continue
        bins = entry.get("bins", {})
        if not isinstance(bins, dict) or not bins:
            continue

        sample_meta: Dict[str, object] = {}
        for _tb, _be in bins.items():
            if isinstance(_be, dict) and isinstance(_be.get("meta", {}), dict):
                sample_meta = _be["meta"]  # type: ignore[assignment]
                break

        analysis_mode = str(sample_meta.get("analysis_mode", "gbm"))
        if "lat" in analysis_mode.lower():
            band_tex = r"$8$--$10^{8}$~keV"
        else:
            band_tex = r"$8$--$4\times10^{4}$~keV"
        bnname_key = str(sample_meta.get("bnname", ""))

        has_pl_models = False
        for _tb, _be in bins.items():
            if not isinstance(_be, dict):
                continue
            for _mn in _be:
                if _mn != "meta" and "pl" in str(_mn).lower():
                    has_pl_models = True
                    break
            if has_pl_models:
                break

        if has_pl_models:
            col_spec = r"lccccccccc"
            n_columns = 10
        else:
            col_spec = r"lcccccccc"
            n_columns = 9

        lines.append(r"\startlongtable")
        lines.append(rf"\begin{{deluxetable*}}{{{col_spec}}}")
        lines.append(
            "\\tablecaption{Spectral fits for %s (%s)\\label{tab:%s_allbins_models}}"
            % (grb_name_key, bnname_key, bnname_key)
        )
        lines.append(rf"\tablecolumns{{{n_columns}}}")
        lines.append(r"\tablewidth{0pt}")
        lines.append(r"\tablehead{")
        lines.append(r"\colhead{Time bin (s)} &")
        lines.append(r"\colhead{Model} &")
        lines.append(r"\colhead{$\\alpha$} &")
        lines.append(r"\colhead{$\\beta$} &")
        if has_pl_models:
            lines.append(r"\colhead{$index$} &")
        lines.append(r"\colhead{$E_{p}/E_{c}$} &")
        lines.append(r"\colhead{$kT$} &")
        lines.append(r"\colhead{$F_{BB}\\times10^{6}$} &")
        lines.append(r"\colhead{$F_{Tot}\\times10^{6}$} &")
        lines.append(r"\colhead{BIC} \\")
        lines.append(r"\colhead{} &")
        lines.append(r"\colhead{} &")
        lines.append(r"\colhead{} &")
        lines.append(r"\colhead{} &")
        if has_pl_models:
            lines.append(r"\colhead{} &")
        lines.append(r"\colhead{} &")
        lines.append(r"\colhead{(keV)} &")
        lines.append(r"\colhead{(erg cm$^{-2}$ s$^{-1}$)} &")
        lines.append(r"\colhead{(erg cm$^{-2}$ s$^{-1}$)} &")
        lines.append(r"\colhead{}")
        lines.append(r"}")
        lines.append(r"\startdata")

        def _bin_sort_key(item: Tuple[str, Dict[str, object]]) -> float:
            ident, be = item
            if not isinstance(be, dict):
                return 0.0
            meta = be.get("meta", {})
            if isinstance(meta, dict):
                v = meta.get("bin_start_time")
                if isinstance(v, (int, float)):
                    return float(v)
            return 0.0

        sorted_bins = sorted(bins.items(), key=_bin_sort_key)

        prev_bin_label: Optional[str] = None

        for time_bin_identifier, bin_entry in sorted_bins:
            if not isinstance(bin_entry, dict):
                continue
            meta = (
                bin_entry.get("meta", {})
                if isinstance(bin_entry.get("meta", {}), dict)
                else {}
            )
            t_start = meta.get("bin_start_time", "")
            t_end = meta.get("bin_end_time", "")

            if isinstance(t_start, (int, float)) and isinstance(t_end, (int, float)):
                bin_label = f"{t_start:.2f}–{t_end:.2f}"
            else:
                bin_label = str(time_bin_identifier)

            if prev_bin_label is not None and bin_label != prev_bin_label:
                lines.append(r"\hline")
            prev_bin_label = bin_label

            models = {
                k: v
                for k, v in bin_entry.items()
                if k != "meta" and isinstance(v, dict)
            }
            if not models:
                continue

            for model_name, payload in models.items():
                bic = payload.get("BIC", "")
                f_tot = payload.get("FTot(erg/cm2/s)", payload.get("Flux(erg/cm2/s)", ""))
                f_bb = payload.get("FBB(erg/cm2/s)", "")

                def _pick_first(d: Dict[str, object], keys: List[str]) -> object:
                    for kk in keys:
                        if kk in d:
                            return d[kk]
                    return ""

                def _pick_pair(
                    d: Dict[str, object],
                    candidates: List[Tuple[str, str]],
                ) -> Tuple[object, object]:
                    for v_key, e_key in candidates:
                        if v_key in d:
                            return d.get(v_key, ""), d.get(e_key, "")
                    return "", ""

                def _fmt_pm(v: object, e: object, sci: bool = False) -> str:
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        v_s = _fmt(v, sci=sci)
                        if isinstance(e, (int, float)) and not isinstance(e, bool):
                            e_s = _fmt(e, sci=sci)
                            return f"{v_s} $\\pm$ {e_s}"
                        return v_s
                    return ""

                alpha_v, alpha_e = _pick_pair(
                    payload,
                    [
                        ("GRB.spectrum.main.Band.alpha_value", "GRB.spectrum.main.Band.alpha_error"),
                        ("GRB.spectrum.main.composite.alpha_1_value", "GRB.spectrum.main.composite.alpha_1_error"),
                        ("GRB.spectrum.main.Cutoff_powerlaw.index_value", "GRB.spectrum.main.Cutoff_powerlaw.index_error"),
                        ("GRB.spectrum.main.composite.index_1_value", "GRB.spectrum.main.composite.index_1_error"),
                    ],
                )
                beta_v, beta_e = _pick_pair(
                    payload,
                    [
                        ("GRB.spectrum.main.Band.beta_value", "GRB.spectrum.main.Band.beta_error"),
                        ("GRB.spectrum.main.composite.beta_1_value", "GRB.spectrum.main.composite.beta_1_error"),
                    ],
                )
                epeak_or_ec_v, epeak_or_ec_e = _pick_pair(
                    payload,
                    [
                        ("GRB.spectrum.main.Band.xp_value", "GRB.spectrum.main.Band.xp_error"),
                        ("GRB.spectrum.main.composite.xp_1_value", "GRB.spectrum.main.composite.xp_1_error"),
                        ("GRB.spectrum.main.Cutoff_powerlaw.xc_value", "GRB.spectrum.main.Cutoff_powerlaw.xc_error"),
                        ("GRB.spectrum.main.composite.xc_1_value", "GRB.spectrum.main.composite.xc_1_error"),
                        ("GRB.spectrum.main.NonDissipativePhotosphere.ec_value", "GRB.spectrum.main.NonDissipativePhotosphere.ec_error"),
                    ],
                )
                kt_v, kt_e = _pick_pair(
                    payload,
                    [
                        ("GRB.spectrum.main.Blackbody.kT_value", "GRB.spectrum.main.Blackbody.kT_error"),
                        ("GRB.spectrum.main.composite.kT_1_value", "GRB.spectrum.main.composite.kT_1_error"),
                        ("GRB.spectrum.main.composite.kT_2_value", "GRB.spectrum.main.composite.kT_2_error"),
                    ],
                )
                pl_index_v, pl_index_e = _pick_pair(
                    payload,
                    [
                        (
                            "GRB.spectrum.main.composite.index_2_value",
                            "GRB.spectrum.main.composite.index_2_error",
                        ),
                        (
                            "GRB.spectrum.main.Powerlaw.index_value",
                            "GRB.spectrum.main.Powerlaw.index_error",
                        ),
                    ],
                )

                display_model = {
                    "comp": "CPL",
                    "blackbody": "BB",
                    "band+bb": "Band+BB",
                    "comp+bb": "CPL+BB",
                    "band+pl": "Band+PL",
                    "comp+pl": "CPL+PL",
                    "pl+bb": "PL+BB",
                }.get(model_name, model_name)

                def _fmt_f6(v: object) -> str:
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        return f"{float(v)*1e6:.3f}"
                    return ""

                line_parts = [
                    bin_label,
                    display_model,
                    _fmt_pm(alpha_v, alpha_e),
                    _fmt_pm(beta_v, beta_e),
                ]
                if has_pl_models:
                    line_parts.append(_fmt_pm(pl_index_v, pl_index_e))
                line_parts.extend(
                    [
                        _fmt_pm(epeak_or_ec_v, epeak_or_ec_e),
                        _fmt_pm(kt_v, kt_e),
                        _fmt_f6(f_bb),
                        _fmt_f6(f_tot),
                        _fmt(bic),
                    ]
                )
                lines.append(" & ".join(line_parts) + r" \\")

        lines.append(r"\enddata")
        table_comment = (
            r"Each row corresponds to one time bin and spectral model; "
            r"$F_{BB}$ is computed from the BB component (if present), and $F_{Tot}$ is the total model flux. "
        )
        if has_pl_models:
            table_comment += (
                r"$index$ is the photon index of the power-law component in composite models. "
            )
        table_comment += (
            r"The time intervals are relative to the GBM trigger; energy band is %s." % band_tex
        )
        lines.append(r"\tablecomments{" + table_comment + "}")
        lines.append(r"\end{deluxetable*}")
        lines.append("")
        break

    if lines:
        tex_path.write_text("\n".join(lines), encoding="utf-8")
        log(f"GRB {base_name} 的 JSON / YAML / TeX 表格已保存至 {grb_dir}")


def _plot_model_comparison(df_summary: pd.DataFrame) -> None:
    if df_summary.empty:
        log("警告: 没有足够的数据绘制对比图")
        return

    required_columns = [
        "bnname",
        "model",
        "AIC",
        "BIC",
    ]
    if not all(col in df_summary.columns for col in required_columns):
        log("警告: DataFrame缺少必要的列，无法绘制对比图")
        log(f"现有列: {list(df_summary.columns)}")
        return

    for metric in ["AIC", "BIC"]:
        plt.figure(figsize=(12, 6))
        all_grbs = list(df_summary["bnname"].unique())
        models = df_summary["model"].unique()
        grb_indices = {bnname: i for i, bnname in enumerate(all_grbs)}

        for model in models:
            subset = df_summary[df_summary["model"] == model]
            x_positions = [grb_indices[bn] for bn in subset["bnname"]]
            y_values = subset[metric]
            plt.scatter(x_positions, y_values, label=model)

            for x, y in zip(x_positions, y_values):
                plt.annotate(
                    model,
                    (x, y),
                    fontsize=8,
                    ha="center",
                    va="bottom",
                )

        plt.xticks(range(len(all_grbs)), all_grbs, rotation=45, ha="right")
        plt.xlabel("GRB Name")
        plt.ylabel(metric)
        plt.title(f"{metric} Comparison of Models")
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(session.result_root, f"{metric}_comparison.png")
        plt.savefig(out_path)
        plt.close()
        log(f"{metric} 对比图已保存")
