# -*- coding: utf-8 -*-
"""汇总表后处理：时间 bin 辅助列、最佳模型、TeX/JSON/YAML 导出。"""

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


def _compute_best_models_and_time_analysis(
    df_summary: pd.DataFrame,
) -> Tuple[List[pd.Series], List[Dict]]:
    best_models: List[pd.Series] = []
    time_analysis: List[Dict] = []

    if df_summary.empty:
        return best_models, time_analysis

    required_columns = [
        "bnname",
        "model",
        "AIC",
        "BIC",
        "log_marginal_likelihood",
    ]
    if not all(col in df_summary.columns for col in required_columns):
        return best_models, time_analysis

    for bnname in df_summary["bnname"].unique():
        grb_data = df_summary[df_summary["bnname"] == bnname]
        grb_data = grb_data.copy()

        best_model_idx = grb_data["AIC"].idxmin()
        best_model_row = grb_data.loc[best_model_idx]
        best_models.append(best_model_row)

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

    return best_models, time_analysis


def _extract_burst_time_info(
    bnname: str,
) -> Tuple[Optional[float], Optional[float], Optional[float], List[str]]:
    grb_dir = os.path.join(session.data_dir, str(bnname))
    intervals: List[Dict] = []

    try:
        for item in os.listdir(grb_dir):
            full_path = os.path.join(grb_dir, item)
            if not os.path.isdir(full_path):
                continue
            if not item.startswith("interval"):
                continue

            time_part = item.replace("interval", "")
            if "-" not in time_part:
                continue
            start_str, end_str = time_part.split("-")
            try:
                start_time = float(start_str)
                end_time = float(end_str)
            except ValueError:
                continue

            intervals.append(
                {
                    "start": start_time,
                    "end": end_time,
                    "duration": end_time - start_time,
                    "interval_name": item,
                }
            )
    except Exception:
        pass

    if not intervals:
        return None, None, None, []

    longest = max(intervals, key=lambda x: x["duration"])
    burst_start_time = longest["start"]
    burst_end_time = longest["end"]
    burst_duration = longest["duration"]
    interval_names = [interval["interval_name"] for interval in intervals]

    return burst_start_time, burst_end_time, burst_duration, interval_names


def _export_best_model_param_tex(combined_row: Dict[str, object]) -> None:
    grb_name = str(combined_row.get("grb_name", ""))
    bnname = str(combined_row.get("bnname", ""))
    best_model = str(
        combined_row.get("best_model", combined_row.get("model", ""))
    )

    if not grb_name or not bnname or not best_model:
        return

    out_dir = Path(session.result_root) / grb_name
    tex_path = out_dir / f"{bnname}_best_model_params.tex"

    params: List[Tuple[str, object, object]] = []
    for key, value in combined_row.items():
        if not isinstance(key, str) or not key.endswith("_value"):
            continue
        base = key[: -len("_value")]
        err_key = base + "_error"
        if err_key not in combined_row:
            continue
        err_val = combined_row.get(err_key)

        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        if err_val is None or (isinstance(err_val, float) and np.isnan(err_val)):
            continue

        name_raw = base.split(".")[-1]
        param_name = name_raw
        params.append((param_name, value, err_val))

    if not params:
        return

    lines: List[str] = []
    lines.append(r"\startlongtable")
    lines.append(r"\begin{deluxetable*}{lcc}")
    lines.append(
        "\\tablecaption{Best-fit parameters for %s (%s), model %s\\label{tab:%s_best_params}}"
        % (grb_name, bnname, best_model, bnname)
    )
    lines.append(r"\tablecolumns{3}")
    lines.append(r"\tablewidth{0pt}")
    lines.append(r"\tablehead{")
    lines.append(r"\colhead{Parameter} &")
    lines.append(r"\colhead{Value} &")
    lines.append(r"\colhead{Error}")
    lines.append(r"}")
    lines.append(r"\startdata")

    for name, val, err in params:
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            val_str = f"{val:.4g}"
        else:
            val_str = str(val)
        if isinstance(err, (int, float)) and not isinstance(err, bool):
            err_str = f"{err:.4g}"
        else:
            err_str = str(err)

        lines.append(f"{name} & {val_str} & {err_str} \\\\")

    lines.append(r"\enddata")
    lines.append(
        r"\tablecomments{Best-fit spectral parameters for the preferred model."
        r" Parameter names follow the internal model notation.}"
    )
    lines.append(r"\end{deluxetable*}")

    tex_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"GRB {grb_name} 的最佳模型参数表已保存至 {tex_path}")


def _save_best_models_and_time_analysis(
    df_summary: pd.DataFrame,
    best_models: List[pd.Series],
    time_analysis: List[Dict],
) -> None:
    combined_results: List[Dict] = []

    for row in pd.DataFrame(best_models).iterrows():
        _, row_series = row
        bnname = row_series["bnname"]
        model = row_series["model"]

        grb_name = None
        for item in os.listdir(session.result_root):
            item_path = os.path.join(session.result_root, item)
            if not os.path.isdir(item_path):
                continue
            if bnname in os.listdir(item_path):
                grb_name = item
                break

        if not grb_name:
            continue

        burst_start_time, burst_end_time, burst_duration, interval_names = (
            _extract_burst_time_info(bnname)
        )

        combined_row: Dict[str, object] = {
            "bnname": str(bnname),
            "grb_name": str(grb_name),
            "best_model": str(model),
            "AIC": float(row_series["AIC"]),
            "BIC": float(row_series["BIC"]),
            "burst_start_time": burst_start_time or "",
            "burst_end_time": burst_end_time or "",
            "burst_duration": burst_duration or "",
            "intervals": "; ".join(interval_names),
        }
        combined_row.update(row_series.to_dict())
        combined_results.append(combined_row)

    if combined_results:
        for combined_row in combined_results:
            grb_name_single = combined_row["grb_name"]
            bnname_single = combined_row["bnname"]
            df_single = pd.DataFrame([combined_row])
            best_models_csv = os.path.join(
                session.result_root,
                str(grb_name_single),
                f"{bnname_single}_best_models.csv",
            )

            write_header = True
            if os.path.isfile(best_models_csv):
                try:
                    with open(best_models_csv, "r", encoding="utf-8") as f:
                        first_line = f.readline().strip()
                        if "bnname" in first_line and "grb_name" in first_line:
                            write_header = False
                except Exception:
                    pass

            df_single.to_csv(
                best_models_csv,
                index=False,
                mode="a",
                header=write_header,
            )
            log(
                f"GRB {bnname_single} 的最佳模型和时间分析结果已保存至 "
                f"{best_models_csv}"
            )

            _export_best_model_param_tex(combined_row)

    if time_analysis:
        time_analysis_df = pd.DataFrame(time_analysis)
        time_analysis_csv = os.path.join(
            session.result_root,
            "time_bin_analysis_results.csv",
        )
        time_analysis_df.to_csv(time_analysis_csv, index=False)
        log(f"时间段分析结果已保存至 {time_analysis_csv}")


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

        safescv = os.path.join(grb_dir, f"{os.path.basename(grb_dir)}_allmodel.csv")
        df_grb.to_csv(safescv, mode="w", index=False)
        log(f"GRB {os.path.basename(grb_dir)} 的所有模型结果已保存至 {safescv}")

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

        lines.append(r"\startlongtable")
        lines.append(r"\begin{deluxetable*}{lcccccccc}")
        lines.append(
            "\\tablecaption{Spectral fits for %s (%s)\\label{tab:%s_allbins_models}}"
            % (grb_name_key, bnname_key, bnname_key)
        )
        lines.append(r"\tablecolumns{9}")
        lines.append(r"\tablewidth{0pt}")
        lines.append(r"\tablehead{")
        lines.append(r"\colhead{Time bin (s)} &")
        lines.append(r"\colhead{Model} &")
        lines.append(r"\colhead{$\\alpha$} &")
        lines.append(r"\colhead{$\\beta$} &")
        lines.append(r"\colhead{$E_{p}/E_{c}$} &")
        lines.append(r"\colhead{$kT$} &")
        lines.append(r"\colhead{$F_{BB}\\times10^{6}$} &")
        lines.append(r"\colhead{$F_{Tot}\\times10^{6}$} &")
        lines.append(r"\colhead{BIC} \\")
        lines.append(r"\colhead{} &")
        lines.append(r"\colhead{} &")
        lines.append(r"\colhead{} &")
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
                        ("GRB.spectrum.main.composite.kT_2_value", "GRB.spectrum.main.composite.kT_2_error"),
                    ],
                )

                display_model = {
                    "comp": "CPL",
                    "blackbody": "BB",
                    "band+bb": "Band+BB",
                    "comp+bb": "CPL+BB",
                }.get(model_name, model_name)

                def _fmt_f6(v: object) -> str:
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        return f"{float(v)*1e6:.3f}"
                    return ""

                line = (
                    f"{bin_label}"
                    f" & {display_model}"
                    f" & {_fmt_pm(alpha_v, alpha_e)}"
                    f" & {_fmt_pm(beta_v, beta_e)}"
                    f" & {_fmt_pm(epeak_or_ec_v, epeak_or_ec_e)}"
                    f" & {_fmt_pm(kt_v, kt_e)}"
                    f" & {_fmt_f6(f_bb)}"
                    f" & {_fmt_f6(f_tot)}"
                    f" & {_fmt(bic)}"
                    r" \\"
                )
                lines.append(line)

        lines.append(r"\enddata")
        lines.append(
            r"\tablecomments{Each row corresponds to one time bin and spectral model; "
            r"$F_{BB}$ is computed from the BB component (if present), and $F_{Tot}$ is the total model flux. "
            r"The time intervals are relative to the GBM trigger; energy band is %s.}" % band_tex
        )
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
