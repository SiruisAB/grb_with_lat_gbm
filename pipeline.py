# -*- coding: utf-8 -*-
"""命令行与批量分析主流程。"""

from __future__ import annotations

import argparse
import os
import re
import sys
import traceback
from typing import Dict, List, Optional, Sequence

import pandas as pd
from tqdm import tqdm

from .analyze_single import analyze_grb
from .config import GRBRunOverrides
from .io_utils import (
    _append_exception_log,
    _append_text_report,
    ensure_dir,
    read_catalog,
    read_gcn_bn_triggers,
)
from .logging_utils import log
from .runtime_env import ensure_analysis_runtime
from .session import export_legacy_aliases, session, set_result_root
from .summary_export import (
    _append_time_bin_info,
    _compute_best_models_and_time_analysis,
    _save_all_models_per_grb,
    _save_best_models_and_time_analysis,
)

ensure_analysis_runtime()


def _sync_legacy_if_loaded() -> None:
    m = sys.modules.get("gbmtest_enhanced1_refactored")
    if m is not None:
        export_legacy_aliases(m.__dict__)


def main(
    target_grbs: Optional[Sequence[str]] = None,
    analysis_mode: str = "gbm+lat",
    result_root: Optional[str] = None,
    summary_csv_name: Optional[str] = None,
    session_log: bool = False,
    fixed_num_time_bins: Optional[int] = None,
    run_overrides: Optional[GRBRunOverrides] = None,
) -> None:
    try:
        if result_root:
            set_result_root(
                result_root,
                summary_csv_name or "summary_results2.csv",
            )

        _sync_legacy_if_loaded()

        ensure_dir(session.result_root)
        if session_log:
            log_path = os.path.join(session.result_root, "analysis_session.log")
            session.log_file_handle = open(log_path, "a", encoding="utf-8")
            _sync_legacy_if_loaded()
            log(f"会话日志文件: {log_path}")

        log("读取 GRB 数据表...")

        df_all = pd.read_excel(session.catalog_xls, sheet_name="fermigbrst")
        df_filtered = df_all.copy()

        df_catalog = read_catalog()

        if target_grbs is not None:
            if isinstance(target_grbs, str):
                target_grbs = [target_grbs]

            target_grbs = list(target_grbs)

            resolved_bnnames: List[str] = []
            if "trigger_name" in df_filtered.columns:
                existing_bn = set(df_filtered["trigger_name"].astype(str))
            elif "bnname" in df_filtered.columns:
                existing_bn = set(df_filtered["bnname"].astype(str))
            else:
                existing_bn = set()

            try:
                df_lat_names = pd.read_excel(
                    session.fermilat_grb_xls,
                    sheet_name="GCN",
                    index_col="trigname",
                )
                grb_to_bn = {
                    re.sub(r"\s+", "", str(row["gcn_name"])): str(idx)
                    for idx, row in df_lat_names.iterrows()
                    if str(idx).strip().lower() != "nan" and str(idx).startswith("bn")
                }
            except Exception:
                grb_to_bn = {}

            for name in target_grbs:
                name_str = str(name)

                if name_str in existing_bn:
                    resolved_bnnames.append(name_str)
                    continue

                key = re.sub(r"\s+", "", name_str)
                if key in grb_to_bn:
                    bn = grb_to_bn[key]
                    if bn in existing_bn:
                        resolved_bnnames.append(bn)
                        continue

                log(
                    f"警告: 无法在目录中找到 {name_str} 对应的 GRB"
                    "（既不是 bnname 也不是已知 grb_name）"
                )

            if not resolved_bnnames:
                log("警告: 未能解析任何有效的 GRB 名称，程序退出。")
                return

            if "trigger_name" in df_filtered.columns:
                mask = df_filtered["trigger_name"].astype(str).isin(resolved_bnnames)
                df_filtered = df_filtered[mask]
                log(
                    f"指定分析 {len(df_filtered)} 个 GRB (按 trigger_name 匹配): "
                    f"{list(df_filtered['trigger_name'])}"
                )
            elif "bnname" in df_filtered.columns:
                mask = df_filtered["bnname"].astype(str).isin(resolved_bnnames)
                df_filtered = df_filtered[mask]
                log(
                    f"指定分析 {len(df_filtered)} 个 GRB (按 bnname 匹配): "
                    f"{list(df_filtered['bnname'])}"
                )
            else:
                log("警告: 数据中没有找到trigger_name或bnname列")
                return
        else:
            log(f"共 {len(df_filtered)} 个 GRB 待处理")

        summary: List[Dict] = []

        for i in tqdm(
            range(len(df_filtered)),
            desc="Processing GRBs",
            ncols=80,
        ):
            if "trigger_name" in df_filtered.columns:
                bn_label = str(df_filtered["trigger_name"].iloc[i])
            elif "bnname" in df_filtered.columns:
                bn_label = str(df_filtered["bnname"].iloc[i])
            else:
                bn_label = f"row_{i}"

            try:
                analyze_grb(
                    idx=i,
                    model_str="band",
                    df_filtered=df_filtered,
                    df_catalog=df_catalog,
                    summary_list=summary,
                    fixed_num_time_bins=fixed_num_time_bins,
                    analysis_mode=analysis_mode,
                    run_overrides=run_overrides,
                )
            except Exception as exc:  # noqa: BLE001
                log(f"⚠️ {bn_label} 出错: {exc}")
                tb_str = traceback.format_exc()
                traceback.print_exc()
                _append_exception_log(session.result_root, bn_label, tb_str)

        df_summary = pd.DataFrame(summary)

        _append_time_bin_info(df_summary)

        best_models, time_analysis = _compute_best_models_and_time_analysis(
            df_summary
        )
        _save_best_models_and_time_analysis(df_summary, best_models, time_analysis)

        _save_all_models_per_grb(df_summary)

    except Exception as exc:  # noqa: BLE001
        log(f"主程序错误: {exc}")
        tb_str = traceback.format_exc()
        traceback.print_exc()
        _append_exception_log(session.result_root, "main/fatal", tb_str)
    finally:
        if session.log_file_handle is not None:
            session.log_file_handle.close()
            session.log_file_handle = None
            _sync_legacy_if_loaded()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fermi GBM GRB 光谱分析脚本（重构版，与原版等价）"
    )
    parser.add_argument(
        "--grbs",
        nargs="+",
        help=(
            "指定要分析的GRB（可以是 bnname 或 GRB 标准名），"
            "例如：bn220617772 或 GRB220617A"
        ),
    )
    parser.add_argument(
        "--analysis-mode",
        choices=["gbm", "gbm+lat"],
        default="gbm+lat",
        help="选择分析 GBM 还是 GBM+LAT 联合（默认 gbm+lat）",
    )
    parser.add_argument(
        "--result-root",
        default=None,
        help="结果输出根目录（覆盖默认 results2）；与 --gcn-all 联用时未指定则默认为 results3",
    )
    parser.add_argument(
        "--gcn-all",
        action="store_true",
        help=(
            "处理 fermilat-grb.xls（GCN）表中全部带 bn 触发名的暴；"
            "仅保留在 GBMcatolog fermigbrst 中存在的触发；"
            "默认结果目录为 /home/mxr/lee/gbmtest/results3，并写入 "
            "analysis_session.log / grb_exceptions.log"
        ),
    )
    parser.add_argument(
        "--session-log",
        action="store_true",
        help=(
            "将运行日志追加写入结果目录下的 analysis_session.log；"
            "异常仍写入 grb_exceptions.log（与 --gcn-all 默认行为一致）"
        ),
    )
    parser.add_argument(
        "--lat-extended-three-ml",
        action="store_true",
        help=(
            "在每个 GRB 完成复制 Extended 数据并进入结果目录之后、"
            "在分时间 bin / LAT 插件 / 光变 / GBM 拟合之前，"
            "运行 LAT Extended + GtBurst + threeML 全流程；需与 --analysis-mode gbm+lat 同用"
        ),
    )
    parser.add_argument(
        "--plot-joint-lightcurve",
        dest="plot_joint_lightcurve",
        action="store_true",
        help="启用 lightcurves 联合光变绘制（默认已启用；通常无需单独指定）",
    )
    parser.add_argument(
        "--no-plot-joint-lightcurve",
        dest="plot_joint_lightcurve",
        action="store_false",
        help=(
            "关闭联合光变图（不调用 lightcurves.plot_gbm_lat_lightcurve_figure；"
            "不生成 {grb}/{bn}_lightcurve.png）"
        ),
    )
    parser.set_defaults(plot_joint_lightcurve=True)
    return parser.parse_args()


def _cli_run_overrides(args: argparse.Namespace) -> Optional[GRBRunOverrides]:
    """合并命令行上的 LAT Extended 与联合光变开关为单个 GRBRunOverrides。"""
    lat_on = (
        getattr(args, "lat_extended_three_ml", False)
        and args.analysis_mode == "gbm+lat"
    )
    if getattr(args, "lat_extended_three_ml", False) and args.analysis_mode != "gbm+lat":
        log(
            "警告: --lat-extended-three-ml 仅在 gbm+lat 模式下运行 LAT Extended 流水线；"
            f"当前为 {args.analysis_mode!r}，已跳过该流水线。"
        )
    lc_on = bool(getattr(args, "plot_joint_lightcurve", True))
    if lat_on and not lc_on:
        return GRBRunOverrides(
            lat_three_ml_full=True,
            plot_joint_lightcurve=False,
        )
    if lat_on:
        return GRBRunOverrides(lat_three_ml_full=True)
    if not lc_on:
        return GRBRunOverrides(plot_joint_lightcurve=False)
    return None


def cli_main() -> None:
    """供 ``python -m grb_project`` 或顶层脚本调用。"""
    args = parse_args()

    cli_ov = _cli_run_overrides(args)

    if args.gcn_all and args.grbs:
        print(
            "[LOG] 错误: 不能同时使用 --gcn-all 与 --grbs，请只选其一。",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.gcn_all:
        batch_root = args.result_root or session.result_root_gcn_batch
        gcn_bns, skipped_trigname = read_gcn_bn_triggers()
        df_all = pd.read_excel(session.catalog_xls, sheet_name="fermigbrst")
        trig_col = "trigger_name" if "trigger_name" in df_all.columns else "bnname"
        existing_bn = set(df_all[trig_col].astype(str))

        resolved = [b for b in gcn_bns if b in existing_bn]
        not_in_cat = [b for b in gcn_bns if b not in existing_bn]

        ensure_dir(batch_root)
        if skipped_trigname:
            lines = "\n".join(skipped_trigname) + "\n"
            _append_text_report(
                batch_root,
                "gcn_skipped_missing_or_invalid_trigname.txt",
                lines,
            )
        if not_in_cat:
            _append_text_report(
                batch_root,
                "gcn_skipped_not_in_fermigbrst.txt",
                "\n".join(not_in_cat) + "\n",
            )

        print(
            f"[LOG] --gcn-all: GCN 中有效 bn {len(gcn_bns)} 个，"
            f"在 fermigbrst 中可运行 {len(resolved)} 个；"
            f"未在目录表中: {len(not_in_cat)}；"
            f"缺少/无效 trigname 行: {len(skipped_trigname)}"
        )

        if not resolved:
            print(
                "[LOG] 错误: 没有可在 GBMcatolog 中匹配的 GCN 触发，已退出。",
                file=sys.stderr,
            )
            sys.exit(1)

        main(
            resolved,
            args.analysis_mode,
            result_root=batch_root,
            summary_csv_name="summary_results3.csv",
            session_log=True,
            run_overrides=cli_ov,
        )
    else:
        main(
            args.grbs,
            args.analysis_mode,
            result_root=args.result_root,
            session_log=args.session_log,
            run_overrides=cli_ov,
        )
