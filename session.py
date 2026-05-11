# -*- coding: utf-8 -*-
"""全局路径与会话状态（取代原脚本内可变模块常量）。"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, TextIO


@dataclass
class AnalysisSessionState:
    data_dir: str = "/home/mxr/lee/gbmtest/GBM_data"
    catalog_xls: str = "/home/mxr/lee/data/GBMcatolog.xls"
    merged_xls: str = "/home/mxr/lee/newflietest/merged_grb_fit_results.xlsx"
    fermilat_grb_xls: str = "/home/mxr/lee/fermilat-grb.xls"
    result_root: str = "/home/mxr/lee/gbmtest/results3"
    summary_csv_name: str = "summary_results.csv"
    result_root_gcn_batch: str = "/home/mxr/lee/gbmtest/results3"
    log_file_handle: Optional[TextIO] = None
    """LAT Extended 原始数据根目录（其下为各 ``{grb_name}`` 子目录）。"""
    extended_lat_data_root: str = "/home/mxr/lee/data/fermilat/Extended_data_ex"
    """是否在 LAT 分析前复制 Extended 数据到 ``结果目录/{grb_name}/{bn_name}/``。"""
    copy_extended_lat_to_bn_dir: bool = True
    """
    是否在复制数据后执行完整 threeML/GtBurst LAT 扩展流水线（原 ``new.py``）。
    耗时较长；可按需在会话或 GRBRunOverrides 中开启。
    """
    lat_extended_three_ml_pipeline: bool = False

    @property
    def summary_csv(self) -> str:
        return os.path.join(self.result_root, self.summary_csv_name)


session = AnalysisSessionState()


def set_result_root(root: str, summary_csv_name: str = "summary_results2.csv") -> None:
    session.result_root = os.path.abspath(os.path.expanduser(root))
    session.summary_csv_name = summary_csv_name
    _maybe_sync_legacy()


def export_legacy_aliases(target: Dict[str, Any]) -> None:
    """将 session 快照写入兼容模块命名空间（如 gbmtest_enhanced1_refactored_globals）。"""
    target["DATA_DIR"] = session.data_dir
    target["CATALOG_XLS"] = session.catalog_xls
    target["MERGED_XLS"] = session.merged_xls
    target["FERMILAT_GRB_XLS"] = session.fermilat_grb_xls
    target["RESULT_ROOT"] = session.result_root
    target["SUMMARY_CSV"] = session.summary_csv
    target["RESULT_ROOT_GCN_BATCH"] = session.result_root_gcn_batch
    target["_LOG_FILE_HANDLE"] = session.log_file_handle


def _maybe_sync_legacy() -> None:
    m = sys.modules.get("gbmtest_enhanced1_refactored")
    if m is not None:
        export_legacy_aliases(m.__dict__)
