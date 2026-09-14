# -*- coding: utf-8 -*-
"""全局路径与会话状态。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TextIO


@dataclass
class AnalysisSessionState:
    data_dir: str = "/home/mxr/lee/gbmtest/GBM_data"
    catalog_xls: str = "/home/mxr/lee/data/GBMcatolog.xls"
    merged_xls: str = "/home/mxr/lee/newflietest/merged_grb_fit_results.xlsx"
    fermilat_grb_xls: str = "/home/mxr/lee/fermilat-grb.xls"
    result_root: str = "/home/mxr/lee/gbmtest/results_last"
    summary_csv_name: str = "summary_results.csv"
    log_file_handle: Optional[TextIO] = None
    extended_lat_data_root: str = "/home/mxr/lee/data/fermilat/Extended_data_ex"
    joint_target_csv: str = "/home/mxr/lee/gbmtest/lat_download_targets.csv"
    copy_extended_lat_to_bn_dir: bool = True
    lat_extended_three_ml_pipeline: bool = False
    default_lat_irfs: str = "p8_transient010e"
    analysis_stop_requested: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def summary_csv(self) -> str:
        return os.path.join(self.result_root, self.summary_csv_name)


class AnalysisCancelled(RuntimeError):
    pass


def request_stop() -> None:
    session.analysis_stop_requested = True


def clear_stop_request() -> None:
    session.analysis_stop_requested = False


def check_stop_requested() -> None:
    if session.analysis_stop_requested:
        raise AnalysisCancelled("分析已被用户中止。")


session = AnalysisSessionState()


def ensure_result_root() -> str:
    return session.result_root


def set_result_root(root: str, summary_csv_name: str = "summary_results.csv") -> None:
    session.result_root = os.path.abspath(os.path.expanduser(root))
    session.summary_csv_name = summary_csv_name
