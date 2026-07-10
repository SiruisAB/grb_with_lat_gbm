# -*- coding: utf-8 -*-
"""单次分析入口，由 Web 工作台和项目对象复用。"""

from __future__ import annotations

from typing import Optional

from .config import GRBRunOverrides


def main(
    target_grbs=None,
    analysis_mode=None,
    result_root=None,
    summary_csv_name=None,
    session_log: bool = False,
    fixed_num_time_bins=None,
    run_overrides: Optional[GRBRunOverrides] = None,
) -> None:
    from .project import run_single_analysis

    run_single_analysis(
        target_grbs=target_grbs,
        analysis_mode=analysis_mode,
        result_root=result_root,
        summary_csv_name=summary_csv_name,
        fixed_num_time_bins=fixed_num_time_bins,
        run_overrides=run_overrides,
    )
