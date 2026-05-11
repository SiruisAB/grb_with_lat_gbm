# -*- coding: utf-8 -*-
"""
GBM/LAT 光谱分析工程包。

轻量符号（不导入 threeML）：``GRBProject``、``GRBProjectConfig``、``session`` 等。

完整流水线请使用子模块，例如::

    from grb_project.pipeline import main, cli_main, parse_args
    from grb_project.analyze_single import analyze_grb
"""

from .config import GRBProjectConfig, GRBRunOverrides, run_overrides_from_config
from .project import GRBProject, apply_project_config
from .session import session, set_result_root

__all__ = [
    "GRBProject",
    "GRBProjectConfig",
    "GRBRunOverrides",
    "apply_project_config",
    "run_overrides_from_config",
    "session",
    "set_result_root",
]
