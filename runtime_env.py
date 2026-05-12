# -*- coding: utf-8 -*-
"""3ML / numpy / warnings 等与分析相关的进程级初始化。"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
from threeML import silence_warnings

_initialized = False
_gtburst_commands_on_path = False


def _ensure_gtburst_commands_on_path() -> None:
    """
    GtBurst 的 ``doTimeResolvedLike.py`` 通过 shell 调用 ``gtdownloadLATdata.py``；
    conda 安装的脚本在 ``GtBurst/commands/`` 下，默认不在 PATH 中。
    """
    global _gtburst_commands_on_path
    if _gtburst_commands_on_path:
        return
    try:
        import GtBurst
    except ImportError:
        _gtburst_commands_on_path = True
        return
    commands = Path(GtBurst.__file__).resolve().parent / "commands"
    if not commands.is_dir():
        _gtburst_commands_on_path = True
        return
    prefix = str(commands)
    path = os.environ.get("PATH", "")
    parts = path.split(os.pathsep) if path else []
    if prefix not in parts:
        os.environ["PATH"] = prefix + os.pathsep + path
    # ``doTimeResolvedLike`` 用 shell 直接执行该脚本；conda 包常无 +x，会导致 Permission denied
    for _name in ("gtdownloadLATdata.py",):
        _p = commands / _name
        if _p.is_file():
            try:
                _p.chmod(_p.stat().st_mode | 0o111)
            except OSError:
                pass
    _gtburst_commands_on_path = True


def ensure_analysis_runtime() -> None:
    _ensure_gtburst_commands_on_path()
    global _initialized
    if _initialized:
        return
    silence_warnings()
    warnings.simplefilter("ignore")
    np.seterr(all="ignore")
    _initialized = True
