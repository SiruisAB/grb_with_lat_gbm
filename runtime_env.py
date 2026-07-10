# -*- coding: utf-8 -*-
"""3ML / numpy / warnings 等与分析相关的进程级初始化。"""

from __future__ import annotations

import os
import tempfile
import warnings
from pathlib import Path


def _ensure_writable_runtime_dirs() -> None:
    """将三方库的用户级缓存/日志目录重定向到可写位置。"""
    base = Path(os.environ.get("GBM_PROJECT_RUNTIME_DIR", tempfile.gettempdir())) / "grb_project_runtime"
    home_dir = base / "home"
    for subdir in ("config", "cache", "data", "state", "home"):
        target = base / subdir
        target.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CONFIG_HOME"] = str(base / "config")
    os.environ["XDG_CACHE_HOME"] = str(base / "cache")
    os.environ["XDG_DATA_HOME"] = str(base / "data")
    os.environ["XDG_STATE_HOME"] = str(base / "state")
    os.environ["HOME"] = str(home_dir)


_ensure_writable_runtime_dirs()

import numpy as np

from .logging_utils import log

try:
    from threeML import silence_warnings
except ModuleNotFoundError:  # pragma: no cover
    def silence_warnings() -> None:  # type: ignore[no-redef]
        return None

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


def ensure_threeML_minimizer_for_background_fit() -> None:
    """
    ``set_background_interval`` 会调用 threeML 多项式本底拟合；``JointLikelihood`` 初始化需要
    已注册的 MLE minimizer（默认常为 minuit）。若 minuit 未安装会抛出 ``MinimizerNotAvailable``。
    在进程内将 threeML 默认 minimizer 设为当前环境第一个可用的（通常 SCIPY）。
    """
    try:
        from threeML.config.config import threeML_config
        from threeML.minimizer import minimization
    except ModuleNotFoundError:
        return

    available = sorted(minimization._minimizers.keys())
    if not available:
        log(
            "threeML 未注册任何 MLE minimizer，GBM 背景窗拟合将失败。"
            "请在 threeML 环境中安装 iminuit 或确保 scipy 可用。"
        )
        return

    try:
        default_key = str(threeML_config["mle"]["default_minimizer"].value).upper()
    except Exception:
        default_key = "MINUIT"

    if default_key in minimization._minimizers:
        return

    fallback = "SCIPY" if "SCIPY" in minimization._minimizers else available[0]
    try:
        threeML_config["mle"]["default_minimizer"].value = fallback.lower()
    except Exception:
        pass
    log(
        f"threeML 默认 minimizer「{default_key}」不可用，已改用「{fallback}」用于 GBM 背景窗拟合；"
        f"当前可用: {available}。若需 minuit：conda install -c conda-forge iminuit"
    )


def ensure_analysis_runtime() -> None:
    _ensure_gtburst_commands_on_path()
    global _initialized
    if _initialized:
        return
    silence_warnings()
    warnings.simplefilter("ignore")
    np.seterr(all="ignore")
    ensure_threeML_minimizer_for_background_fit()
    _initialized = True
