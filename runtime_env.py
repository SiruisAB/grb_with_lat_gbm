# -*- coding: utf-8 -*-
"""3ML / numpy / warnings 等与分析相关的进程级初始化。"""

from __future__ import annotations

import warnings

import numpy as np
from threeML import silence_warnings

_initialized = False


def ensure_analysis_runtime() -> None:
    global _initialized
    if _initialized:
        return
    silence_warnings()
    warnings.simplefilter("ignore")
    np.seterr(all="ignore")
    _initialized = True
