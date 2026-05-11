# -*- coding: utf-8 -*-

from __future__ import annotations

from .session import session


def log(msg: str) -> None:
    """简单日志输出，统一带 [LOG] 前缀；若已打开会话日志则同步写入文件。"""
    line = f"[LOG] {msg}"
    print(line)
    if session.log_file_handle is not None:
        session.log_file_handle.write(line + "\n")
        session.log_file_handle.flush()
