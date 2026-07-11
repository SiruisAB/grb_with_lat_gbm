# -*- coding: utf-8 -*-
"""Helpers for bounded multiprocessing worker execution."""

from __future__ import annotations

import queue as queue_module
import time


def wait_for_worker_result(proc, result_queue, timeout_s: float):
    deadline = time.monotonic() + timeout_s
    while True:
        wait_s = min(0.1, max(0.0, deadline - time.monotonic()))
        try:
            result = result_queue.get(timeout=wait_s)
            break
        except queue_module.Empty as exc:
            if proc.exitcode is not None:
                proc.join(timeout=5.0)
                raise RuntimeError(
                    f"worker exited with code {proc.exitcode} without returning a result"
                ) from exc
            if time.monotonic() >= deadline:
                if proc.is_alive() or proc.exitcode is None:
                    proc.terminate()
                proc.join(timeout=5.0)
                raise TimeoutError(
                    f"worker did not return within {timeout_s:g} seconds"
                ) from exc

    proc.join(timeout=5.0)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5.0)
    return result
