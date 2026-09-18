# -*- coding: utf-8 -*-
"""GBM 探测器几何选择：由 trigdat 与 RSP 头得到卫星轴向与源方向，再调用 ``gbm_selector``。"""

from __future__ import annotations

from typing import List, Tuple

from astropy.io import fits as pyfits

from .io_utils import find_files, find_files_any


def select_gbm_detectors(grb_dir: str) -> Tuple[List[str], float, float, float, float]:
    """根据 GRB 工作目录下的 trigdat 与 rsp2 选择最优 NaI/BGO 探测器列表。

    :returns: ``(detector_names, ra_scx, dec_scx, ra_scz, dec_scz)``
    """
    trigdat_candidates = [
        f for f in find_files(grb_dir, ".fit") if "trigdat_all" in f
    ]
    if not trigdat_candidates:
        raise FileNotFoundError(
            f"{grb_dir}: 未找到 trigdat（glg_trigdat_all_*.fit），无法选择探测器；"
            "该目录可能只是下载中断留下的空壳"
        )
    with pyfits.open(trigdat_candidates[0]) as trig:
        ra_scx, dec_scx = trig[0].header["RA_SCX"], trig[0].header["DEC_SCX"]
        ra_scz, dec_scz = trig[0].header["RA_SCZ"], trig[0].header["DEC_SCZ"]

    rsp_candidates = find_files_any(grb_dir, (".rsp2", ".rsp"))
    if not rsp_candidates:
        raise FileNotFoundError(
            f"{grb_dir}: 未找到响应矩阵（.rsp2 或 .rsp），无法选择探测器"
        )
    rsp_file = rsp_candidates[0]

    ra_obj = pyfits.getval(
        rsp_file,
        "RA_OBJ",
        extname="SPECRESP MATRIX",
        extver=1,
    )
    dec_obj = pyfits.getval(
        rsp_file,
        "DEC_OBJ",
        extname="SPECRESP MATRIX",
        extver=1,
    )

    import gbm_selector

    dets = gbm_selector.select_best_detectors(
        ra_scx,
        dec_scx,
        ra_scz,
        dec_scz,
        ra_obj,
        dec_obj,
    )
    return dets, ra_scx, dec_scx, ra_scz, dec_scz
