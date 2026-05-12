# -*- coding: utf-8 -*-
"""GBM 探测器几何选择：由 trigdat 与 RSP 头得到卫星轴向与源方向，再调用 ``gbm_selector``。"""

from __future__ import annotations

from typing import List, Tuple

import gbm_selector
from astropy.io import fits as pyfits

from .io_utils import find_files


def select_gbm_detectors(grb_dir: str) -> Tuple[List[str], float, float, float, float]:
    """根据 GRB 工作目录下的 trigdat 与 rsp2 选择最优 NaI/BGO 探测器列表。

    :returns: ``(detector_names, ra_scx, dec_scx, ra_scz, dec_scz)``
    """
    with pyfits.open(
        next(f for f in find_files(grb_dir, ".fit") if "trigdat_all" in f)
    ) as trig:
        ra_scx, dec_scx = trig[0].header["RA_SCX"], trig[0].header["DEC_SCX"]
        ra_scz, dec_scz = trig[0].header["RA_SCZ"], trig[0].header["DEC_SCZ"]

    rsp_file = find_files(grb_dir, ".rsp2")[0]

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

    dets = gbm_selector.select_best_detectors(
        ra_scx,
        dec_scx,
        ra_scz,
        dec_scz,
        ra_obj,
        dec_obj,
    )
    return dets, ra_scx, dec_scx, ra_scz, dec_scz
