# -*- coding: utf-8 -*-
"""LAT FT1/FT2：gtselect / gtbin / 响应与 OGIPLike。"""

from __future__ import annotations

import os
import re
import subprocess
import traceback
from typing import Optional, Tuple

import pandas as pd
from threeML import OGIPLike

import gt_apps as my_apps

from .logging_utils import log
from .session import session


def _find_lat_files(lat_data_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """在 LAT 数据目录中寻找 FT1 / FT2 文件。"""
    ft1_file, ft2_file = None, None

    for file in os.listdir(lat_data_dir):
        if not file.lower().endswith((".fit", ".fits")):
            continue

        filepath = os.path.join(lat_data_dir, file)
        file_upper = file.upper()

        if "FT1" in file_upper or "EV" in file_upper:
            ft1_file = filepath
            print(f"FT1 found: {file}")
        elif "FT2" in file_upper or "SC" in file_upper:
            ft2_file = filepath
            print(f"FT2 found: {file}")

    return ft1_file, ft2_file


def _configure_gtselect_filter(
    t0: float,
    t1: float,
    trig_time: float,
    ra: float,
    dec: float,
    ft1_file: str,
) -> str:
    """配置并运行 gtselect (my_apps.filter)，返回筛选后的事件文件名。"""
    my_apps.filter["evclass"] = 8
    my_apps.filter["ra"] = ra
    my_apps.filter["dec"] = dec
    my_apps.filter["rad"] = 12
    my_apps.filter["emin"] = 30.0  # MeV
    my_apps.filter["emax"] = 100000.0
    my_apps.filter["zmax"] = 100
    my_apps.filter["tmin"] = t0 + trig_time
    my_apps.filter["tmax"] = t1 + trig_time
    my_apps.filter["infile"] = ft1_file

    gtselect_outfile = "prompt_selection_new.fit"
    my_apps.filter["outfile"] = gtselect_outfile
    my_apps.filter.run()

    return gtselect_outfile


def _configure_gtbin_counts_map(
    ft2_file: str,
    evfile: str,
) -> str:
    """配置并运行 gtbin (my_apps.counts_map)，返回 PHA 文件名。"""
    my_apps.counts_map["scfile"] = ft2_file
    my_apps.counts_map["evfile"] = evfile
    my_apps.counts_map["algorithm"] = "PHA1"
    my_apps.counts_map["emin"] = 30.0  # MeV
    my_apps.counts_map["emax"] = 100000.0
    my_apps.counts_map["enumbins"] = 20
    my_apps.counts_map["ebinalg"] = "LOG"

    gtbin_outfile = "lat_new.pha"
    my_apps.counts_map["outfile"] = gtbin_outfile
    my_apps.counts_map.run()

    return gtbin_outfile


def _generate_lat_response(ft2_file: str, pha_file: str) -> str:
    """调用 gtrspgen 为 LAT 生成响应文件，返回响应文件名。"""
    rsp_file = "lat_new.rsp"
    cmd = (
        "gtrspgen "
        f"specfile={pha_file} "
        f"scfile={ft2_file} "
        "respalg=PS "
        f"outfile={rsp_file} "
        "irfs=CALDB "
        "ebinalg=LOG "
        "emin=100.0 emax=100000.0 "
        "enumbins=100 "
        "thetacut=90.0 dcostheta=0.05"
    )
    subprocess.call(cmd, shell=True)
    return rsp_file


def process_lat_data(
    bnname: str,
    t0: float,
    t1: float,
    result_dir: str,  # 与原接口保持一致，虽然当前未直接使用
) -> Optional[OGIPLike]:
    """
    处理 LAT 数据，返回 3ML 的 OGIPLike 插件。
    """
    try:
        log(f"开始处理LAT数据: {bnname}")

        df = pd.read_excel(
            session.fermilat_grb_xls,
            sheet_name="GCN",
            index_col="trigname",
        )
        row = df.loc[bnname]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        grb_name = re.sub(r"\s+", "", row["gcn_name"])
        trig_time = row["trigger_met"]
        ra_str, dec_str = row["ra,dec"].split(",")
        ra, dec = float(ra_str), float(dec_str)

        ext_root = getattr(
            session,
            "extended_lat_data_root",
            "/home/mxr/lee/data/fermilat/Extended_data_ex",
        )
        fallback_dir = os.path.join(
            os.path.expanduser(ext_root), grb_name
        )
        local_bn_dir = os.path.join(
            session.result_root, grb_name, bnname
        )
        if os.path.isdir(local_bn_dir) and os.listdir(local_bn_dir):
            lat_data_dir = local_bn_dir
            log(f"使用结果目录下已复制的 LAT 数据: {lat_data_dir}")
        else:
            lat_data_dir = fallback_dir

        if not os.path.exists(lat_data_dir):
            log(f"警告: LAT数据目录不存在: {lat_data_dir}")
            return None

        ft1_file, ft2_file = _find_lat_files(lat_data_dir)
        if not ft1_file or not ft2_file:
            log(
                "警告: 无法找到必要的LAT文件 "
                f"(ft1: {ft1_file}, ft2: {ft2_file})"
            )
            return None

        log(f"找到LAT文件: ft1={ft1_file}, ft2={ft2_file}")

        if t0 <= 0:
            t0 = 0.0
        if bnname == "bn231222310":
            t0, t1 = float(0.1), float(85)
        if bnname == "bn221023862":
            t0, t1 = float(8), float(30)

        gtselect_outfile = _configure_gtselect_filter(
            t0=t0,
            t1=t1,
            trig_time=trig_time,
            ra=ra,
            dec=dec,
            ft1_file=ft1_file,
        )

        pha_file = _configure_gtbin_counts_map(
            ft2_file=ft2_file,
            evfile=gtselect_outfile,
        )

        rsp_file = _generate_lat_response(ft2_file=ft2_file, pha_file=pha_file)

        lat_transient = OGIPLike("LAT", observation=pha_file, response=rsp_file)
        lat_transient.set_active_measurements("100000.-100000000.")  # keV

        return lat_transient

    except Exception as exc:  # noqa: BLE001
        log(f"警告: LAT数据处理失败: {exc}")
        traceback.print_exc()
        return None
