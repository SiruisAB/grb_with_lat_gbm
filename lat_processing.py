# -*- coding: utf-8 -*-
"""LAT FT1/FT2：gtselect / gtbin / 响应与 OGIPLike。"""

from __future__ import annotations

import os
import re
import subprocess
import traceback
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from .logging_utils import log
from .runtime_env import ensure_analysis_runtime
from .session import session

ensure_analysis_runtime()

try:
    from threeML import OGIPLike
    import gt_apps as my_apps
except ModuleNotFoundError:  # pragma: no cover
    OGIPLike = object  # type: ignore[assignment]
    my_apps = None  # type: ignore[assignment]

_P8R3_IRF_EVCLASS: dict[str, int] = {
    "p8_transient100e": 2,
    "p8_transient100": 4,
    "p8_transient020e": 8,
    "p8_transient020": 16,
    "p8_transient010e": 32,
    "p8_transient010": 64,
    "p8_source": 128,
    "p8_clean": 256,
    "p8_ultraclean": 512,
    "p8_ultracleanveto": 1024,
    "p8_sourceveto": 2048,
}


def _gtselect_evclass_for_session() -> int:
    raw = getattr(session, "default_lat_irfs", None) or "p8_transient010e"
    key = str(raw).strip().lower().replace("-", "_")
    if key in _P8R3_IRF_EVCLASS:
        return _P8R3_IRF_EVCLASS[key]
    for prefix in ("p8r3_", "p8r2_", "p8_"):
        if key.startswith(prefix):
            short = "p8_" + key[len(prefix) :]
            if short in _P8R3_IRF_EVCLASS:
                return _P8R3_IRF_EVCLASS[short]
    log(f"警告: 未识别的 LAT IRF「{raw}」，gtselect evclass 使用 32 (TRANSIENT010E)")
    return 32


def _resolve_lat_data_dir(grb_name: str, ext_root: Optional[str] = None) -> str:
    root = ext_root or getattr(session, "extended_lat_data_root", "/home/mxr/lee/data/fermilat/Extended_data_ex")
    return os.path.join(os.path.expanduser(root), grb_name)


def _find_lat_files(lat_data_dir: str, bnname: str) -> Tuple[Optional[str], Optional[str]]:
    prefer_ft1 = os.path.join(lat_data_dir, f"gll_ft1_tr_{bnname}_v00.fit")
    prefer_ft2 = os.path.join(lat_data_dir, f"gll_ft2_tr_{bnname}_v00.fit")
    ft1_file = prefer_ft1 if os.path.isfile(prefer_ft1) else None
    ft2_file = prefer_ft2 if os.path.isfile(prefer_ft2) else None
    for file in sorted(os.listdir(lat_data_dir)):
        if not file.lower().endswith((".fit", ".fits")):
            continue
        filepath = os.path.join(lat_data_dir, file)
        file_upper = file.upper()
        if ft1_file is None and ("FT1" in file_upper or re.search(r"EV\d+", file_upper)):
            ft1_file = filepath
            log(f"LAT FT1 选用: {file}")
        elif ft2_file is None and ("FT2" in file_upper or re.search(r"SC\d+", file_upper)):
            ft2_file = filepath
            log(f"LAT FT2 选用: {file}")
    return ft1_file, ft2_file


def _gtselect_output_has_events(gtselect_outfile: str) -> bool:
    """检查 gtselect 输出里 GTI 与 EVENTS 是否非空。

    LAT 辐射窗晚于分析 bin 的暴（2FLGC 里 T0 可到数千秒）在 prompt bin
    里选出 0 个事件，GTI 为空——这是科学上正常的情况，应优雅跳过；
    不检查的话会一路走到 gtrspgen 才以难看的异常链失败。
    """
    try:
        from astropy.io import fits as pyfits

        with pyfits.open(gtselect_outfile) as hdul:
            for hdu in hdul:
                if hdu.name == "GTI" and int(hdu.header.get("NAXIS2", 0)) == 0:
                    return False
            events = next((h for h in hdul if h.name == "EVENTS"), None)
            if events is None or int(events.header.get("NAXIS2", 0)) == 0:
                return False
        return True
    except Exception:  # noqa: BLE001
        # 检查本身失败时不拦截，让后续工具报出真正的错误
        return True


def _configure_gtselect_filter(t0: float, t1: float, trig_time: float, ra: float, dec: float, ft1_file: str, out_dir: str) -> str:
    if my_apps is None:
        raise ModuleNotFoundError("threeML/gt_apps 未安装")
    evclass = _gtselect_evclass_for_session()
    my_apps.filter["evclass"] = evclass
    my_apps.filter["ra"] = ra
    my_apps.filter["dec"] = dec
    my_apps.filter["rad"] = 12
    my_apps.filter["emin"] = 30.0
    my_apps.filter["emax"] = 100000.0
    my_apps.filter["zmax"] = 100
    my_apps.filter["tmin"] = t0 + trig_time
    my_apps.filter["tmax"] = t1 + trig_time
    my_apps.filter["infile"] = ft1_file
    interval_tag = f"LAT_{t0:g}_{t1:g}"
    gtselect_outfile = os.path.join(out_dir, f"prompt_selection_new_{interval_tag}.fit")
    my_apps.filter["outfile"] = gtselect_outfile
    stdin, stdout = my_apps.filter.runWithOutput(print_command=True)
    lines: list[str] = []
    bad = False
    try:
        for line in stdout:
            lines.append(line)
            if "at the top level:" in line:
                bad = True
    finally:
        stdin.close()
        stdout.close()
    if bad:
        raise RuntimeError(f"gtselect execution failed (evclass={evclass}). Tool output tail:\n{''.join(lines[-50:])}")
    log(f"gtselect 完成: evclass={evclass}, outfile={gtselect_outfile}")
    return gtselect_outfile


def _configure_gtbin_counts_map(ft2_file: str, evfile: str) -> str:
    if my_apps is None:
        raise ModuleNotFoundError("threeML/gt_apps 未安装")
    my_apps.counts_map["scfile"] = ft2_file
    my_apps.counts_map["evfile"] = evfile
    my_apps.counts_map["algorithm"] = "PHA1"
    my_apps.counts_map["emin"] = 30.0
    my_apps.counts_map["emax"] = 100000.0
    my_apps.counts_map["enumbins"] = 20
    my_apps.counts_map["ebinalg"] = "LOG"
    ev_stem = os.path.splitext(os.path.basename(evfile))[0]
    gtbin_outfile = f"{ev_stem}.pha"
    my_apps.counts_map["outfile"] = gtbin_outfile
    my_apps.counts_map.run()
    return gtbin_outfile


def _generate_lat_response(ft2_file: str, pha_file: str) -> str:
    pha_stem = os.path.splitext(os.path.basename(pha_file))[0]
    rsp_file = f"{pha_stem}.rsp"
    cmd = [
        "gtrspgen",
        f"specfile={pha_file}",
        f"scfile={ft2_file}",
        "respalg=PS",
        f"outfile={rsp_file}",
        "irfs=CALDB",
        "ebinalg=LOG",
        "emin=100.0",
        "emax=100000.0",
        "enumbins=100",
        "thetacut=90.0",
        "dcostheta=0.05",
    ]
    subprocess.run(cmd, check=True)
    return rsp_file


def _lookup_lat_target_info(bnname: str) -> Optional[dict]:
    """按 bnname 取 LAT 目标信息（grb_name / trigger_met / ra / dec）。

    先查 fermilat-grb.xls 的 GCN 精选表（2022+ 手工维护，原行为）；
    查不到再退回联合目标交叉表 lat_download_targets.csv——它覆盖全部
    已下载 LAT 数据的暴（2008 起），批量的旧暴此前在这里 KeyError，
    导致 LAT 插件建不起来、拟合静默降级为 GBM-only。
    """
    try:
        df = pd.read_excel(session.fermilat_grb_xls, sheet_name="GCN", index_col="trigname")
        row = df.loc[bnname]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        grb_name = re.sub(r"\s+", "", str(row["gcn_name"]))
        trig_time = float(row["trigger_met"])
        ra_str, dec_str = str(row["ra,dec"]).split(",")
        return {
            "grb_name": grb_name,
            "trigger_met": trig_time,
            "ra": float(ra_str),
            "dec": float(dec_str),
        }
    except KeyError:
        pass
    except Exception as exc:  # noqa: BLE001
        log(f"提示: fermilat-grb.xls 读取失败（{exc}），改查联合目标交叉表")

    csv_path = getattr(session, "joint_target_csv", None)
    if not csv_path:
        return None
    try:
        table = pd.read_csv(csv_path)
    except Exception as exc:  # noqa: BLE001
        log(f"警告: 联合目标交叉表读取失败: {exc}")
        return None
    rows = table[table["bnname"].astype(str).str.strip().str.lower() == str(bnname).strip().lower()]
    if rows.empty:
        return None
    r = rows.iloc[0]
    met = pd.to_numeric(r.get("trigger_met"), errors="coerce")
    ra = pd.to_numeric(r.get("ra"), errors="coerce")
    dec = pd.to_numeric(r.get("dec"), errors="coerce")
    if pd.isna(met) or pd.isna(ra) or pd.isna(dec):
        return None
    return {
        "grb_name": str(r["grb_name"]).strip(),
        "trigger_met": float(met),
        "ra": float(ra),
        "dec": float(dec),
    }


def process_lat_data(bnname: str, t0: float, t1: float, _result_dir: str) -> Optional[OGIPLike]:
    if my_apps is None:
        log("警告: threeML/gt_apps 不可用，跳过 LAT 处理")
        return None
    try:
        log(f"开始处理LAT数据: {bnname}")
        info = _lookup_lat_target_info(bnname)
        if info is None:
            log(
                f"警告: 未找到 {bnname} 的 LAT 目标信息"
                "（fermilat-grb.xls 与联合目标交叉表均无该暴），跳过 LAT 处理"
            )
            return None
        grb_name = info["grb_name"]
        trig_time = info["trigger_met"]
        ra, dec = info["ra"], info["dec"]
        lat_data_dir = _resolve_lat_data_dir(grb_name)
        if not os.path.exists(lat_data_dir):
            log(f"警告: LAT数据目录不存在: {lat_data_dir}")
            return None
        ft1_file, ft2_file = _find_lat_files(lat_data_dir, bnname)
        if not ft1_file or not ft2_file:
            log(f"警告: 无法找到必要的LAT文件 (ft1: {ft1_file}, ft2: {ft2_file})")
            return None
        result_dir = os.path.abspath(os.path.expanduser(_result_dir))
        os.makedirs(result_dir, exist_ok=True)
        cwd = os.getcwd()
        os.chdir(result_dir)
        try:
            gtselect_outfile = _configure_gtselect_filter(t0=t0, t1=t1, trig_time=trig_time, ra=ra, dec=dec, ft1_file=ft1_file, out_dir=result_dir)
            if not _gtselect_output_has_events(gtselect_outfile):
                log(
                    f"{bnname}: bin [{t0:g}, {t1:g}] 内没有 LAT 事件"
                    "（LAT 辐射窗与分析 bin 不重叠），跳过 LAT 插件"
                )
                return None
            pha_file = _configure_gtbin_counts_map(ft2_file=ft2_file, evfile=gtselect_outfile)
            rsp_file = _generate_lat_response(ft2_file=ft2_file, pha_file=pha_file)
        finally:
            os.chdir(cwd)
        lat_transient = OGIPLike("LAT", observation=os.path.join(result_dir, pha_file), response=os.path.join(result_dir, rsp_file))
        lat_transient.set_active_measurements("100000.-100000000.")
        return lat_transient
    except Exception as exc:  # noqa: BLE001
        log(f"警告: LAT数据处理失败: {exc}")
        traceback.print_exc()
        return None
