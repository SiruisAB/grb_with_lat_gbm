# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re
import shutil
from typing import List, Tuple

import pandas as pd

from .logging_utils import log
from .session import session


def normalize_lat_trigger_name(value: object) -> str | None:
    match = re.search(r"(\d{9})", str(value).strip())
    return f"bn{match.group(1)}" if match else None


def merge_lat_catalog_frames(historical: pd.DataFrame, gcn: pd.DataFrame) -> pd.DataFrame:
    historical = historical.copy()
    gcn = gcn.copy()

    historical.index = historical.get("name", pd.Series(index=historical.index, dtype=object)).map(
        normalize_lat_trigger_name
    )
    gcn.index = gcn.get("trigname", pd.Series(index=gcn.index, dtype=object)).map(
        normalize_lat_trigger_name
    )

    merged = pd.concat(
        [historical.loc[historical.index.notna()], gcn.loc[gcn.index.notna()]],
        axis=0,
        sort=False,
    )
    return merged.loc[~merged.index.duplicated(keep="last")]


def load_lat_catalog(xls_path: str | None = None) -> pd.DataFrame:
    path = xls_path if xls_path is not None else session.fermilat_grb_xls
    historical = pd.read_excel(path, sheet_name="fermilgrb")
    gcn = pd.read_excel(path, sheet_name="GCN")
    return merge_lat_catalog_frames(historical, gcn)


def read_gcn_bn_triggers(xls_path: str | None = None) -> Tuple[List[str], List[str]]:
    """
    从 fermilat-grb.xls 的 GCN 表读取 GBM 触发名（bn……）。

    返回：
        (有效 trigname 列表（去重保序）, 因缺少 trigname 而跳过的 gcn_name 描述列表)
    """
    path = xls_path if xls_path is not None else session.fermilat_grb_xls
    df = pd.read_excel(path, sheet_name="GCN")
    if "trigname" not in df.columns:
        return [], ["(GCN 表无 trigname 列)"]

    triggers: List[str] = []
    skipped_desc: List[str] = []
    for _, row in df.iterrows():
        v = row["trigname"]
        gcn = row.get("gcn_name", "")
        if pd.isna(v) or str(v).strip() == "" or str(v).strip().lower() == "nan":
            label = str(gcn).strip() if pd.notna(gcn) else "(无 gcn_name)"
            skipped_desc.append(f"缺少 trigname: {label}")
            continue
        s = str(v).strip()
        if not s.startswith("bn"):
            skipped_desc.append(f"非 bn 触发名: {s} ({gcn})")
            continue
        triggers.append(s)

    seen: set[str] = set()
    uniq: List[str] = []
    for t in triggers:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq, skipped_desc


def _append_text_report(result_root: str, filename: str, text: str) -> None:
    """将说明性文本追加写入 result_root 下的文件（用于跳过原因等）。"""
    try:
        path = os.path.join(result_root, filename)
        with open(path, "a", encoding="utf-8") as f:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
    except OSError:
        pass


def _append_exception_log(result_root: str, bn_label: str, tb_text: str) -> None:
    """将异常 traceback 追加到结果目录下的统一日志。"""
    try:
        path = os.path.join(result_root, "grb_exceptions.log")
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 72 + "\n")
            f.write(f"{pd.Timestamp.now()}  {bn_label}\n")
            f.write(tb_text)
            if not tb_text.endswith("\n"):
                f.write("\n")
    except OSError:
        pass


def find_files(path: str, ext: str) -> List[str]:
    """在 path 下递归查找指定扩展名的所有文件。"""
    return [
        os.path.join(root, fname)
        for root, _, files in os.walk(path)
        for fname in files
        if fname.endswith(ext)
    ]


def ensure_dir(path: str) -> str:
    """确保目录存在并返回该路径。"""
    os.makedirs(path, exist_ok=True)
    return path


def _gll_basename_for_extended_lat_copy(fname: str, bn_name: str) -> str | None:
    """
    若 ``fname`` 为 Extended 常见命名 ``L*EV00.(fits|fit)`` / ``L*SC00.(fits|fit)``，返回
    ``lat_extended_three_ml._resolve_ft_paths`` 与 threeML ``make_LAT_dataset`` 所期望的
    ``gll_ft1_tr_{bn}_v00.fit`` / ``gll_ft2_tr_{bn}_v00.fit`` 基名；否则返回 ``None``。
    """
    if not fname.startswith("L"):
        return None
    low = fname.lower()
    if not low.endswith((".fits", ".fit")):
        return None
    if "ev00" in low:
        return f"gll_ft1_tr_{bn_name}_v00.fit"
    if "sc00" in low:
        return f"gll_ft2_tr_{bn_name}_v00.fit"
    return None


def copy_extended_lat_to_bn_dir(
    grb_name: str,
    dest_bn_dir: str,
    extended_root: str | None = None,
) -> bool:
    """
    将 ``Extended_data_ex/{grb_name}/`` 下的 LAT 文件复制到当前 GRB 的 ``{bn_name}`` 结果目录。

    :param grb_name: GCN 名称（与 Extended_data_ex 子目录名一致，如 GRB231129C）
    :param dest_bn_dir: 目标目录，必须为 ``{result_root}/{grb_name}/lat/{bn_name}``
    :param extended_root: 默认为 ``session.extended_lat_data_root``
    :return: 是否执行了复制（源目录存在且至少复制了一个文件则为 True）
    """
    root = (
        extended_root
        if extended_root is not None
        else getattr(session, "extended_lat_data_root", "")
    )
    src = os.path.join(os.path.expanduser(root), grb_name)
    if not os.path.isdir(src):
        log(f"Extended LAT 源目录不存在，跳过复制: {src}")
        return False

    ensure_dir(dest_bn_dir)
    bn_name = os.path.basename(os.path.normpath(dest_bn_dir))
    n = 0
    for name in os.listdir(src):
        sp = os.path.join(src, name)
        dp = os.path.join(dest_bn_dir, name)
        if os.path.isfile(sp):
            shutil.copy2(sp, dp)
            n += 1
            gll_base = _gll_basename_for_extended_lat_copy(name, bn_name)
            if gll_base is not None and gll_base != name:
                new_p = os.path.join(dest_bn_dir, gll_base)
                os.replace(dp, new_p)
                log(f"Extended LAT: {name} → {gll_base}")
        elif os.path.isdir(sp):
            if os.path.exists(dp):
                shutil.rmtree(dp)
            shutil.copytree(sp, dp)
            n += 1

    log(f"已从 Extended 复制 {n} 项到 {dest_bn_dir}")
    return n > 0


def read_catalog() -> pd.DataFrame:
    """读取 GBM 目录表（fermigbrst）。"""
    return pd.read_excel(
        session.catalog_xls,
        sheet_name="fermigbrst",
        index_col="trigger_name",
    )
