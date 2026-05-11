# -*- coding: utf-8 -*-
"""GBM/LAT 联合分析：路径与单次运行参数（可与核心脚本解耦导入）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class GRBProjectConfig:
    """
    工程级默认路径与分析选项；可在实例化后逐项赋值，再调用 GRBProject.run()。

    bnname / grbname / t0 / t1 等设为 None 时表示沿用核心脚本原有逻辑（目录表与 GCN）。
    """

    # --- 数据与目录路径（默认值与原 gbmtest_enhanced1_refactored.py 一致）---
    data_dir: str = "/home/mxr/lee/gbmtest/GBM_data"
    catalog_xls: str = "/home/mxr/lee/data/GBMcatolog.xls"
    merged_xls: str = "/home/mxr/lee/newflietest/merged_grb_fit_results.xlsx"
    fermilat_grb_xls: str = "/home/mxr/lee/fermilat-grb.xls"
    result_root: str = "/home/mxr/lee/gbmtest/results3"
    summary_csv_name: str = "summary_results.csv"
    result_root_gcn_batch: str = "/home/mxr/lee/gbmtest/results3"

    # --- 运行模式 ---
    analysis_mode: str = "gbm+lat"
    fixed_num_time_bins: Optional[int] = None

    # --- 可选：先占位，run 前再赋值 ---
    bnname: Optional[str] = None
    """触发名，如 bn221009888；若设置且未传 target_grbs，则只分析该暴。"""
    grbname: Optional[str] = None
    """GRB 标准名或输出用名称；None 时从 fermilat GCN 表解析。"""
    t0: Optional[float] = None
    t1: Optional[float] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    lat_three_ml_full: Optional[bool] = None


@dataclass
class GRBRunOverrides:
    """传入核心 analyze_grb 的覆盖项；仅非 None 字段生效。

    lat_three_ml_full: 为 True/False 时覆盖 session.lat_extended_three_ml_pipeline，
    控制是否运行完整 LAT Extended / GtBurst / threeML 流水线。
    """

    grbname: Optional[str] = None
    t0: Optional[float] = None
    t1: Optional[float] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    lat_three_ml_full: Optional[bool] = None

    def is_empty(self) -> bool:
        return all(
            getattr(self, k) is None
            for k in (
                "grbname",
                "t0",
                "t1",
                "ra",
                "dec",
                "lat_three_ml_full",
            )
        )


def run_overrides_from_config(cfg: GRBProjectConfig) -> Optional[GRBRunOverrides]:
    """由工程配置构造覆盖对象；若无可覆盖字段则返回 None。"""
    ov = GRBRunOverrides(
        grbname=cfg.grbname,
        t0=cfg.t0,
        t1=cfg.t1,
        ra=cfg.ra,
        dec=cfg.dec,
        lat_three_ml_full=cfg.lat_three_ml_full,
    )
    return None if ov.is_empty() else ov
