# -*- coding: utf-8 -*-
"""工程入口：将 GRBProjectConfig 同步到 session 并调用 pipeline.main。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd

_GBMTEST_DIR = Path(__file__).resolve().parent.parent
if str(_GBMTEST_DIR) not in sys.path:
    sys.path.insert(0, str(_GBMTEST_DIR))

from .config import (
    GRBProjectConfig,
    GRBRunOverrides,
    run_overrides_from_config,
)
from .io_utils import _append_text_report, ensure_dir, read_gcn_bn_triggers
from .session import session, set_result_root


def apply_project_config(cfg: GRBProjectConfig) -> None:
    """把配置写入 session（路径与汇总 CSV）。"""
    session.data_dir = cfg.data_dir
    session.catalog_xls = cfg.catalog_xls
    session.merged_xls = cfg.merged_xls
    session.fermilat_grb_xls = cfg.fermilat_grb_xls
    session.result_root_gcn_batch = cfg.result_root_gcn_batch
    set_result_root(cfg.result_root, cfg.summary_csv_name)
    if cfg.lat_three_ml_full is not None:
        session.lat_extended_three_ml_pipeline = bool(cfg.lat_three_ml_full)


class GRBProject:
    """
    工程化封装：先构造或修改 config（bnname、grbname、t0、t1 等），再 run。

    示例::

        p = GRBProject()
        p.bnname = "bn221009888"
        p.t0 = 0.0
        p.t1 = 50.0
        p.run()
    """

    def __init__(
        self,
        config: Optional[GRBProjectConfig] = None,
        **config_fields: object,
    ) -> None:
        self.config = config or GRBProjectConfig()
        for key, val in config_fields.items():
            if not hasattr(self.config, key):
                raise TypeError(f"GRBProjectConfig 无字段 {key!r}")
            setattr(self.config, key, val)

    @property
    def bnname(self) -> Optional[str]:
        return self.config.bnname

    @bnname.setter
    def bnname(self, value: Optional[str]) -> None:
        self.config.bnname = value

    @property
    def grbname(self) -> Optional[str]:
        return self.config.grbname

    @grbname.setter
    def grbname(self, value: Optional[str]) -> None:
        self.config.grbname = value

    @property
    def t0(self) -> Optional[float]:
        return self.config.t0

    @t0.setter
    def t0(self, value: Optional[float]) -> None:
        self.config.t0 = value

    @property
    def t1(self) -> Optional[float]:
        return self.config.t1

    @t1.setter
    def t1(self, value: Optional[float]) -> None:
        self.config.t1 = value

    @property
    def ra(self) -> Optional[float]:
        return self.config.ra

    @ra.setter
    def ra(self, value: Optional[float]) -> None:
        self.config.ra = value

    @property
    def dec(self) -> Optional[float]:
        return self.config.dec

    @dec.setter
    def dec(self, value: Optional[float]) -> None:
        self.config.dec = value

    def run(
        self,
        target_grbs: Optional[Sequence[str]] = None,
        analysis_mode: Optional[str] = None,
        result_root: Optional[str] = None,
        summary_csv_name: Optional[str] = None,
        session_log: bool = False,
        run_overrides: Optional[GRBRunOverrides] = None,
    ) -> None:
        apply_project_config(self.config)

        mode = analysis_mode if analysis_mode is not None else self.config.analysis_mode

        tg: Optional[Union[Sequence[str], str]]
        if target_grbs is not None:
            tg = target_grbs
        elif self.config.bnname:
            tg = [self.config.bnname]
        else:
            tg = None

        ov = run_overrides if run_overrides is not None else run_overrides_from_config(
            self.config
        )

        from .pipeline import main

        main(
            tg,
            mode,
            result_root=result_root,
            summary_csv_name=summary_csv_name,
            session_log=session_log,
            fixed_num_time_bins=self.config.fixed_num_time_bins,
            run_overrides=ov,
        )

    def run_gcn_batch(
        self,
        result_root: Optional[str] = None,
        summary_csv_name: str = "summary_results3.csv",
        session_log: bool = True,
    ) -> None:
        """等价于命令行 ``--gcn-all``。"""
        apply_project_config(self.config)
        batch_root = result_root or self.config.result_root_gcn_batch
        gcn_bns, skipped_trigname = read_gcn_bn_triggers()
        df_all = pd.read_excel(self.config.catalog_xls, sheet_name="fermigbrst")
        trig_col = "trigger_name" if "trigger_name" in df_all.columns else "bnname"
        existing_bn = set(df_all[trig_col].astype(str))

        resolved = [b for b in gcn_bns if b in existing_bn]
        not_in_cat = [b for b in gcn_bns if b not in existing_bn]

        ensure_dir(batch_root)
        if skipped_trigname:
            lines = "\n".join(skipped_trigname) + "\n"
            _append_text_report(
                batch_root,
                "gcn_skipped_missing_or_invalid_trigname.txt",
                lines,
            )
        if not_in_cat:
            _append_text_report(
                batch_root,
                "gcn_skipped_not_in_fermigbrst.txt",
                "\n".join(not_in_cat) + "\n",
            )

        print(
            f"[LOG] --gcn-all: GCN 中有效 bn {len(gcn_bns)} 个，"
            f"在 fermigbrst 中可运行 {len(resolved)} 个；"
            f"未在目录表中: {len(not_in_cat)}；"
            f"缺少/无效 trigname 行: {len(skipped_trigname)}"
        )

        if not resolved:
            print(
                "[LOG] 错误: 没有可在 GBMcatolog 中匹配的 GCN 触发，已退出。",
                file=sys.stderr,
            )
            sys.exit(1)

        set_result_root(batch_root, summary_csv_name)
        from .pipeline import main

        main(
            resolved,
            self.config.analysis_mode,
            result_root=batch_root,
            summary_csv_name=summary_csv_name,
            session_log=session_log,
            fixed_num_time_bins=self.config.fixed_num_time_bins,
            run_overrides=run_overrides_from_config(self.config),
        )
