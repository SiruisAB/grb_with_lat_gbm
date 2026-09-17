#!/usr/bin/env python3
"""把重建行合并进批次汇总 CSV。

背景：summary_results.csv 由批次在每个目标完成时增量写，因此它只覆盖
"本批跑过的目标"；此前批次已完成、但不在本批 --only 清单里的目标
（例如 2008–2013 那批，以及一轮因 PATH 缺 gtrspgen 而静默降级成纯 GBM
的 2014 目标）永远不会被写进去。reconstruct_summary.py 能从各暴落盘的
产物把这些行补回来。

合并规则：
  * 以 (bnname, model) 为键去重
  * 同时存在的目标**保留批次的行**（批次那行是在正常流程里直接产出的，
    未经重建近似）
  * 只保留批次 CSV 里出现过的列，保证 schema 一致

幂等：重复执行结果相同。写入用临时文件 + os.replace 原子替换。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

KEY = ["bnname", "model"]


def merge(batch_csv: Path, recon_csv: Path, out_csv: Path) -> dict:
    batch = pd.read_csv(batch_csv)
    recon = pd.read_csv(recon_csv)

    # 只保留批次 CSV 的列（顺序也一致），重建行缺的列留空
    cols = list(batch.columns)
    recon = recon.reindex(columns=cols)

    batch_keys = set(map(tuple, batch[KEY].astype(str).values))
    recon_keys = set(map(tuple, recon[KEY].astype(str).values))

    only_batch = batch_keys - recon_keys
    only_recon = recon_keys - batch_keys
    both = batch_keys & recon_keys

    recon_only = recon[
        ~recon[KEY].astype(str).apply(tuple, axis=1).isin(batch_keys)
    ]
    merged = pd.concat([batch, recon_only], ignore_index=True)

    # 排一下序，便于人看
    if "bnname" in merged.columns:
        merged = merged.sort_values(["bnname", "model"], kind="stable").reset_index(drop=True)

    tmp = out_csv.with_name(out_csv.name + ".part")
    merged.to_csv(tmp, index=False)
    os.replace(tmp, out_csv)

    return {
        "batch_rows": len(batch),
        "recon_rows": len(recon),
        "batch_bursts": batch["bnname"].nunique() if "bnname" in batch else 0,
        "recon_bursts": recon["bnname"].nunique() if "bnname" in recon else 0,
        "only_batch": len(only_batch),
        "only_recon": len(only_recon),
        "overlap_kept_batch": len(both),
        "total_rows": len(merged),
        "total_bursts": merged["bnname"].nunique(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="合并重建行到批次汇总 CSV")
    ap.add_argument("--batch-csv", required=True, help="批次增量写出的 summary CSV")
    ap.add_argument("--recon-csv", required=True, help="reconstruct_summary.py 的输出")
    ap.add_argument("--output", required=True, help="合并结果路径（原地更新时与 batch-csv 相同）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    batch_csv = Path(args.batch_csv)
    recon_csv = Path(args.recon_csv)
    if not batch_csv.exists():
        print(f"批次汇总不存在: {batch_csv}", file=sys.stderr)
        return 1
    if not recon_csv.exists():
        print(f"重建结果不存在: {recon_csv}", file=sys.stderr)
        return 1

    stats = merge(batch_csv, recon_csv, Path(args.output))
    for k, v in stats.items():
        print(f"  {k:20s} {v}")
    if args.dry_run:
        print("(dry-run，未写入)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
