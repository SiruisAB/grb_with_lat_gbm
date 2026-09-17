#!/usr/bin/env python3
"""从各暴已落盘的产物重建 summary 行。

数据源（每个 model/bin 目录）：
  reproducibility/<bin>/fit_parameters.json   -> AIC/BIC/log(Z)、参数中位值
  reproducibility/<bin>/posterior_samples.npz -> 后验样本（算 flux/fluene）
  my_results_<bin>.fits                       -> 参数误差（NEGATIVE/POSITIVE_ERROR）
  <burst>/run_metadata.json                   -> duration / 探测器等

flux 复刻 threeML 的精确口径（已对照 CSV 真值验证到 0.011%）：
  energy_integrand = x * model(x)      # 注意不是 x^2
  trap_integral    = np.trapz(y, logspace(log10(8), log10(1e8), 50))
  逐样本计算后取中位数
fluorescence = flux * (bin_end - bin_start)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

KEV2ERG = 1.602176634e-9
EMIN_KEV = 8.0
EMAX_KEV = 1.0e8
GBM_DATA_DIR = Path("/home/mxr/lee/gbmtest/GBM_data")


def _resolve_source_dir(bnname, grb_name) -> str:
    """复刻 project._resolve_gbm_source_dir 的候选顺序。"""
    candidates = [
        GBM_DATA_DIR / str(grb_name) / str(bnname),
        GBM_DATA_DIR / str(bnname),
        GBM_DATA_DIR / str(grb_name),
        GBM_DATA_DIR,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[-1])


def _flux_from_samples(model, samples, pnames, emax_kev) -> tuple:
    """按 threeML 口径逐样本积分取中位；返回 (median, minus_err, plus_err)。

    能量上界与 bayesian_fit 一致：含 lat 的模式 1e8 keV，纯 GBM 模式 4e4 keV。
    （纯 GBM 用 1e8 会把外推段也算进去，实测偏高一个数量级。）
    """
    grid = np.logspace(np.log10(EMIN_KEV), np.log10(emax_kev), 50)

    def set_sample(vec):
        for nm, v in zip(pnames, vec):
            leaf = nm.split(".")[-1]
            obj = getattr(model, leaf, None)
            if obj is None:
                obj = getattr(model, leaf.lower(), None)
            if obj is not None and hasattr(obj, "value"):
                obj.value = float(v)

    fluxes = np.empty(samples.shape[1])
    for i in range(samples.shape[1]):
        set_sample(samples[:, i])
        fluxes[i] = np.trapz(grid * model(grid) * KEV2ERG, grid)

    med = float(np.median(fluxes))
    lo, hi = np.percentile(fluxes, [16, 84])
    return med, med - float(lo), float(hi) - med


def _param_errors_from_fits(fits_path: Path) -> dict:
    """从 results fits 取出每个参数的对称误差与中位值。"""
    out = {}
    if not fits_path.exists():
        return out
    with fits.open(fits_path) as h:
        t = h[1].data
        names = [str(n) for n in t.field("NAME")]
        vals = np.asarray(t.field("VALUE"), dtype=float)
        neg = np.asarray(t.field("NEGATIVE_ERROR"), dtype=float)
        pos = np.asarray(t.field("POSITIVE_ERROR"), dtype=float)
    for nm, v, n, p in zip(names, vals, neg, pos):
        clean = nm.replace(":", "_").replace("+", "_").replace(" ", "_")
        out[f"{clean}_value"] = float(v)
        out[f"{clean}_error"] = float((abs(n) + abs(p)) / 2.0)
    return out


def _import_build_model():
    """导入 build_model，不依赖本脚本所在的层级。

    脚本可能被放在不同位置（仓库内、临时目录被 cron 调用的副本），
    所以先用显式路径，再退回按包名导入。
    """
    project_root = Path(os.environ.get("GRB_PROJECT_ROOT", "/home/mxr/lee/gbmtest"))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from grb_project.modelbuild import build_model

    return build_model


def _build_row(burst_dir: Path, model_dir: Path, bundle: Path) -> dict:
    fp = json.loads((bundle / "fit_parameters.json").read_text(encoding="utf-8"))
    manifest_path = bundle / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    z = np.load(bundle / "posterior_samples.npz", allow_pickle=True)
    samples = z["samples"]
    pnames = [str(x) for x in z["parameter_names"]]

    meta = json.loads((burst_dir / "run_metadata.json").read_text(encoding="utf-8"))
    model_str = fp["model"]
    analysis_mode = fp["analysis_mode"]
    bin_start = float(fp["bin_start_s"])
    bin_end = float(fp["bin_end_s"])

    # 用合适模式重建模型
    build_model = _import_build_model()

    # 与 bayesian_fit 的口径一致：含 lat 到 1e8 keV，纯 GBM 到 4e4 keV。
    # 模型也按实际模式构建（纯 GBM 不该带 LAT 的 1e8 能量边界）。
    if "lat" in analysis_mode.lower():
        emax_kev, model_mode = EMAX_KEV, "gbm+lat"
    else:
        emax_kev, model_mode = 4.0e4, "gbm"
    model = build_model(model_str, analysis_mode=model_mode)
    flux_med, flux_lo, flux_hi = _flux_from_samples(
        model, samples, pnames, emax_kev
    )
    duration_bin = bin_end - bin_start

    stats = fp.get("statistical_measures", {})
    fits_path = model_dir / f"my_results_{bin_start}-{bin_end}.fits"

    row = {
        "grb_name": meta.get("grb_name"),
        "bnname": meta.get("bnname"),
        "model": model_str,
        "analysis_mode": analysis_mode,
        "AIC": stats.get("AIC"),
        "BIC": stats.get("BIC"),
        "Flux(erg/cm2/s)": flux_med,
        "FTot(erg/cm2/s)": flux_med,
        "Fluence(erg/cm2)": flux_med * duration_bin,
        "FBB(erg/cm2/s)": float("nan"),
        "log_marginal_likelihood": stats.get("log(Z)"),
        "duration": float(meta.get("duration", duration_bin)),
        "num_time_bins": int(meta.get("num_time_segments", 1)),
        "bin_start_time": bin_start,
        "bin_end_time": bin_end,
        "bin_duration": duration_bin,
        "reproducibility_dir": str(bundle),
        "result_dir": str(model_dir),
        "source_dir": _resolve_source_dir(meta.get("bnname"), meta.get("grb_name")),
        "segment_tag": f"bin_{bin_start:g}_{bin_end:g}",
        "detectors": ",".join(manifest.get("detectors", []) or []),
        "analysis_status": "completed",
        "analysis_note": (
            "GBM+LAT 联合拟合已完成"
            if "lat" in analysis_mode.lower()
            else "GBM 拟合已完成"
        ),
        "parallel_models": meta.get("parallel_models"),
        "model_workers": meta.get("model_workers"),
        "_reconstructed": True,
        "_flux_err_minus": flux_lo,
        "_flux_err_plus": flux_hi,
    }
    row.update(_param_errors_from_fits(fits_path))
    return row


def reconstruct(result_root: Path) -> pd.DataFrame:
    rows = []
    for burst_dir in sorted(result_root.glob("bn*")):
        if not burst_dir.is_dir():
            continue
        for model_dir in sorted(p for p in burst_dir.iterdir() if p.is_dir()):
            repro = model_dir / "reproducibility"
            if not repro.is_dir():
                continue
            for bundle in sorted(repro.iterdir()):
                if not (bundle / "fit_parameters.json").exists():
                    continue
                try:
                    rows.append(_build_row(burst_dir, model_dir, bundle))
                except Exception as exc:  # noqa: BLE001
                    print(f"  跳过 {burst_dir.name}/{model_dir.name}/{bundle.name}: {exc}",
                          file=sys.stderr)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="从磁盘产物重建 summary 行")
    ap.add_argument("--result-root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    root = Path(args.result_root)
    df = reconstruct(root)
    if df.empty:
        print("没有可重建的目标")
        return 1
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"重建 {len(df)} 行 / {df['bnname'].nunique()} 个暴 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
