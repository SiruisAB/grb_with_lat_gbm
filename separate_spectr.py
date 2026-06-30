from __future__ import annotations

from pyexpat import model
from matplotlib import rcParams

rcParams["font.family"] = "Sans-serif"

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

try:
    import spectres
except ImportError:  # pragma: no cover - optional dependency in redraw-only environments
    spectres = None

try:
    from astropy import units as u
except ImportError:  # pragma: no cover - optional dependency in redraw-only environments
    u = None

try:
    from astromodels import (
        Band,
        Blackbody,
        Cutoff_powerlaw,
        Gaussian,
        Log_uniform_prior,
        Model,
        PointSource,
        Powerlaw,
        Uniform_prior,
        NonDissipativePhotosphere,
    )
except ImportError:  # pragma: no cover - optional dependency in redraw-only environments
    Band = Blackbody = Cutoff_powerlaw = Gaussian = Log_uniform_prior = Model = PointSource = Powerlaw = Uniform_prior = NonDissipativePhotosphere = None

try:
    from threeML import *  # noqa: F401,F403
except ImportError:  # pragma: no cover - optional dependency in redraw-only environments
    pass

try:
    from .modelbuild import MultiColorBlackBody, build_model
except ImportError:  # pragma: no cover - optional dependency in redraw-only environments
    MultiColorBlackBody = build_model = None

mpl.rcParams.update(
    {
        "lines.linewidth": 1.8,
        "axes.linewidth": 1.8,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "font.size": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "xtick.major.width": 1.6,
        "ytick.major.width": 1.6,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "legend.fontsize": 12,
        "legend.frameon": False,
        "savefig.dpi": 300,
        "figure.dpi": 120,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _style_publication_axes(ax):
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=False,
        right=False,
        length=6,
        width=1.6,
        labelsize=14,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=False,
        right=False,
        length=3,
        width=1.2,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.8)
    ax.spines["bottom"].set_linewidth(1.8)


_FLUXDATA_RE = re.compile(
    r"^band\+bb_(nai_[^_]+|bgo_[^_]+|lat)_data_point_(.+)\.txt$"
)

_JSON_TIMEBIN_RE = re.compile(r"^bn[^_]+_bin_(?P<start>[0-9.]+)_(?P<end>[0-9.]+)$")


def discover_fluxdata_groups(fluxdata_dir: Path) -> dict[str, dict[str, Path]]:
    groups: dict[str, dict[str, Path]] = {}
    for path in sorted(Path(fluxdata_dir).glob("*.txt")):
        match = _FLUXDATA_RE.match(path.name)
        if not match:
            continue
        detector_tag, timebin = match.groups()
        groups.setdefault(timebin, {})[detector_tag] = path
    return groups


def _read_fluxdata_file(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = np.atleast_2d(data)
    return data


def _normalize_timebin(timebin: str) -> str:
    text = str(timebin).strip()
    if "-" not in text:
        return text
    start_s, end_s = text.split("-", 1)
    return f"{float(start_s):g}-{float(end_s):g}"


def _find_json_bin_entry(json_payload: dict, bnname: str, timebin: str) -> tuple[str, dict]:
    burst_root = json_payload.get(bnname, {})
    bins = burst_root.get("bins", {})
    timebin_norm = _normalize_timebin(timebin)
    for bin_key, entry in bins.items():
        meta = entry.get("meta", {})
        if meta.get("time_bin_identifier") == timebin_norm:
            return bin_key, entry
        match = _JSON_TIMEBIN_RE.match(bin_key)
        if match:
            start = float(match.group("start"))
            end = float(match.group("end"))
            if _normalize_timebin(f"{start}-{end}") == timebin_norm:
                return bin_key, entry
            if f"{start:.2f}-{end:.2f}" == timebin_norm:
                return bin_key, entry
        if meta.get("bin_start_time") is not None and meta.get("bin_end_time") is not None:
            cand = _normalize_timebin(f"{float(meta['bin_start_time'])}-{float(meta['bin_end_time'])}")
            if cand == timebin_norm:
                return bin_key, entry
    raise KeyError(f"No fit entry found for {bnname} {timebin}")


def load_bandbb_fit_params(json_path: Path, bnname: str, timebin: str) -> dict[str, float]:
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    _, entry = _find_json_bin_entry(payload, bnname, timebin)
    bandbb = entry.get("band+bb")
    if not bandbb:
        raise KeyError(f"No band+bb fit found for {bnname} {timebin}")
    return {
        "K_1": float(bandbb["GRB.spectrum.main.composite.K_1_value"]),
        "alpha_1": float(bandbb["GRB.spectrum.main.composite.alpha_1_value"]),
        "xp_1": float(bandbb["GRB.spectrum.main.composite.xp_1_value"]),
        "beta_1": float(bandbb["GRB.spectrum.main.composite.beta_1_value"]),
        "K_2": float(bandbb["GRB.spectrum.main.composite.K_2_value"]),
        "kT_2": float(bandbb["GRB.spectrum.main.composite.kT_2_value"]),
    }


def _plot_fluxdata_detector(ax, path: Path, label: str, marker: str, color: str) -> None:
    data = _read_fluxdata_file(path)
    x = data[:, 0]
    xerr = np.vstack((data[:, 1], data[:, 2]))
    y = data[:, 3]
    yerr = np.vstack((data[:, 4], data[:, 5]))
    ax.errorbar(
        x,
        y,
        xerr=np.abs(xerr),
        yerr=np.abs(yerr),
        marker=marker,
        label=label,
        fmt="none",
        ms=5,
        color=color,
        elinewidth=1.2,
        capthick=1.2,
        capsize=2,
    )


def _format_timebin_title(bnname: str, timebin: str) -> str:
    return f"{bnname}_spectra_{timebin}"


def compute_bandbb_curves(
    xs: np.ndarray,
    K_1: float,
    alpha_1: float,
    xp_1: float,
    beta_1: float,
    K_2: float,
    kT_2: float,
) -> dict[str, np.ndarray]:
    xs = np.asarray(xs, dtype=float)
    band = np.empty_like(xs)
    ebreak = (alpha_1 - beta_1) * xp_1 / (2.0 + alpha_1)
    band_low = xs <= ebreak
    band_high = ~band_low
    band[band_low] = K_1 * (xs[band_low] / 100.0) ** alpha_1 * np.exp(-xs[band_low] * (2.0 + alpha_1) / xp_1)
    band[band_high] = (
        K_1
        * ((alpha_1 - beta_1) * xp_1 / (100.0 * (2.0 + alpha_1))) ** (alpha_1 - beta_1)
        * np.exp(beta_1 - alpha_1)
        * (xs[band_high] / 100.0) ** beta_1
    )
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        bb = K_2 * xs**2 / np.expm1(xs / kT_2)
    bb = np.where(np.isfinite(bb), bb, 0.0)
    total = band + bb
    return {"total": total, "band": band, "bb": bb}


def _overlay_fit_curves(
    ax,
    timebin: str,
    bnname: str,
    fit_json_path: Path | None,
) -> None:
    if fit_json_path is None:
        return
    params = load_bandbb_fit_params(fit_json_path, bnname=bnname, timebin=timebin)
    xs = np.logspace(np.log10(8.0), np.log10(1e5), 400)
    curves = compute_bandbb_curves(xs, **params)
    ax.plot(xs, curves["total"], color="#1f77b4", linewidth=2.2, label="Band+BB total fit")
    ax.plot(xs, curves["band"], color="#d62728", linestyle=":", linewidth=1.8, label="Band fit")
    ax.plot(xs, curves["bb"], color="#2ca02c", linestyle="--", linewidth=1.8, label="BB fit")
    param_text = (
        f"K1={params['K_1']:.3g}\n"
        f"alpha={params['alpha_1']:.3g}\n"
        f"xp={params['xp_1']:.3g}\n"
        f"beta={params['beta_1']:.3g}\n"
        f"K2={params['K_2']:.3g}\n"
        f"kT={params['kT_2']:.3g}"
    )
    ax.text(
        0.98,
        0.98,
        param_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75, edgecolor="#999999"),
    )


def redraw_fluxdata_timebin(
    timebin: str,
    fluxdata_dir: Path,
    output_dir: Path,
    bnname: str,
    fit_json_path: Path | None = None,
) -> Path:
    groups = discover_fluxdata_groups(fluxdata_dir)
    if timebin not in groups:
        raise FileNotFoundError(f"No fluxdata files found for time bin {timebin}")

    detector_files = groups[timebin]
    fig, ax = plt.subplots(figsize=(12, 8))

    detector_specs = [
        ("nai_n3", "NaI (n3)", "+", "#4d4d4d"),
        ("nai_n7", "NaI (n7)", "o", "#4d4d4d"),
        ("bgo_b0", "BGO (b0)", "s", "#4d4d4d"),
        ("lat", "LAT", "v", "#4d4d4d"),
    ]
    plotted_any = False
    for detector_tag, label, marker, color in detector_specs:
        path = detector_files.get(detector_tag)
        if path is None:
            continue
        _plot_fluxdata_detector(ax, path, label=label, marker=marker, color=color)
        plotted_any = True

    if not plotted_any:
        raise FileNotFoundError(f"No recognized detector files found for time bin {timebin}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$E$ [keV]")
    ax.set_ylabel("$E^{2} dN/dE$ [erg s$^{-1}$cm$^{-2}$]")
    _overlay_fit_curves(ax, timebin=timebin, bnname=bnname, fit_json_path=fit_json_path)
    ax.set_title(_format_timebin_title(bnname, timebin))
    ax.minorticks_on()
    _style_publication_axes(ax)
    ax.legend(loc=2)
    plt.tight_layout()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"bs_{bnname}_gbm_lat_spectra_band+bb_{timebin}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def redraw_fluxdata_overview(
    groups: dict[str, dict[str, Path]],
    fluxdata_dir: Path,
    output_dir: Path,
    bnname: str,
    fit_json_path: Path | None = None,
) -> Path:
    timebins = list(groups)
    n_plots = len(timebins)
    if n_plots == 0:
        raise FileNotFoundError(f"No fluxdata groups found in {fluxdata_dir}")

    ncols = 2 if n_plots > 1 else 1
    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12 * ncols, 6 * nrows), squeeze=False)

    for idx, timebin in enumerate(timebins):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        detector_files = groups[timebin]
        for detector_tag, label, marker, color in [
            ("nai_n3", "NaI (n3)", "+", "#4d4d4d"),
            ("nai_n7", "NaI (n7)", "o", "#4d4d4d"),
            ("bgo_b0", "BGO (b0)", "s", "#4d4d4d"),
            ("lat", "LAT", "v", "#4d4d4d"),
        ]:
            path = detector_files.get(detector_tag)
            if path is None:
                continue
            _plot_fluxdata_detector(ax, path, label=label, marker=marker, color=color)
        _overlay_fit_curves(ax, timebin=timebin, bnname=bnname, fit_json_path=fit_json_path)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(timebin)
        ax.minorticks_on()
        _style_publication_axes(ax)
        if row == nrows - 1:
            ax.set_xlabel("$E$ [keV]")
        if col == 0:
            ax.set_ylabel("$E^{2} dN/dE$ [erg s$^{-1}$cm$^{-2}$]")
        ax.legend(loc=2, fontsize=10)

    for idx in range(n_plots, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].axis("off")

    fig.suptitle(f"{bnname} fluxdata overview", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"bs_{bnname}_gbm_lat_spectra_band+bb_overview.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def redraw_all_fluxdata_spectra(
    fluxdata_dir: Path,
    output_dir: Path,
    bnname: str,
    fit_json_path: Path | None = None,
) -> dict[str, list[Path]]:
    groups = discover_fluxdata_groups(fluxdata_dir)
    single_plots = [
        redraw_fluxdata_timebin(timebin, fluxdata_dir, output_dir, bnname, fit_json_path=fit_json_path)
        for timebin in groups
    ]
    overview = redraw_fluxdata_overview(groups, fluxdata_dir, output_dir, bnname, fit_json_path=fit_json_path)
    return {"single_plots": single_plots, "overview": [overview]}


def redraw_bandbb_fluxdata_outputs(
    bnname: str,
    result_root: Path,
    fit_json_path: Path | None = None,
) -> dict[str, list[Path]]:
    fluxdata_dir = Path(result_root) / bnname / "band+bb" / "fluxdata"
    return redraw_all_fluxdata_spectra(
        fluxdata_dir=fluxdata_dir,
        output_dir=fluxdata_dir,
        bnname=bnname,
        fit_json_path=fit_json_path,
    )


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Redraw band+bb spectra from saved fluxdata files."
    )
    parser.add_argument("--bnname", required=True, help="Burst name, e.g. GRB231129C")
    parser.add_argument(
        "--result-root",
        required=True,
        type=Path,
        help="Root directory containing <bnname>/band+bb/fluxdata",
    )
    parser.add_argument(
        "--fluxdata-dir",
        type=Path,
        default=None,
        help="Optional explicit fluxdata directory. Overrides --result-root when set.",
    )
    parser.add_argument(
        "--save-result-dir",
        type=Path,
        default=None,
        help="Optional directory for PDF output. Defaults to the fluxdata directory.",
    )
    parser.add_argument(
        "--timebin",
        default=None,
        help="Redraw only one time bin, e.g. 0.1-1.",
    )
    parser.add_argument(
        "--fit-json",
        type=Path,
        default=None,
        help="Optional JSON file with best-fit band+bb parameters.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional path to save a log file.",
    )
    return parser


def _resolve_fluxdata_dir(
    result_root: Path,
    bnname: str,
    fluxdata_dir: Optional[Path],
) -> Path:
    return fluxdata_dir if fluxdata_dir is not None else Path(result_root) / bnname / "band+bb" / "fluxdata"


def _resolve_output_dir(fluxdata_dir: Path, save_result_dir: Optional[Path]) -> Path:
    return save_result_dir if save_result_dir is not None else fluxdata_dir


def _setup_logger(log_file: Optional[Path]) -> logging.Logger:
    logger = logging.getLogger("grb_project.separate_spectr.redraw")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        logger.addHandler(file_handler)

    return logger


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)
    fluxdata_dir = _resolve_fluxdata_dir(args.result_root, args.bnname, args.fluxdata_dir)
    output_dir = _resolve_output_dir(fluxdata_dir, args.save_result_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = args.log_file or (output_dir / "fluxdata_redraw.log")
    logger = _setup_logger(log_file)
    logger.info("Fluxdata directory: %s", fluxdata_dir)
    logger.info("Output directory: %s", output_dir)
    logger.info("Fit JSON: %s", args.fit_json if args.fit_json else "<none>")
    logger.info("Log file: %s", log_file)

    if args.timebin:
        logger.info("Redrawing time bin: %s", args.timebin)
        out_path = redraw_fluxdata_timebin(
            timebin=args.timebin,
            fluxdata_dir=fluxdata_dir,
            output_dir=output_dir,
            bnname=args.bnname,
            fit_json_path=args.fit_json,
        )
        logger.info("Saved: %s", out_path)
    else:
        logger.info("Redrawing all fluxdata spectra")
        outputs = redraw_all_fluxdata_spectra(
            fluxdata_dir=fluxdata_dir,
            output_dir=output_dir,
            bnname=args.bnname,
            fit_json_path=args.fit_json,
        )
        for path in outputs["single_plots"]:
            logger.info("Saved: %s", path)
        for path in outputs["overview"]:
            logger.info("Saved: %s", path)

    logger.info("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def discrete_spectr(
    fluence_plugins,
    lat,
    model_str,
    parameter_values,
    bs,
    result_dir,
    bnname,
    gbm_detectors,
    bin_start=None,
    bin_end=None,
    analysis_mode: str = "gbm+lat",
    output_dir: Optional[str] = None,
):
    # log(f"{bnname}: 探测器 {gbm_detectors}")
    # if model_str  in ['pl','band','blackbody','comp','SBPL']:
    #     model1=build_model(model_str, parameter_values)
    #     modelTotal=model1
    #     model_str1=model_str
    if model_str == 'pl':
        model1 = Powerlaw(piv=1E2)
        # model1.K.min_value, model1.K.max_value = 1e-8, 1e3
        # model1.index.min_value, model1.index.max_value = -5.0, 0.0
        model1.K ,model1.index = parameter_values[:2]
        modelTotal=model1
        model_str1='PL'

    if model_str == 'blackbody':
        model1 = Blackbody()
        model1.K ,model1.kT = parameter_values[:2]
        modelTotal=model1
        model_str1= 'BB'

    if model_str == 'comp':
        model1 = Cutoff_powerlaw(piv=1E5)
        model1.K.min_value, model1.K.max_value = 1e-7, 1e6
        model1.index.min_value, model1.index.max_value = -10.0, 10.0
        model1.xc.min_value, model1.xc.max_value = 1e-99, 1e7
        model1.K ,model1.index,model1.xc = parameter_values[:3]
        modelTotal=model1
        model_str1='comp'

    if  model_str == 'band':
        model1 = Band(piv=1E2)
        # model1.alpha.min_value = -2.0    # 改硬边界
        # model1.alpha.max_value =  5.0
        # model1.xp.min_value,model1.xp.max_value = 1.0, 1e8
        model1.K ,model1.alpha,model1.xp ,model1.beta = parameter_values[:4]
        modelTotal=model1
        model_str1='Band'

    if model_str == 'comp+pl':

        model1 = Cutoff_powerlaw(piv=1E2)
        # model1.K.min_value, model1.K.max_value = 1e-7, 1e2
        # model1.index.min_value, model1.index.max_value = -5.0, 0.0
        # model1.xc.min_value, model1.xc.max_value = 1, 1e4
        model1.K,model1.index,model1.xc = parameter_values[:3]

        model2 = Powerlaw(piv=1E2)
        # model2.K.min_value, model2.K.max_value = 1e-7, 1e2
        # model2.index.min_value, model2.index.max_value = -5.0, 0.0
        model2.K, model2.index = parameter_values[3:5]
        modelTotal=model1+model2
        model_str1='comp'
        model_str2='PL'

    if model_str == 'NDP':
        model1 = NonDissipativePhotosphere(piv=1E2)
        model1.k, model1.ec = parameter_values[:2]
        modelTotal=model1
        model_str1='NDP'

    if model_str == 'mbb':
        # MBB: K, kT_min, kT_max, m（与 modelbuild.py 一致）
        model1 = MultiColorBlackBody()
        model1.K, model1.kT_min, model1.kT_max, model1.m = parameter_values[:4]
        modelTotal = model1
        model_str1 = 'mBB'

    if model_str == 'mbb+pl':
        # mBB + PL: K, kT_min, kT_max, m, K_pl, index（与 modelbuild.py 一致）
        model1 = MultiColorBlackBody()
        model1.K, model1.kT_min, model1.kT_max, model1.m = parameter_values[:4]
        model2 = Powerlaw(piv=1E2)
        model2.K, model2.index = parameter_values[4:6]
        modelTotal = model1 + model2
        model_str1 = 'mBB'
        model_str2 = 'PL'

    if model_str == 'band+bb':
        model1 = Band(K=parameter_values[0],alpha=parameter_values[1],xp=parameter_values[2],beta=parameter_values[3],piv=1E2)          # 先用默认值构造出来
        # model1.alpha.min_value = -5.0    # 改硬边界
        # model1.alpha.max_value =  3.0
        model1.xp.min_value,model1.xp.max_value = 1.0, 1e8
        # model1.k,model1.alpha,model1.xp,model1.beta  = parameter_values[:4]    

        model2 = Blackbody(K=parameter_values[4],kT=parameter_values[5])           # 返回 Blackbody 实例
        # model2.K.min_value, model2.K.max_value = 1e-9, 1e1
        # model2.kT.min_value, model2.kT.max_value = 0, 1e6
        # model2.k,model2.kT = parameter_values[4],parameter_values[5]
        modelTotal = model1 + model2
        model_str1 = 'Band'
        model_str2 = 'BB'

    if model_str == 'band+pl':
        model1 = Band(piv=1E2)                  # 先用默认值构造出来
        model1.alpha.min_value = -2.0    # 改硬边界
        model1.alpha.max_value =  5.0
        # model1.xp.min_value,model1.xp.max_value = 1.0, 1e8
        model1.K , model1.alpha , model1.xp , model1.beta  = parameter_values[:4]
        model2=Powerlaw(piv=1E2)
        # model2.K.min_value, model2.K.max_value = 1e-7, 1e6
        # model2.index.min_value, model2.index.max_value = -5.0, 5.0
        model2.K,model2.index = parameter_values[4],parameter_values[5]
        modelTotal=model1+model2
        model_str1='Band'
        model_str2='PL'
    
    if model_str == 'comp+bb':
        model1 = Cutoff_powerlaw(piv=1E2)
        # model1.K.min_value, model1.K.max_value = 1e-7, 1e2
        # model1.index.min_value, model1.index.max_value = -4.0, 0.0
        # model1.xc.min_value, model1.xc.max_value = 1.0, 1e4
        model1.K ,model1.index,model1.xc = parameter_values[:3]
        model2 = Blackbody()
        # model2.K.min_value, model2.K.max_value = 1e-6, 1
        # model2.kT.min_value, model2.kT.max_value = 0, 1e3
        model2.K ,model2.kT = parameter_values[3:5]
        modelTotal=model1+model2
        model_str1='comp'
        model_str2='BB'

    if model_str == 'pl+bb':
        model1=Blackbody(K=parameter_values[0],kT=parameter_values[1])
        model2=Powerlaw(K=parameter_values[2],index=parameter_values[3],piv=1E2)
        modelTotal=model1+model2
        model_str1='BB'
        model_str2='PL'

    if model_str == 'band+bb+pl':
        model1 = Band(piv=1E2)
        model1.K , model1.alpha , model1.xp , model1.beta  = parameter_values[:4]
        model2 = Blackbody()
        model2.K ,model2.kT = parameter_values[4:6]
        model3 = Powerlaw(piv=1E2)
        model3.K,model3.index = parameter_values[6:8]
        modelTotal=model1+model2+model3
        model_str1='Band'
        model_str2='BB'
        model_str3='PL'
    
    if model_str == 'band+gauss':
        model1 = Band(piv=1E2)
        model1.K , model1.alpha , model1.xp , model1.beta  = parameter_values[:4]
        model2 = Gaussian()
        model2.F, model2.mu , model2.sigma = parameter_values[4:7]
        modelTotal=model1+model2
        model_str1='Band'
        model_str2='Gauss'

    conversion = u.Unit("keV2/(cm2 s keV)").to("erg/(cm2 s)")

    
    k0 = 1.602e-9                  ## convert keV to erg

    def _fmt_time(t):
        try:
            return f"{float(t):.2f}".rstrip("0").rstrip(".")
        except Exception:
            return str(t)

    time_bin_tag = (
        f"{_fmt_time(bin_start)}-{_fmt_time(bin_end)}"
        if bin_start is not None and bin_end is not None
        else None
    )
    time_bin_suffix = f"_{time_bin_tag}" if time_bin_tag else ""

    if output_dir is None:
        output_dir = result_dir
    os.makedirs(output_dir, exist_ok=True)

    figModel = plt.figure(figsize=(12, 8))
    # threeML plot_spectra(bs.results, ...) 已暂时关闭；若要与库自带 folded/总谱叠图，取消下面注释即可。
    # # 修复：bs可能是JointLikelihood或BayesianAnalysis，需要统一处理
    # try:
    #     if hasattr(bs, "results"):
    #         # 根据是否包含 LAT，调整谱图最高能量
    #         ene_max = 100 * u.GeV if "lat" in analysis_mode.lower() else 100 * u.MeV
    #         fig = plot_spectra(
    #             bs.results,
    #             ene_min=1 * u.keV,
    #             ene_max=ene_max,
    #             flux_unit="erg/(cm2 s)",
    #             use_components=True,
    #             components_to_use="total",
    #         )
    #     else:
    #         # 如果没有results属性，创建一个空图
    #         fig = plt.figure(figsize=(18, 12))
    # except Exception:
    #     # 如果plot_spectra失败，创建一个空图
    #     fig = plt.figure(figsize=(18, 12))
                            ################## First NaI data  ###################
    nai_index = 0
    # print '#### plot First NaI data     ####'
    area = fluence_plugins[nai_index].expected_model_rate/modelTotal(fluence_plugins[nai_index]._observed_spectrum.mid_points) 
    dataOrigin = fluence_plugins[nai_index].source_rate/area   # source_rate它是从观测到的总计数率中减去背景计数率后的结果。
    #  dataOrigin当前模型校准后的“观测光子通量谱” (Model-dependent Unfolded Photon Flux)
    dataOriginErr = fluence_plugins[nai_index].source_rate_error/area

    # select data x-range
    wavelength = fluence_plugins[nai_index]._observed_spectrum.mid_points
    rebinnedWavelength = np.logspace(np.log10(wavelength[6]),np.log10(wavelength[-6]),21)

    # rebin x-axis
    rebinnedWavelengthFull = np.insert(rebinnedWavelength, 0, 8)
    rebinnedWavelengthFull = np.append(rebinnedWavelengthFull, 900)
    rebinnedWavelengthErrPos = (rebinnedWavelengthFull[2:] - rebinnedWavelengthFull[1:-1])/2
    rebinnedWavelengthErrNeg = (rebinnedWavelengthFull[1:-1] - rebinnedWavelengthFull[:-2])/2
    # rebin y-axis
    rebinned = spectres.spectres(rebinnedWavelength, wavelength, dataOrigin, spec_errs=dataOriginErr)
    rebinnedFlux = rebinned[0]
    rebinnedFluxErrPos = rebinned[1]
    rebinnedFluxErrNeg = rebinned[1]
    rebinnedFluxErrNeg[rebinnedFluxErrNeg > rebinnedFlux] = rebinnedFlux[rebinnedFluxErrNeg > rebinnedFlux]*0.95
    #  如果误差箱比流量还大，强行把下端改成 95 % 流量，避免画成负条
    #rebinnedWavelength[-1]=0


    plt.errorbar(rebinnedWavelength,k0*rebinnedWavelength*rebinnedWavelength*rebinnedFlux,\
                xerr=np.abs([rebinnedWavelengthErrNeg,rebinnedWavelengthErrPos]), \
                yerr=np.abs([k0*rebinnedWavelength*rebinnedWavelength*rebinnedFluxErrNeg,k0*rebinnedWavelength*rebinnedWavelength*rebinnedFluxErrPos]),marker='+',
                        label='NaI ('+gbm_detectors[nai_index]+')',
                        fmt='.',ms=5,color='#4d4d4d',elinewidth=1.2,capthick=1.2,capsize=2)

    text_file = os.path.join(
        output_dir,
        model_str
        + "_nai_"
        + gbm_detectors[nai_index]
        + "_data_point"
        + time_bin_suffix
        + ".txt",
    )
    data = np.column_stack((rebinnedWavelength, rebinnedWavelengthErrNeg,rebinnedWavelengthErrPos,k0*rebinnedWavelength*rebinnedWavelength*rebinnedFlux,\
    k0*rebinnedWavelength*rebinnedWavelength*rebinnedFluxErrNeg,k0*rebinnedWavelength*rebinnedWavelength*rebinnedFluxErrPos))
    np.savetxt(text_file, data, delimiter=' ')

                            ################## Second NaI data  ###################
    nai_index = 1
    # print '#### plot Second NaI data     ####'
    area = fluence_plugins[nai_index].expected_model_rate/modelTotal(fluence_plugins[nai_index]._observed_spectrum.mid_points)
    dataOrigin = fluence_plugins[nai_index].source_rate/area
    dataOriginErr = fluence_plugins[nai_index].source_rate_error/area

    # select data x-range
    wavelength = fluence_plugins[nai_index]._observed_spectrum.mid_points
    rebinnedWavelength = np.logspace(np.log10(wavelength[6]),np.log10(wavelength[-6]),21)

    # rebin x-axis
    rebinnedWavelengthFull = np.insert(rebinnedWavelength, 0, 8)
    rebinnedWavelengthFull = np.append(rebinnedWavelengthFull, 900)
    rebinnedWavelengthErrPos = (rebinnedWavelengthFull[2:] - rebinnedWavelengthFull[1:-1])/2
    rebinnedWavelengthErrNeg = (rebinnedWavelengthFull[1:-1] - rebinnedWavelengthFull[:-2])/2
    # rebin y-axis
    rebinned = spectres.spectres(rebinnedWavelength, wavelength, dataOrigin, spec_errs=dataOriginErr)
    rebinnedFlux = rebinned[0]
    rebinnedFluxErrPos = rebinned[1]
    rebinnedFluxErrNeg = rebinned[1]
    rebinnedFluxErrNeg[rebinnedFluxErrNeg > rebinnedFlux] = rebinnedFlux[rebinnedFluxErrNeg > rebinnedFlux]*0.95

    #rebinnedWavelength[-1]=0

    plt.errorbar(rebinnedWavelength,k0*rebinnedWavelength*rebinnedWavelength*rebinnedFlux,\
                xerr=np.abs([rebinnedWavelengthErrNeg,rebinnedWavelengthErrPos]), \
                yerr=np.abs([k0*rebinnedWavelength*rebinnedWavelength*rebinnedFluxErrNeg,k0*rebinnedWavelength*rebinnedWavelength*rebinnedFluxErrPos]),marker='o',
                        label='NaI ('+gbm_detectors[nai_index]+')',
                        fmt='.',ms=5,color='#4d4d4d',elinewidth=1.2,capthick=1.2,capsize=2)

    text_file = os.path.join(
        output_dir,
        model_str
        + "_nai_"
        + gbm_detectors[nai_index]
        + "_data_point"
        + time_bin_suffix
        + ".txt",
    )
    data = np.column_stack((rebinnedWavelength, rebinnedWavelengthErrNeg,rebinnedWavelengthErrPos,k0*rebinnedWavelength*rebinnedWavelength*rebinnedFlux,\
    k0*rebinnedWavelength*rebinnedWavelength*rebinnedFluxErrNeg,k0*rebinnedWavelength*rebinnedWavelength*rebinnedFluxErrPos))
    np.savetxt(text_file, data, delimiter=' ')


    #                         ################## Third NaI data  ###################
    # nai_index = 2
    # # print '#### plot Third NaI data     ####'
    # area = fluence_plugins[nai_index].expected_model_rate/modelTotal(fluence_plugins[nai_index]._observed_spectrum.mid_points)
    # dataOrigin = fluence_plugins[nai_index].source_rate/area
    # dataOriginErr = fluence_plugins[nai_index].source_rate_error/area

    # # select data x-range
    # wavelength = fluence_plugins[nai_index]._observed_spectrum.mid_points
    # rebinnedWavelength = np.logspace(np.log10(wavelength[6]),np.log10(wavelength[-6]),15)

    # # rebin x-axis
    # rebinnedWavelengthFull = np.insert(rebinnedWavelength, 0, 8)
    # rebinnedWavelengthFull = np.append(rebinnedWavelengthFull, 900)
    # rebinnedWavelengthErrPos = (rebinnedWavelengthFull[2:] - rebinnedWavelengthFull[1:-1])/2
    # rebinnedWavelengthErrNeg = (rebinnedWavelengthFull[1:-1] - rebinnedWavelengthFull[:-2])/2
    # # rebin y-axis
    # rebinned = spectres.spectres(rebinnedWavelength, wavelength, dataOrigin, spec_errs=dataOriginErr)
    # rebinnedFlux = rebinned[0]
    # rebinnedFluxErrPos = rebinned[1]
    # rebinnedFluxErrNeg = rebinned[1]
    # rebinnedFluxErrNeg[rebinnedFluxErrNeg > rebinnedFlux] = rebinnedFlux[rebinnedFluxErrNeg > rebinnedFlux]*0.95

    # #rebinnedWavelength[-1]=0

    # plt.errorbar(rebinnedWavelength,k0*rebinnedWavelength*rebinnedWavelength*rebinnedFlux,\
    #             xerr=[rebinnedWavelengthErrNeg,rebinnedWavelengthErrPos], \
    #             yerr=[k0*rebinnedWavelength*rebinnedWavelength*rebinnedFluxErrNeg,k0*rebinnedWavelength*rebinnedWavelength*rebinnedFluxErrPos],marker='x',
    #                     # label='NaI ('+gbm_detectors[nai_index]+')',
    #                     fmt='.',ms=10,color='#aaaaaa')

    # # text_file = model_str+'_nai_'+ gbm_detectors[nai_index]+'_data_point.txt'
    # data = np.column_stack((rebinnedWavelength, rebinnedWavelengthErrNeg,rebinnedWavelengthErrPos,k0*rebinnedWavelength*rebinnedWavelength*rebinnedFlux,\
    # k0*rebinnedWavelength*rebinnedWavelength*rebinnedFluxErrNeg,k0*rebinnedWavelength*rebinnedWavelength*rebinnedFluxErrPos))
    # # np.savetxt(text_file, data, delimiter=' ')



                            ################## BGO data  ###################
    # print '#### plot BGO data     ####'

    bgo_index= 2
    area = fluence_plugins[bgo_index].expected_model_rate/modelTotal(fluence_plugins[bgo_index]._observed_spectrum.mid_points)
    dataOrigin = fluence_plugins[bgo_index].source_rate/area
    dataOriginErr = fluence_plugins[bgo_index].source_rate_error/area

    # select data x-range
    wavelength = fluence_plugins[bgo_index]._observed_spectrum.mid_points
    rebinnedWavelength = np.logspace(np.log10(wavelength[2]),np.log10(wavelength[-3]),21)

    # rebin x-axis
    rebinnedWavelengthFull = np.insert(rebinnedWavelength, 0, 250)
    rebinnedWavelengthFull = np.append(rebinnedWavelengthFull, 1.5e4)
    rebinnedWavelengthErrPos = (rebinnedWavelengthFull[2:] - rebinnedWavelengthFull[1:-1])/2
    rebinnedWavelengthErrNeg = (rebinnedWavelengthFull[1:-1] - rebinnedWavelengthFull[:-2])/2
    # rebin y-axis
    rebinned = spectres.spectres(rebinnedWavelength, wavelength, dataOrigin, spec_errs=dataOriginErr)
    rebinnedFlux = rebinned[0]
    rebinnedFluxErrPos = rebinned[1]
    rebinnedFluxErrNeg = rebinned[1]
    rebinnedFluxErrNeg[rebinnedFluxErrNeg > rebinnedFlux] = rebinnedFlux[rebinnedFluxErrNeg > rebinnedFlux]*0.95	

    # rebinnedWavelength[-1]=0

    plt.errorbar(rebinnedWavelength,k0*rebinnedWavelength*rebinnedWavelength*rebinnedFlux,\
                xerr=np.abs([rebinnedWavelengthErrNeg,rebinnedWavelengthErrPos]), \
                yerr=np.abs([k0*rebinnedWavelength*rebinnedWavelength*rebinnedFluxErrNeg,k0*rebinnedWavelength*rebinnedWavelength*rebinnedFluxErrPos]),marker='s',
                        label='BGO ('+gbm_detectors[bgo_index]+')',
                        fmt='.',ms=5,color='#4d4d4d',elinewidth=1.2,capthick=1.2,capsize=2)

    text_file = os.path.join(
        output_dir,
        model_str
        + "_bgo_"
        + gbm_detectors[bgo_index]
        + "_data_point"
        + time_bin_suffix
        + ".txt",
    )
    data = np.column_stack((rebinnedWavelength, rebinnedWavelengthErrNeg,rebinnedWavelengthErrPos,k0*rebinnedWavelength*rebinnedWavelength*rebinnedFlux,\
    k0*rebinnedWavelength*rebinnedWavelength*rebinnedFluxErrNeg,k0*rebinnedWavelength*rebinnedWavelength*rebinnedFluxErrPos))
    np.savetxt(text_file, data, delimiter=' ')


    ###################  convert to origin data and rebin for LLE if has LLE data

    # if LLElabel !='no LLE':
    #     print '#### plot LLE data     ####'

    #     area = lat_lle.expected_model_rate/modelTotal(lat_lle._observed_spectrum.mid_points)
    #     dataOrigin = lat_lle.source_rate/area
    #     dataOriginErr = lat_lle.source_rate_error/area

    #     # select data x-range
    #     wavelength = lat_lle._observed_spectrum.mid_points
    #     rebinnedWavelength = np.logspace(np.log10(wavelength[4]),np.log10(wavelength[15]),5)
    #     #rebinnedWavelength = wavelength

    #     # rebin x-axis
    #     rebinnedWavelengthFull = np.insert(rebinnedWavelength, 0, 2e4)
    #     rebinnedWavelengthFull = np.append(rebinnedWavelengthFull, 1e5)
    #     rebinnedWavelengthErrPos = (rebinnedWavelengthFull[2:] - rebinnedWavelengthFull[1:-1])/2
    #     rebinnedWavelengthErrNeg = (rebinnedWavelengthFull[1:-1] - rebinnedWavelengthFull[:-2])/2
    #     # rebin y-axis
    #     rebinned = spectres.spectres(rebinnedWavelength, wavelength, dataOrigin, spec_errs=dataOriginErr)
    #     rebinnedFlux = rebinned[0]
    #     rebinnedFluxErrPos = rebinned[1]
    #     rebinnedFluxErrNeg = rebinned[1]
    #     rebinnedFluxErrNeg[rebinnedFluxErrNeg > rebinnedFlux] = rebinnedFlux[rebinnedFluxErrNeg > rebinnedFlux]*0.95

    #     #rebinnedWavelength[-1]=0

    #     plt.errorbar(rebinnedWavelength,k0*rebinnedWavelength*rebinnedWavelength*rebinnedFlux,\
    #                 xerr=[rebinnedWavelengthErrNeg,rebinnedWavelengthErrPos], \
    #                 yerr=[k0*rebinnedWavelength*rebinnedWavelength*rebinnedFluxErrNeg,k0*rebinnedWavelength*rebinnedWavelength*rebinnedFluxErrPos],marker='d',label='LLE',fmt='.',ms=12,color='#aaaaaa')

    #     text_file = model_str+'_lle_data_point.txt'
    #     data = np.column_stack((rebinnedWavelength, rebinnedWavelengthErrNeg,rebinnedWavelengthErrPos,k0*rebinnedWavelength*rebinnedWavelength*rebinnedFlux,\
    # k0*rebinnedWavelength*rebinnedWavelength*rebinnedFluxErrNeg,k0*rebinnedWavelength*rebinnedWavelength*rebinnedFluxErrPos))
    #     np.savetxt(text_file, data, delimiter=' ')

    ###################  convert to origin data and rebin for LAT if has LAT data
    if "lat" in analysis_mode.lower() and lat is not None:
        area = lat.expected_model_rate / modelTotal(
            lat._observed_spectrum.mid_points
        )
        dataOrigin = lat.source_rate / area
        dataOriginErr = lat.source_rate_error / area

        # select data x-range
        wavelength = lat._observed_spectrum.mid_points
        end_bin = len(wavelength) - 1
        rebinnedWavelength = np.logspace(
            np.log10(wavelength[0]),
            np.log10(wavelength[end_bin]),
            6,
        )

        rebinnedWavelength = wavelength

        # rebin x-axis
        rebinnedWavelengthFull = np.insert(rebinnedWavelength, 0, 0.3e5)
        rebinnedWavelengthFull = np.append(rebinnedWavelengthFull, 1e8)

        rebinnedWavelengthErrPos = (rebinnedWavelengthFull[2:] - rebinnedWavelengthFull[1:-1]) / 2
        rebinnedWavelengthErrNeg = (rebinnedWavelengthFull[1:-1] - rebinnedWavelengthFull[:-2]) / 2
        # rebin y-axis
        rebinned = spectres.spectres(
            rebinnedWavelength,
            wavelength,
            dataOrigin,
            spec_errs=dataOriginErr,
        )
        rebinnedFlux = rebinned[0]
        rebinnedFluxErrPos = rebinned[1]
        rebinnedFluxErrNeg = rebinned[1]
        rebinnedFluxErrNeg[rebinnedFluxErrNeg > rebinnedFlux] = (
            rebinnedFlux[rebinnedFluxErrNeg > rebinnedFlux] * 0.95
        )

        plt.errorbar(
            rebinnedWavelength,
            k0 * rebinnedWavelength * rebinnedWavelength * rebinnedFlux,
            xerr=np.abs(
                [rebinnedWavelengthErrNeg, rebinnedWavelengthErrPos]
            ),
            yerr=np.abs(
                [
                    k0
                    * rebinnedWavelength
                    * rebinnedWavelength
                    * rebinnedFluxErrNeg,
                    k0
                    * rebinnedWavelength
                    * rebinnedWavelength
                    * rebinnedFluxErrPos,
                ]
            ),
            marker="v",
            label="LAT",
            fmt=".",
            ms=5,
            color="#4d4d4d",
            elinewidth=1.2,
            capthick=1.2,
            capsize=2,
        )

        text_file = os.path.join(
            output_dir,
            model_str + "_lat_data_point" + time_bin_suffix + ".txt",
        )
        data = np.column_stack(
            (
                rebinnedWavelength,
                rebinnedWavelengthErrNeg,
                rebinnedWavelengthErrPos,
                k0 * rebinnedWavelength * rebinnedWavelength * rebinnedFlux,
                k0
                * rebinnedWavelength
                * rebinnedWavelength
                * rebinnedFluxErrNeg,
                k0
                * rebinnedWavelength
                * rebinnedWavelength
                * rebinnedFluxErrPos,
            )
        )
        np.savetxt(text_file, data, delimiter=" ")


    # 能量上限：GBM-only 用 1e5 keV（100 MeV），GBM+LAT 扩展到 1e8 keV（100 GeV）
    if "lat" in analysis_mode.lower() and lat is not None:
        emax = 1e8
    else:
        emax = 1e5
    xs = np.logspace(np.log10(8.),5.0,100)
    xs1 = np.logspace(5.0,np.log10(emax),100)
    xs=np.append(xs, xs1[1:])
    fluxPL = k0*xs*xs*modelTotal(xs)
    if model_str in ['pl','band','blackbody','comp','SBPL','NDP','mbb']:
        plt.loglog(xs,k0*xs*xs*modelTotal(xs),'-',linewidth=2,label=f'{model_str}', color='b')
        if model_str in ['blackbody']:
            Epeak = 3.92*parameter_values[1]
            fluxEpeak = max(k0*xs*xs*modelTotal(xs))
            flux100mev = fluxPL[np.where( xs == 100000.0)]
        elif model_str in ['mbb']:
            kT_min, kT_max = parameter_values[1], parameter_values[2]
            Epeak = 3.92 * np.sqrt(kT_min * kT_max)
            fluxEpeak = max(k0*xs*xs*modelTotal(xs))
            flux100mev = fluxPL[np.where(xs == 100000.0)]
        elif model_str in ['pl']:
            Epeak = 0
            fluxEpeak = 0
            flux100mev = fluxPL[np.where( xs == 100000.0)]
        elif model_str in ['NDP']:
            # NDP (NonDissipativePhotosphere) 模型的特征能量使用 ec
            Epeak = parameter_values[1]
            fluxEpeak = max(k0*xs*xs*modelTotal(xs))
            flux100mev = fluxPL[np.where( xs == 100000.0)]

        # elif model_str in ['comp']:
        #     # 康普顿化模型的特征能量计算
        #     # comp 模型通常峰值在 Ep 附近
        #     Epeak = parameter_values[2]  # 直接用 Ep 作为峰值能量
        #     fluxEpeak = max(k0*xs*xs*modelTotal(xs))
        #     flux100mev = fluxPL[np.where( xs == 100000.0)]
        elif model_str in ['SBPL']:
            Epeak = parameter_values[3]  # E0作为参考峰值
            fluxEpeak = max(k0*xs*xs*modelTotal(xs))
            flux100mev = fluxPL[np.where( xs == 100000.0)]
        else:
            Epeak = (2+parameter_values[1])*parameter_values[2]
            fluxEpeak = max(k0*xs*xs*modelTotal(xs))
            flux100mev = fluxPL[np.where( xs == 100000.0)]

    else:
        # 对于组合模型，先分别计算各个组件的flux，然后相加
        # 这样可以确保模型计算正确
        if model_str in ['band+bb+pl']:
            # 三模型组合: Band + Blackbody + Powerlaw
            flux1 = k0*xs*xs*model1(xs)  # Band component
            flux2 = k0*xs*xs*model2(xs)  # Blackbody component
            flux3 = k0*xs*xs*model3(xs)  # Powerlaw component
            fluxTotal = flux1 + flux2 + flux3
            
            # 绘制总模型和各个组件
            plt.loglog(xs, fluxTotal, '-', linewidth=2, label=model_str1+'+'+model_str2+'+'+model_str3, color='b')
            plt.loglog(xs, flux1, ':', linewidth=1.5, label=model_str1, color='r')
            plt.loglog(xs, flux2, '--', linewidth=2, label=model_str2, color='g')
            plt.loglog(xs, flux3, '-.', linewidth=2, label=model_str3, color='m')
            
            # 计算峰值能量和通量
            Epeak = (2+parameter_values[1])*parameter_values[2]  # 基于Band模型参数计算
            fluxEpeak = max(flux1)  # 使用Band组件的峰值
            idx_100mev = np.argmin(np.abs(xs - 100000.0))
            flux100mev = fluxTotal[idx_100mev]  # 使用总模型的100MeV通量
        else:
            # 对于其他组合模型（双模型）
            flux1 = k0*xs*xs*model1(xs)
            flux2 = k0*xs*xs*model2(xs)
            # modelTotal应该是两个组件的和
            fluxTotal = flux1 + flux2
            
            # 绘制总模型和各个组件
            plt.loglog(xs, fluxTotal, '-', linewidth=2, label=model_str1+'+'+model_str2, color='b')
            plt.loglog(xs, flux1, ':', linewidth=1.5, label=model_str1, color='r')
            plt.loglog(xs, flux2, '--', linewidth=2, label=model_str2, color='g')

            
            if model_str in ['pl+bb']:
                Epeak = 3.92*parameter_values[1]

            elif model_str in ['comp+pl']:
                # cutoff power-law峰值，pl用于高能延拓
                Epeak = (2 + parameter_values[1]) * parameter_values[2]
                fluxEpeak = max(flux1)  # 使用comp组件的峰值
                idx_100mev = np.argmin(np.abs(xs - 100000.0))
                flux100mev = fluxTotal[idx_100mev]  # 使用总模型的100MeV通量

            elif model_str in ['band+gauss']:
                # Band + Gaussian 组合模型
                Epeak = (2+parameter_values[1])*parameter_values[2]  # 基于Band模型参数计算
                fluxEpeak = max(flux1)  # 使用Band组件的峰值
                idx_100mev = np.argmin(np.abs(xs - 100000.0))
                flux100mev = fluxTotal[idx_100mev]  # 使用总模型的100MeV通量
            
            elif model_str in ['band+bb']:
                # 查找最接近 100 MeV 的点
                Epeak = (2+parameter_values[1])*parameter_values[2]
                # idx_100mev = np.argmin(np.abs(xs - 100000.0))
                # flux100mev = fluxPL[idx_100mev]
                fluxEpeak = max(k0*xs*xs*model1(xs))
                fluxPL = k0*xs*xs*model2(xs)
                flux100mev = fluxPL[np.where( xs == 100000.0)]
            elif model_str in ['band+pl']:
                # Band + Power-law 组合模型
                Epeak = (2 + parameter_values[1]) * parameter_values[2]
                fluxEpeak = max(k0 * xs * xs * model1(xs))  # 使用Band组件的峰值
                fluxPL = k0 * xs * xs * model2(xs)
                idx_100mev = np.argmin(np.abs(xs - 100000.0))
                flux100mev = fluxPL[idx_100mev]  # 使用pl组件在100 MeV的通量
            elif model_str in ['mbb+pl']:
                kT_min, kT_max = parameter_values[1], parameter_values[2]
                Epeak = 3.92 * np.sqrt(kT_min * kT_max)
                fluxEpeak = max(flux1)
                fluxPL = k0 * xs * xs * model2(xs)
                idx_100mev = np.argmin(np.abs(xs - 100000.0))
                flux100mev = fluxPL[idx_100mev]
            else:
                Epeak = (2+parameter_values[1])*parameter_values[2]
                fluxEpeak = max(k0*xs*xs*model1(xs))
                fluxPL = k0*xs*xs*model2(xs)
                flux100mev = fluxPL[np.where( xs == 100000.0)]

    plt.xlim([5e0, emax*2.0])

    plt.ylim([k0*min(xs*xs*modelTotal(xs))*0.02, k0*max(xs*xs*modelTotal(xs))*50.0])
    if model_str == 'pl':
        plt.ylim([k0*min(xs*xs*modelTotal(xs))*0.02, k0*max(xs*xs*modelTotal(xs))*50.0])
    if model_str == 'band':
        plt.ylim([k0*min(xs*xs*modelTotal(xs))*0.02, k0*max(xs*xs*modelTotal(xs))*50.0])
    if model_str == 'comp':
        xc=parameter_values[2]
        plt.ylim([k0*xc*xc*modelTotal(xc)*0.001, k0*max(xs*xs*modelTotal(xs))*100.0])
    if model_str == 'NDP':
        ec = parameter_values[1]
        plt.ylim([k0*ec*ec*modelTotal(ec)*0.001, k0*max(xs*xs*modelTotal(xs))*100.0])
    if model_str == 'blackbody':
        kT = parameter_values[1]
        plt.ylim([k0*kT*kT*modelTotal(kT)*0.001, k0*max(xs*xs*modelTotal(xs))*100.0])
    if model_str == 'mbb':
        kT_min, kT_max = parameter_values[1], parameter_values[2]
        kT_ref = np.sqrt(kT_min * kT_max)
        plt.ylim(
            [
                k0 * kT_ref * kT_ref * modelTotal(kT_ref) * 0.001,
                k0 * max(xs * xs * modelTotal(xs)) * 100.0,
            ]
        )
    if model_str in ['mbb+pl']:
        kT_min, kT_max = parameter_values[1], parameter_values[2]
        kT_ref = np.sqrt(kT_min * kT_max)
        plt.ylim(
            [
                k0 * kT_ref * kT_ref * modelTotal(kT_ref) * 0.001,
                k0 * max(xs * xs * modelTotal(xs)) * 100.0,
            ]
        )
    if model_str in ['band+bb']:
        plt.ylim([k0*min(xs*xs*modelTotal(xs))*0.01, 
                  k0*max(xs*xs*modelTotal(xs))*100.0])

    if model_str == 'comp+bb':
        plt.ylim(1e-10, 1e-5)
        # plt.ylim([k0*min(xs*xs*modelTotal(xs))*0.01, 
        #           k0*max(xs*xs*modelTotal(xs))*100.0])

    if model_str == 'SBPL':
        # 在转折能量附近多采样一些点以更好地显示转折特征
        xs_around_E0 = np.logspace(np.log10(parameter_values[3]*0.1), 
                                    np.log10(parameter_values[3]*10), 100)
        xs = np.sort(np.append(xs, xs_around_E0))
        
        # 自适应设置 y 轴范围
        y_values = k0*xs*xs*modelTotal(xs)
        y_min = np.min(y_values[y_values > 0])
        y_max = np.max(y_values)
        plt.ylim([y_min*0.5, y_max*5.0])

    ax = plt.gca()
    ax.set_xlabel('$E$ [keV]')
    ax.set_ylabel('$E^{2} dN/dE$ [erg s$^{-1}$cm$^{-2}$]')
    # ax.set_title(f"{bnname}_spectra{time_bin_suffix}")
    ax.minorticks_on()
    _style_publication_axes(ax)
    ax.legend(loc=2)
    suffix = "gbm_lat" if "lat" in analysis_mode.lower() and lat is not None else "gbm"
    out_name = f"bs_{bnname}_{suffix}_spectra_{model_str}{time_bin_suffix}.pdf"
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, out_name), bbox_inches="tight")
