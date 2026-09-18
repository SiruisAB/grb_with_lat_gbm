"""Redraw threeML diagnostics from saved posterior FITS files without refitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import yaml

from .gbm_detector_selection import select_gbm_detectors
from .io_utils import find_files
from .publication_style import (
    save_publication_figure,
    style_corner_figure,
    style_diagnostic_figure,
)


@dataclass(frozen=True)
class BurstPlotConfig:
    grb_name: str
    bnname: str
    background_interval: str
    source_interval: str
    intervals: tuple[tuple[float, float], ...]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Redraw count-space and posterior diagnostics from my_results_*.fits"
    )
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--grb-name", required=True)
    parser.add_argument("--bnname", required=True)
    parser.add_argument("--raw-data-root", type=Path, required=True)
    parser.add_argument(
        "--special-yaml",
        type=Path,
        default=Path(__file__).with_name("special_bursts.yaml"),
    )
    parser.add_argument("--models", nargs="+", default=("band", "band+bb"))
    parser.add_argument("--corner-models", nargs="+", default=("band+bb",))
    parser.add_argument(
        "--lat-max-bins",
        type=int,
        default=7,
        help="Maximum approximate number of displayed LAT energy bins",
    )
    parser.add_argument(
        "--gbm-min-rate",
        type=float,
        default=20.0,
        help="Display-only GBM minimum rate used for channel grouping",
    )
    parser.add_argument(
        "--gbm-max-bins",
        type=int,
        default=60,
        help="Maximum approximate number of displayed bins per GBM detector",
    )
    parser.add_argument(
        "--interval",
        action="append",
        default=[],
        help="Optional start-stop interval; may be repeated",
    )
    return parser.parse_args(argv)


def _parse_interval(text: str) -> tuple[float, float]:
    start_text, stop_text = str(text).strip().split("-", 1)
    start, stop = float(start_text), float(stop_text)
    if stop <= start:
        raise ValueError(f"Invalid interval: {text}")
    return start, stop


def _load_burst_config(
    yaml_path: Path,
    grb_name: str,
    bnname: str,
    requested_intervals: Iterable[str] = (),
) -> BurstPlotConfig:
    payload = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8")) or {}
    entries = payload.get("special_bursts", [])
    match = next(
        (
            entry
            for entry in entries
            if str(entry.get("bnname", "")).strip() == bnname
            or str(entry.get("name", "")).strip() == grb_name
        ),
        None,
    )
    if match is None:
        raise KeyError(f"No special-burst entry for {grb_name}/{bnname}")

    if requested_intervals:
        intervals = tuple(_parse_interval(text) for text in requested_intervals)
    else:
        intervals = tuple(
            (float(segment["start"]), float(segment["stop"]))
            for segment in match.get("time_segments", [])
        )
    if not intervals:
        raise ValueError(f"No time intervals configured for {grb_name}")

    source_interval = str(match.get("active_interval", "")).strip()
    if not source_interval:
        source_interval = f"{intervals[0][0]:g}-{intervals[-1][1]:g}"
    return BurstPlotConfig(
        grb_name=grb_name,
        bnname=bnname,
        background_interval=str(match["background_interval"]),
        source_interval=source_interval,
        intervals=intervals,
    )


def _normalized_interval(text: str) -> tuple[float, float]:
    start, stop = _parse_interval(text)
    return round(start, 8), round(stop, 8)


def _find_result_fits(model_dir: Path, start: float, stop: float) -> Path:
    expected = round(start, 8), round(stop, 8)
    matches = []
    for path in sorted(Path(model_dir).glob("my_results_*.fits")):
        token = path.stem.removeprefix("my_results_")
        try:
            candidate = _normalized_interval(token)
        except (TypeError, ValueError):
            continue
        if candidate == expected:
            matches.append(path)
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one result FITS for {start:g}-{stop:g} in {model_dir}; found {matches}"
        )
    return matches[0]


def _interval_token(result_fits: Path) -> str:
    return result_fits.stem.removeprefix("my_results_")


def _lat_token(start: float, stop: float) -> str:
    return f"{start:g}_{stop:g}"


def _find_lat_plugin_files(result_dir: Path, start: float, stop: float) -> tuple[Path, Path]:
    lat_dir = result_dir / "lat" if (result_dir / "lat").is_dir() else result_dir
    token = _lat_token(start, stop)
    pha = lat_dir / f"prompt_selection_new_LAT_{token}.pha"
    rsp = lat_dir / f"prompt_selection_new_LAT_{token}.rsp"
    if not pha.is_file() or not rsp.is_file():
        raise FileNotFoundError(f"Missing LAT PHA/RSP for {start:g}-{stop:g}: {pha}, {rsp}")
    return pha, rsp


def _background_dir(result_dir: Path) -> Path:
    gbm_dir = result_dir / "gbm"
    return gbm_dir if gbm_dir.is_dir() else result_dir


def _prepare_work_dir(result_dir: Path, output_root: Path, grb_name: str) -> Path:
    work_dir = Path(output_root) / grb_name / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    for background in sorted(_background_dir(result_dir).glob("*_bkg.h5")):
        shutil.copy2(background, work_dir / background.name)
    return work_dir


def _find_detector_file(source_dir: Path, detector: str, suffix: str, marker: str) -> Path:
    matches = sorted(
        path
        for path in find_files(str(source_dir), suffix)
        if marker in Path(path).name
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {detector} {suffix} file containing {marker} in "
            f"{source_dir}; found {matches}"
        )
    return Path(matches[0]).resolve()


def _build_restored_gbm_plugin(
    detector: str,
    source_dir: Path,
    work_dir: Path,
    start: float,
    stop: float,
    source_interval: str,
    time_series: dict,
):
    """Build one GBM plugin while reusing the saved background polynomial."""
    from threeML import OGIPLike, TimeSeriesBuilder

    ts_tte = time_series.get(detector)
    if ts_tte is None:
        tte = _find_detector_file(source_dir, detector, ".fit", f"_tte_{detector}_")
        try:
            rsp = _find_detector_file(
                source_dir, detector, ".rsp2", f"_cspec_{detector}_"
            )
        except FileNotFoundError:
            rsp = _find_detector_file(
                source_dir, detector, ".rsp", f"_cspec_{detector}_"
            )
        background = (work_dir / f"{detector}_bkg.h5").resolve()
        if not background.is_file():
            raise FileNotFoundError(
                f"Missing saved background for detector {detector}: {background}"
            )
        ts_tte = TimeSeriesBuilder.from_gbm_tte(
            detector,
            tte_file=str(tte),
            rsp_file=str(rsp),
            restore_background=str(background),
            poly_order=-1,
        )
        time_series[detector] = ts_tte

    ts_tte.set_active_time_interval(source_interval)
    ts_tte.create_time_bins(start=[start], stop=[stop], method="custom")

    prefix = f"gbm_tte_{detector}"
    previous_cwd = Path.cwd()
    os.chdir(work_dir)
    try:
        ts_tte.write_pha_from_binner(
            file_name=prefix,
            overwrite=True,
            force_rsp_write=True,
        )
        plugin = OGIPLike(
            detector,
            f"{prefix}.pha",
            f"{prefix}_bak.pha",
            f"{prefix}.rsp",
            spectrum_number=1,
        )
    finally:
        os.chdir(previous_cwd)

    if detector.startswith("b"):
        if (
            hasattr(plugin, "energy_boundaries")
            and len(plugin.energy_boundaries) > 1
            and len(plugin.energy_boundaries[1]) > 0
            and plugin.energy_boundaries[1][0] >= 200
        ):
            plugin.set_active_measurements(exclude=["c0-c1", "40000-c128"])
        else:
            plugin.set_active_measurements(exclude=["0-200", "40000-c128"])
    else:
        plugin.set_active_measurements(exclude=["0-8", "30-40", "c126-c128"])
    return plugin


def _build_data_list(
    config: BurstPlotConfig,
    result_dir: Path,
    raw_data_root: Path,
    work_dir: Path,
    start: float,
    stop: float,
    time_series: dict,
):
    from threeML import DataList, OGIPLike

    source_dir = Path(raw_data_root) / config.bnname
    detectors, *_ = select_gbm_detectors(str(source_dir))
    if not detectors:
        raise RuntimeError(f"No GBM detectors selected from {source_dir}")

    gbm_plugins = []
    for detector in detectors:
        plugin = _build_restored_gbm_plugin(
            detector=detector,
            source_dir=source_dir,
            work_dir=work_dir,
            start=start,
            stop=stop,
            source_interval=config.source_interval,
            time_series=time_series,
        )
        gbm_plugins.append(plugin)

    lat_pha, lat_rsp = _find_lat_plugin_files(result_dir, start, stop)
    lat_plugin = OGIPLike("LAT", observation=str(lat_pha), response=str(lat_rsp))
    lat_plugin.set_active_measurements("100000.-100000000.")
    return DataList(lat_plugin, *gbm_plugins), detectors


def _save_both(fig, output_stem: Path) -> tuple[Path, Path]:
    # Keep every diagnostic media box fixed to the reference spectra canvas.
    with mpl.rc_context({"savefig.bbox": None}):
        png = save_publication_figure(
            fig, Path(f"{output_stem}.png"), dpi=300, bbox_inches=None
        )
        pdf = save_publication_figure(
            fig, Path(f"{output_stem}.pdf"), dpi=300, bbox_inches=None
        )
    return png, pdf


def _corner_parameter_aliases(result) -> dict[str, str]:
    short_aliases = {
        "K_1": r"$K_{\rm Band}$",
        "alpha_1": r"$\alpha$",
        "xp_1": r"$E_{\rm peak}$",
        "beta_1": r"$\beta$",
        "K_2": r"$K_{\rm BB}$",
        "kT_2": r"$kT$",
    }
    return {
        path: short_aliases.get(path.split(".")[-1], path.split(".")[-1])
        for path in result.optimized_model.free_parameters
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def redraw_saved_diagnostics(
    config: BurstPlotConfig,
    result_root: Path,
    output_root: Path,
    raw_data_root: Path,
    models: Sequence[str] = ("band", "band+bb"),
    corner_models: Sequence[str] = ("band+bb",),
    lat_max_bins: int = 7,
    gbm_min_rate: float = 20.0,
    gbm_max_bins: int = 60,
) -> list[dict]:
    from threeML import BayesianAnalysis, display_spectrum_model_counts, load_analysis_results
    from threeML.utils.binner import NotEnoughData

    if lat_max_bins < 2:
        raise ValueError("LAT maximum bin count must be at least two")
    if gbm_min_rate < 0:
        raise ValueError("GBM display minimum rate must be non-negative")
    if gbm_max_bins < 2:
        raise ValueError("GBM maximum bin count must be at least two")

    result_dir = Path(result_root) / config.grb_name
    output_root = Path(output_root)
    work_dir = _prepare_work_dir(result_dir, output_root, config.grb_name)
    time_series: dict = {}
    manifest: list[dict] = []

    for start, stop in config.intervals:
        data_list, detectors = _build_data_list(
            config,
            result_dir,
            raw_data_root,
            work_dir,
            start,
            stop,
            time_series,
        )
        for model_name in models:
            result_fits = _find_result_fits(result_dir / model_name, start, stop)
            token = _interval_token(result_fits)
            result = load_analysis_results(str(result_fits))
            analysis = BayesianAnalysis(result.optimized_model, data_list)
            # Group display channels only; the restored fit and likelihood are unchanged.
            data_keys = list(analysis.data_list.keys())
            min_rate: list[float] = []
            display_bin_counts: list[int] = []
            for data_index, data_key in enumerate(data_keys):
                plugin = analysis.data_list[data_key]
                if data_index == 0:
                    model_rate_total = float(plugin.expected_model_rate.sum())
                    target_rate = model_rate_total / (lat_max_bins - 1)
                else:
                    model_rate_total = float(plugin.expected_model_rate.sum())
                    target_rate = max(
                        gbm_min_rate,
                        model_rate_total / (gbm_max_bins - 1),
                    )
                for grouping_scale in (1.0, 0.5, 0.25, 0.1, 0.05, 0.0):
                    candidate_rate = target_rate * grouping_scale
                    try:
                        grouped = plugin._construct_counts_arrays(candidate_rate)
                    except NotEnoughData:
                        continue
                    min_rate.append(candidate_rate)
                    display_bin_counts.append(len(grouped["mean_energy"]))
                    break
                else:  # pragma: no cover - zero-rate grouping should always succeed
                    raise RuntimeError(f"Unable to group {data_key}")

            figure = display_spectrum_model_counts(
                analysis,
                min_rate=min_rate,
                show_background=False,
                figsize=(3.5, 2.65),
            )
            style_diagnostic_figure(figure)
            count_stem = (
                output_root
                / config.grb_name
                / model_name
                / f"bs_{config.bnname}_{model_name}_counts_gbm_lat_spectrum_{token}"
            )
            count_png, count_pdf = _save_both(figure, count_stem)
            axes_count = len(figure.axes)
            plt.close(figure)

            outputs = [count_png, count_pdf]
            if model_name in corner_models:
                corner_figure = result.corner_plot(
                    renamed_parameters=_corner_parameter_aliases(result),
                    max_n_ticks=3,
                    use_math_text=True,
                )
                style_corner_figure(corner_figure)
                corner_stem = (
                    output_root
                    / config.grb_name
                    / model_name
                    / f"bs_{config.bnname}_{model_name}_{token}_gbm_lat_corner_plot"
                )
                corner_png, corner_pdf = _save_both(corner_figure, corner_stem)
                outputs.extend((corner_png, corner_pdf))
                plt.close(corner_figure)

            manifest.append(
                {
                    "grb_name": config.grb_name,
                    "bnname": config.bnname,
                    "interval": token,
                    "model": model_name,
                    "result_fits": str(result_fits.resolve()),
                    "detectors": ["LAT", *detectors],
                    "display_min_rate": dict(zip(data_keys, min_rate)),
                    "display_bin_counts": dict(
                        zip(data_keys, display_bin_counts)
                    ),
                    "count_axes": axes_count,
                    "outputs": [
                        {
                            "path": str(path.resolve()),
                            "bytes": path.stat().st_size,
                            "sha256": _sha256(path),
                        }
                        for path in outputs
                    ],
                }
            )

    manifest_path = output_root / config.grb_name / "diagnostic_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = _load_burst_config(
        args.special_yaml,
        args.grb_name,
        args.bnname,
        args.interval,
    )
    manifest = redraw_saved_diagnostics(
        config=config,
        result_root=args.result_root,
        output_root=args.output_root,
        raw_data_root=args.raw_data_root,
        models=args.models,
        corner_models=args.corner_models,
        lat_max_bins=args.lat_max_bins,
        gbm_min_rate=args.gbm_min_rate,
        gbm_max_bins=args.gbm_max_bins,
    )
    print(json.dumps({"generated_model_intervals": len(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
