# -*- coding: utf-8 -*-
"""单次分析入口，由 Web 工作台和项目对象复用。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from .config import GRBRunOverrides
from .session import session


def main(
    target_grbs=None,
    analysis_mode=None,
    result_root=None,
    summary_csv_name=None,
    session_log: bool = False,
    fixed_num_time_bins=None,
    run_overrides: Optional[GRBRunOverrides] = None,
):
    from .project import run_single_analysis

    owned_log_handle = None
    if session_log and session.log_file_handle is None:
        log_root = Path(result_root or session.result_root).expanduser()
        log_root.mkdir(parents=True, exist_ok=True)
        owned_log_handle = (log_root / "analysis_session.log").open("a", encoding="utf-8")
        session.log_file_handle = owned_log_handle

    try:
        return run_single_analysis(
            target_grbs=target_grbs,
            analysis_mode=analysis_mode,
            result_root=result_root,
            summary_csv_name=summary_csv_name,
            fixed_num_time_bins=fixed_num_time_bins,
            run_overrides=run_overrides,
        )
    finally:
        if owned_log_handle is not None:
            owned_log_handle.close()
            session.log_file_handle = None


def _build_analysis_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fermi GBM/LAT single-GRB analysis")
    parser.add_argument("--grbs", nargs="+", default=None)
    parser.add_argument("--analysis-mode", choices=("gbm", "lat", "gbm+lat"), default="gbm+lat")
    parser.add_argument("--result-root", default=None)
    parser.add_argument("--summary-csv-name", default=None)
    parser.add_argument("--session-log", action="store_true")
    parser.add_argument("--fixed-num-time-bins", type=int, default=None)
    parser.add_argument("--lat-extended-three-ml", action="store_true")
    parser.add_argument("--plot-joint-lightcurve", dest="plot_joint_lightcurve", action="store_true")
    parser.add_argument("--no-plot-joint-lightcurve", dest="plot_joint_lightcurve", action="store_false")
    parser.set_defaults(plot_joint_lightcurve=None)
    return parser


def _build_download_parser() -> argparse.ArgumentParser:
    from .gbm_download import add_download_arguments

    parser = argparse.ArgumentParser(description="Download Fermi GBM burst data")
    add_download_arguments(parser)
    return parser


def _build_lat_gcn_parser() -> argparse.ArgumentParser:
    from .lat_gcn_extract import add_extract_arguments

    parser = argparse.ArgumentParser(description="Refresh GCN archive and extract Fermi-LAT data")
    add_extract_arguments(parser)
    return parser


def _build_lat_download_parser() -> argparse.ArgumentParser:
    from .lat_download import add_download_arguments

    parser = argparse.ArgumentParser(description="Download Fermi-LAT Extended data through threeML")
    add_download_arguments(parser)
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "download-gbm":
        args = _build_download_parser().parse_args(argv[1:])
        args.command = "download-gbm"
        return args
    if argv and argv[0] == "update-lat-gcn":
        args = _build_lat_gcn_parser().parse_args(argv[1:])
        args.command = "update-lat-gcn"
        return args
    if argv and argv[0] == "download-lat":
        args = _build_lat_download_parser().parse_args(argv[1:])
        args.command = "download-lat"
        return args

    args = _build_analysis_parser().parse_args(argv)
    args.command = "analyze"
    return args


def cli_main(argv=None):
    args = parse_args(argv)
    if args.command == "download-gbm":
        from .gbm_download import cli_main_from_args

        return cli_main_from_args(args)
    if args.command == "update-lat-gcn":
        from .lat_gcn_extract import cli_main_from_args

        return cli_main_from_args(args)
    if args.command == "download-lat":
        from .lat_download import cli_main_from_args

        return cli_main_from_args(args)

    overrides = GRBRunOverrides(
        lat_three_ml_full=True if args.lat_extended_three_ml else None,
        plot_joint_lightcurve=args.plot_joint_lightcurve,
    )
    if overrides.is_empty():
        overrides = None
    return main(
        target_grbs=args.grbs,
        analysis_mode=args.analysis_mode,
        result_root=args.result_root,
        summary_csv_name=args.summary_csv_name,
        session_log=args.session_log,
        fixed_num_time_bins=args.fixed_num_time_bins,
        run_overrides=overrides,
    )
