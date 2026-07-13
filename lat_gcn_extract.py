# -*- coding: utf-8 -*-
"""Download the current GCN Circular archive and extract Fermi-LAT records."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import tarfile
import tempfile
from typing import Callable
import urllib.request

import pandas as pd


DEFAULT_ARCHIVE_URL = "https://gcn.nasa.gov/circulars/archive.json.tar.gz"
DEFAULT_ARCHIVE_TAR = Path("/home/mxr/lee/data/archive.json.tar.gz")
DEFAULT_ARCHIVE_DIR = Path("/home/mxr/lee/data/archive.json")
DEFAULT_OUTPUT_CSV = Path("/home/mxr/lee/data/GRB_FermiLAT_Time_Extended.csv")

OUTPUT_COLUMNS = [
    "Circular ID",
    "GRB Name",
    "Trigger Time",
    "T0 (s)",
    "T1 (s)",
    "Photon Energy",
    "HE Photon Time (s)",
    "RA",
    "DEC",
]

RE_GRB_NAME = re.compile(r"(GRB\s*\d{6}[A-Z]?)", re.IGNORECASE)
RE_TIME_MAIN = re.compile(
    r"time\s+interval[*]*\s*(?:of\s*)?([-+]?\d*\.?\d+)\s*(?:-|–|to)\s*([-+]?\d*\.?\d+)\s*s",
    re.IGNORECASE,
)
RE_TIME_T0PLUS = re.compile(r"t0\+([0-9]+)\s*(?:to|and|-)\s*t0\+([0-9]+)", re.IGNORECASE)
RE_TIME_BETWEEN = re.compile(
    r"between\s*t0\s*\+\s*([0-9]+)\s*s\s*and\s*t0\s*\+\s*([0-9]+)\s*s", re.IGNORECASE
)
RE_TIME_MIXED = re.compile(
    r"t0\+([0-9]+)\s*(ks|s|m|h)?\s*[-–]\s*t0\+([0-9]+)\s*(ks|s|m|h)", re.IGNORECASE
)
RE_TIME_T0_START = re.compile(r"t0\s*[-–]\s*t0\+([0-9]+)\s*(ks|s|m|h)?", re.IGNORECASE)
RE_TIME_DURING = re.compile(r"during the interval\s+0\s+([0-9]+)\s*s", re.IGNORECASE)
RE_TIME_NUMBERS = re.compile(
    r"([0-9]+)\s*[-–]\s*([0-9]+)\s*(ks|s|m|h)(?=\s*(?:after|from|starting))", re.IGNORECASE
)
RE_PHOTON = re.compile(
    r"highest[- ]energy photon.*?is\s+a\s+([0-9]*\.?[0-9]+)\s*(gev|mev).*?"
    r"observed\s*(?:~|about|around)?\s*([0-9]*\.?[0-9]+)\s*seconds\s*[\n ]*after the "
    r"(?:gbm|swift|fermi) trigger",
    re.IGNORECASE | re.DOTALL,
)
RE_PHOTON_ALT = re.compile(
    r"observed\s+(?:at\s+)?(?:~|about|approximately)?\s*([0-9]*\.?[0-9]+)\s*seconds?\s+after.*?"
    r"(?:gbm|swift|fermi).*?trigger",
    re.IGNORECASE,
)
RE_TRIGGER = re.compile(r"trigger\s+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
RE_TRIGGER_SLASH = re.compile(
    r"trigger\s+([0-9]+(?:\.[0-9]+)?)\s*/\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE
)
RE_FLUX_INTERVAL = re.compile(
    r"the photon flux.*?in the time interval\s+([0-9]*\.?[0-9]+)\s*[-–]\s*"
    r"([0-9]*\.?[0-9]+)\s*(ks|s|m|h).*?after ",
    re.IGNORECASE,
)
RE_POSITION = re.compile(
    r"RA,\s*Dec\s*\)?\s*=\s*([0-9]+\.[0-9]+)\s*,\s*([+-]?[0-9]+\.[0-9]+)", re.IGNORECASE
)
RE_POSITION_ALT = re.compile(
    r"RA[=:]\s*([0-9]+\.[0-9]+)\s*(?:deg)?\s*(?:[,/&]\s*|and\s+)"
    r"(?:Dec[=:]\s*|\s+)([+-]?[0-9]+\.[0-9]+)\s*(?:deg)?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LatGcnRefreshResult:
    archive_dir: Path
    archive_tar_path: Path
    output_csv: Path
    record_count: int


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "grb-project/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def _safe_extract(archive_path: Path, destination: Path) -> Path:
    destination_resolved = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if target != destination_resolved and destination_resolved not in target.parents:
                raise ValueError(f"GCN archive contains an unsafe path: {member.name}")
        archive.extractall(destination, members=members)

    extracted = destination / "archive.json"
    if not extracted.is_dir() or not any(extracted.glob("*.json")):
        raise ValueError("GCN archive does not contain archive.json/*.json")
    return extracted


def _seconds(value: float, unit: str) -> float:
    return value * {"s": 1, "ks": 1000, "m": 60, "h": 3600}.get(unit.lower(), 1)


def _extract_interval(body: str) -> tuple[str, str]:
    patterns = [
        ("main", RE_TIME_MAIN),
        ("t0plus", RE_TIME_T0PLUS),
        ("between", RE_TIME_BETWEEN),
        ("mixed", RE_TIME_MIXED),
        ("t0_start", RE_TIME_T0_START),
        ("during", RE_TIME_DURING),
        ("numbers", RE_TIME_NUMBERS),
    ]
    for name, pattern in patterns:
        match = pattern.search(body)
        if not match:
            continue
        if name == "mixed":
            start = _seconds(float(match.group(1)), match.group(2) or "s")
            stop = _seconds(float(match.group(3)), match.group(4))
            return str(int(start)), str(int(stop))
        if name in {"t0_start", "during"}:
            start, stop = 0.0, float(match.group(1))
            unit = match.group(2) if name == "t0_start" and match.lastindex and match.lastindex >= 2 else "s"
        else:
            start, stop = float(match.group(1)), float(match.group(2))
            unit = match.group(3) if name == "numbers" else "s"
        return str(int(_seconds(start, unit or "s"))), str(int(_seconds(stop, unit or "s")))
    return "N/A", "N/A"


def _parse_record(filename: str, data: dict) -> dict | None:
    subject = str(data.get("subject", ""))
    body = str(data.get("body", ""))
    if "GRB" not in subject or re.search(r"fermi[-\s]+lat detection", subject, re.IGNORECASE) is None:
        return None

    name_match = RE_GRB_NAME.search(subject)
    grb_name = name_match.group(1).replace(" ", "").upper() if name_match else "N/A"
    t0, t1 = _extract_interval(body)
    flux_match = RE_FLUX_INTERVAL.search(body)
    if flux_match:
        unit = flux_match.group(3)
        if t0 == "N/A":
            t0 = str(int(_seconds(float(flux_match.group(1)), unit)))
        if t1 == "N/A":
            t1 = str(int(_seconds(float(flux_match.group(2)), unit)))

    photon_match = RE_PHOTON.search(body)
    if photon_match:
        photon_energy = f"{photon_match.group(1)} {photon_match.group(2).upper()}"
        photon_time = photon_match.group(3)
    else:
        photon_energy = "N/A"
        alt_match = RE_PHOTON_ALT.search(body)
        photon_time = alt_match.group(1) if alt_match else "N/A"

    if grb_name in {"GRB221025A", "GRB221027A", "GRB221119A"}:
        slash_match = RE_TRIGGER_SLASH.search(body)
        trigger_time = slash_match.group(2) if slash_match else "N/A"
    else:
        trigger_match = RE_TRIGGER.search(body)
        trigger_time = trigger_match.group(1) if trigger_match else "N/A"

    position_match = RE_POSITION.search(body) or RE_POSITION_ALT.search(body)
    ra = position_match.group(1) if position_match else "N/A"
    dec = position_match.group(2) if position_match else "N/A"
    return {
        "Circular ID": data.get("circularId", "N/A"),
        "GRB Name": grb_name,
        "Trigger Time": trigger_time,
        "T0 (s)": t0,
        "T1 (s)": t1,
        "Photon Energy": photon_energy,
        "HE Photon Time (s)": photon_time,
        "RA": ra,
        "DEC": dec,
        "File": filename,
    }


def extract_lat_records(archive_dir: Path) -> pd.DataFrame:
    records: list[dict] = []
    for path in archive_dir.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as source:
                record = _parse_record(path.name, json.load(source))
            if record is not None:
                records.append(record)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    frame["Circular ID"] = pd.to_numeric(frame["Circular ID"], errors="coerce")
    return frame.sort_values("Circular ID", ascending=False, na_position="last")[OUTPUT_COLUMNS]


def _replace_directory(source: Path, destination: Path) -> None:
    backup = destination.with_name(f".{destination.name}.backup")
    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        destination.rename(backup)
    try:
        source.rename(destination)
    except Exception:
        if backup.exists() and not destination.exists():
            backup.rename(destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def refresh_gcn_lat_data(
    *,
    archive_url: str = DEFAULT_ARCHIVE_URL,
    archive_tar_path: Path = DEFAULT_ARCHIVE_TAR,
    archive_dir: Path = DEFAULT_ARCHIVE_DIR,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    downloader: Callable[[str, Path], None] = _download,
) -> LatGcnRefreshResult:
    archive_tar_path = Path(archive_tar_path).expanduser()
    archive_dir = Path(archive_dir).expanduser()
    output_csv = Path(output_csv).expanduser()
    archive_tar_path.parent.mkdir(parents=True, exist_ok=True)
    archive_dir.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="gcn-lat-", dir=str(archive_dir.parent)) as temp_name:
        temp_root = Path(temp_name)
        downloaded_tar = temp_root / "archive.json.tar.gz"
        downloader(archive_url, downloaded_tar)
        extracted_dir = _safe_extract(downloaded_tar, temp_root / "extracted")
        frame = extract_lat_records(extracted_dir)
        temp_csv = temp_root / "lat.csv"
        frame.to_csv(temp_csv, index=False)

        os.replace(downloaded_tar, archive_tar_path)
        _replace_directory(extracted_dir, archive_dir)
        os.replace(temp_csv, output_csv)

    return LatGcnRefreshResult(archive_dir, archive_tar_path, output_csv, len(frame))


def add_extract_arguments(parser) -> None:
    parser.add_argument("--archive-url", default=DEFAULT_ARCHIVE_URL)
    parser.add_argument("--archive-tar", default=str(DEFAULT_ARCHIVE_TAR))
    parser.add_argument("--archive-dir", default=str(DEFAULT_ARCHIVE_DIR))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))


def cli_main_from_args(args) -> LatGcnRefreshResult:
    result = refresh_gcn_lat_data(
        archive_url=args.archive_url,
        archive_tar_path=Path(args.archive_tar),
        archive_dir=Path(args.archive_dir),
        output_csv=Path(args.output_csv),
    )
    print(f"GCN LAT 数据更新完成：{result.record_count} 条，输出 {result.output_csv}")
    return result
