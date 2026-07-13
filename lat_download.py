# -*- coding: utf-8 -*-
"""Download Fermi-LAT Extended data through the Fermi LAT Query service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import time
from typing import Callable, Iterable, Optional, Sequence

import pandas as pd
import requests


DEFAULT_CATALOG_XLS = Path("/home/mxr/lee/fermilat-grb.xls")
DEFAULT_DATA_ROOT = Path("/home/mxr/lee/data/fermilat/Extended_data_ex")
DEFAULT_RADIUS_DEG = 12.0
DEFAULT_PAD_BEFORE_S = 1000.0
DEFAULT_PAD_AFTER_S = 2000.0
DEFAULT_ENERGY_MIN_MEV = 100.0
DEFAULT_ENERGY_MAX_MEV = 100000.0
LAT_QUERY_URL = "https://fermi.gsfc.nasa.gov/cgi-bin/ssc/LAT/LATDataQuery.cgi"
QUERY_RESULT_PATTERN = re.compile(
    r"https://fermi\.gsfc\.nasa\.gov/cgi-bin/ssc/LAT/QueryResults\.cgi\?id=L[\dA-F]+"
)
FITS_URL_PATTERN = re.compile(
    r"https://fermi\.gsfc\.nasa\.gov/FTP/fermi/data/lat/queries/[^\s\"<>]+\.fits"
)


def _request_with_retry(
    request: Callable,
    *,
    attempts: int,
    sleep: Callable[[float], None],
    session,
    **kwargs,
):
    if attempts < 1:
        raise ValueError("request_retries 必须至少为 1")
    for attempt in range(attempts):
        try:
            response = request(**kwargs)
            response.raise_for_status()
            return response
        except (requests.ConnectionError, requests.Timeout):
            if attempt + 1 == attempts:
                raise
            session.close()
            sleep(min(2 ** attempt, 10))
    raise RuntimeError("HTTP 重试循环异常退出")


@dataclass(frozen=True)
class LatDownloadTarget:
    bnname: str
    grb_name: str
    trigger_met: float
    t0: float
    t1: float
    ra: float
    dec: float


@dataclass(frozen=True)
class LatDownloadResult:
    target: LatDownloadTarget
    destination: Path
    tstart: int
    tstop: int


def normalize_bnname(value: str) -> str:
    text = str(value).strip().lower()
    if not text.startswith("bn") or not text[2:].isdigit():
        raise ValueError(f"无效 bnname: {value}")
    return text


def _target_from_row(row: pd.Series) -> LatDownloadTarget:
    bnname = normalize_bnname(row["trigname"])
    grb_name = str(row["gcn_name"]).replace(" ", "").strip()
    if not grb_name:
        raise ValueError(f"{bnname}: gcn_name 为空")
    position = [part.strip() for part in str(row["ra,dec"]).split(",")]
    if len(position) != 2:
        raise ValueError(f"{bnname}: ra,dec 格式无效")
    return LatDownloadTarget(
        bnname=bnname,
        grb_name=grb_name,
        trigger_met=float(row["trigger_met"]),
        t0=float(row["T0"]),
        t1=float(row["T1"]),
        ra=float(position[0]),
        dec=float(position[1]),
    )


def select_targets(
    frame: pd.DataFrame,
    *,
    bnnames: Optional[Sequence[str]] = None,
    year: Optional[int] = None,
    month_from: int = 1,
) -> list[LatDownloadTarget]:
    required = {"trigname", "gcn_name", "trigger_met", "ra,dec", "T0", "T1"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("LAT catalog 缺少列: " + ", ".join(missing))

    rows = frame.dropna(subset=list(required)).copy()
    rows["_bnname"] = rows["trigname"].map(normalize_bnname)
    if bnnames:
        requested = [normalize_bnname(value) for value in bnnames]
        by_name = rows.set_index("_bnname", drop=False)
        unknown = [value for value in requested if value not in by_name.index]
        if unknown:
            raise ValueError("LAT catalog 中找不到: " + ", ".join(unknown))
        return [_target_from_row(by_name.loc[value]) for value in dict.fromkeys(requested)]

    if year is None:
        raise ValueError("批量下载必须指定 year")
    if not 1 <= int(month_from) <= 12:
        raise ValueError("month_from 必须在 1-12 之间")
    year_two_digits = int(year) % 100
    selected = rows[
        rows["_bnname"].map(
            lambda value: int(value[2:4]) == year_two_digits and int(value[4:6]) >= int(month_from)
        )
    ]
    return [_target_from_row(row) for _, row in selected.iterrows()]


def load_targets(
    catalog_xls: Path = DEFAULT_CATALOG_XLS,
    *,
    bnnames: Optional[Sequence[str]] = None,
    year: Optional[int] = None,
    month_from: int = 1,
) -> list[LatDownloadTarget]:
    frame = pd.read_excel(Path(catalog_xls).expanduser(), sheet_name="GCN")
    return select_targets(frame, bnnames=bnnames, year=year, month_from=month_from)


def download_lat_data_http(
    *,
    ra: float,
    dec: float,
    radius: float,
    tstart: int,
    tstop: int,
    destination: Path,
    energy_min: float = DEFAULT_ENERGY_MIN_MEV,
    energy_max: float = DEFAULT_ENERGY_MAX_MEV,
    session=None,
    sleep: Callable[[float], None] = time.sleep,
    poll_interval: float = 15.0,
    max_polls: int = 40,
    request_retries: int = 3,
) -> list[Path]:
    http = session or requests.Session()
    headers = {"User-Agent": "grb-project/1.0"}
    payload = {
        "coordfield": f"{float(ra):g},{float(dec):g}",
        "coordsystem": "J2000",
        "shapefield": f"{float(radius):g}",
        "radius": f"{float(radius):g}",
        "timefield": f"{int(tstart)},{int(tstop)}",
        "timetype": "MET",
        "energyfield": f"{float(energy_min):g},{float(energy_max):g}",
        "photonOrExtendedOrNone": "Extended",
        "spacecraft": "on",
        "destination": "query",
        "submit": "Start Search",
    }
    response = _request_with_retry(
        http.post,
        attempts=request_retries,
        sleep=sleep,
        session=http,
        url=LAT_QUERY_URL,
        data=payload,
        headers=headers,
        timeout=60,
    )
    match = QUERY_RESULT_PATTERN.search(response.text)
    if match is None:
        raise RuntimeError("Fermi LAT 查询响应中没有结果页面链接")
    result_url = match.group(0)

    file_urls: list[str] = []
    for poll_index in range(max_polls):
        if poll_index:
            sleep(poll_interval)
        result = _request_with_retry(
            http.get,
            attempts=request_retries,
            sleep=sleep,
            session=http,
            url=result_url,
            headers=headers,
            timeout=60,
        )
        file_urls = list(dict.fromkeys(FITS_URL_PATTERN.findall(result.text)))
        if file_urls:
            break
    if not file_urls:
        raise TimeoutError(f"Fermi LAT 查询在 {max_polls} 次轮询后仍未完成: {result_url}")

    destination = Path(destination).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for file_url in file_urls:
        path = destination / Path(file_url).name
        download = _request_with_retry(
            http.get,
            attempts=request_retries,
            sleep=sleep,
            session=http,
            url=file_url,
            headers=headers,
            stream=True,
            timeout=120,
        )
        with path.open("wb") as output:
            for chunk in download.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
        paths.append(path)
    return paths


def download_target(
    target: LatDownloadTarget,
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    radius: float = DEFAULT_RADIUS_DEG,
    pad_before: float = DEFAULT_PAD_BEFORE_S,
    pad_after: float = DEFAULT_PAD_AFTER_S,
    downloader: Callable = download_lat_data_http,
) -> LatDownloadResult:
    destination = Path(data_root).expanduser() / target.grb_name
    destination.mkdir(parents=True, exist_ok=True)
    tstart = int(target.trigger_met + target.t0 - float(pad_before))
    tstop = int(target.trigger_met + target.t1 + float(pad_after))
    downloader(
        ra=target.ra,
        dec=target.dec,
        radius=float(radius),
        tstart=tstart,
        tstop=tstop,
        destination=destination,
    )
    return LatDownloadResult(target=target, destination=destination, tstart=tstart, tstop=tstop)


def download_targets(
    targets: Iterable[LatDownloadTarget],
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    downloader: Callable = download_lat_data_http,
) -> list[LatDownloadResult]:
    return [download_target(target, data_root=data_root, downloader=downloader) for target in targets]


def run_download(
    *,
    bnnames: Optional[Sequence[str]],
    year: Optional[int],
    month_from: int,
    catalog_xls: Path = DEFAULT_CATALOG_XLS,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> list[LatDownloadResult]:
    targets = load_targets(catalog_xls, bnnames=bnnames, year=year, month_from=month_from)
    if not targets:
        raise ValueError("没有符合条件的 LAT 下载目标")
    return download_targets(targets, data_root=data_root)


def add_download_arguments(parser) -> None:
    parser.add_argument("--bnname", action="append", help="下载单个目标，可重复传入")
    parser.add_argument("--year", type=int, default=None, help="批量下载年份，例如 2025")
    parser.add_argument("--month-from", type=int, default=1, help="批量下载起始月份")
    parser.add_argument("--catalog-xls", default=str(DEFAULT_CATALOG_XLS))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))


def cli_main_from_args(args) -> list[LatDownloadResult]:
    results = run_download(
        bnnames=args.bnname,
        year=args.year,
        month_from=args.month_from,
        catalog_xls=Path(args.catalog_xls),
        data_root=Path(args.data_root),
    )
    for result in results:
        print(
            f"LAT 下载完成: {result.target.bnname} -> {result.destination} "
            f"(MET {result.tstart}-{result.tstop})"
        )
    return results
