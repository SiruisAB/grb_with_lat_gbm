#!/usr/bin/env python3
"""GBM burst data bulk downloader (HTTPS + multi-threading).

Optimized version of gbm_download.py:
- HTTPS instead of single FTP_TLS connection, 12 worker threads (~6x faster)
- File filtering: keep glg_tte/trigdat/bcat/tcat .fit, .pha, .rsp/.rsp2/.rsp3,
  glg_lc_tot_*.pdf; drop glg_scat_* (band/comp/plaw/sbpl), .gif/.png, other pdfs
- Year-range bulk mode: --year-from/--year-to
- Resumable: .part atomic writes + per-burst .complete markers
- Disk space guard per target mount

Usage:
  python3 gbm_bulk_download.py --bnname bn080714086          # single burst test
  python3 gbm_bulk_download.py --year-from 2008 --year-to 2017
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/bursts"
SAVE_ROOT = Path("/home/mxr/gbm_bursts")

# year -> real storage dir (years not listed here are stored under SAVE_ROOT)
YEAR_DIR_MAP = {
    2008: Path("/home/data1/gbm_bursts"),
    2009: Path("/home/data1/gbm_bursts"),
    2010: Path("/home/data2/gbm_bursts"),
    2011: Path("/home/data2/gbm_bursts"),
}

WORKERS = 12
RETRIES = 3
TIMEOUT_FETCH = 60       # listing pages
TIMEOUT_FILE = 300       # data files
MIN_FREE_GB = 5.0        # abort burst if target mount has less free space

HEADERS = {"User-Agent": "Mozilla/5.0 (gbm-bulk-downloader/1.0)"}

ROW_RE = re.compile(
    r'<a href="([^"?/][^"]*)">[^<]+</a>\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s+[\d.]+[KMG]?'
)
BURST_RE = re.compile(r'bn\d{9}')

# keep: detector/data fits, pha, rsp family, total lightcurve pdf
KEEP_RE = re.compile(
    r'glg_(tte|trigdat|bcat|tcat)_.*\.fit'
    r'|.*\.pha'
    r'|.*\.rsp\d*'
    r'|glg_lc_tot_.*\.pdf'
)

_progress_lock = threading.Lock()
_stats = {"bursts_done": 0, "bursts_skip": 0, "files_ok": 0, "files_skip": 0, "files_fail": 0}


def log(msg: str) -> None:
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def urlopen_retry(url: str, timeout: int, binary: bool):
    last_exc = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            resp = urllib.request.urlopen(req, timeout=timeout)
            if binary:
                return resp  # caller closes
            with resp:
                return resp.read().decode("utf-8", "replace")
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"fetch failed after {RETRIES} tries: {url} ({last_exc})")


def list_year(year: int) -> list[str]:
    html = urlopen_retry(f"{BASE}/{year}/", TIMEOUT_FETCH, binary=False)
    return sorted({m.group(0) for m in BURST_RE.finditer(html) if len(m.group(0)) == 11})


def list_current_files(year: int, bnname: str) -> list[str]:
    html = urlopen_retry(f"{BASE}/{year}/{bnname}/current/", TIMEOUT_FETCH, binary=False)
    return [m.group(1) for m in ROW_RE.finditer(html)]


def keep_file(name: str) -> bool:
    return bool(KEEP_RE.fullmatch(name))


def burst_dest(year: int) -> Path:
    base = YEAR_DIR_MAP.get(year, SAVE_ROOT)
    return base / str(year)


def free_gb(path: Path) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / (1024 ** 3)


def download_file(url: str, dest: Path) -> str:
    if dest.exists():
        return "skip"
    part = dest.with_name(dest.name + ".part")
    for attempt in range(RETRIES):
        try:
            resp = urlopen_retry(url, TIMEOUT_FILE, binary=True)
            try:
                with resp, open(part, "wb") as fh:
                    while True:
                        chunk = resp.read(256 * 1024)
                        if not chunk:
                            break
                        fh.write(chunk)
            finally:
                pass
            part.replace(dest)
            return "ok"
        except Exception:  # noqa: BLE001
            if part.exists():
                part.unlink()
            time.sleep(2 * (attempt + 1))
    part.unlink(missing_ok=True)
    return "fail"


def download_burst(year: int, bnname: str) -> str:
    dest_dir = burst_dest(year) / bnname
    marker = dest_dir / ".complete"
    if marker.exists():
        with _progress_lock:
            _stats["bursts_skip"] += 1
        return "skip"

    dest_dir.mkdir(parents=True, exist_ok=True)
    if free_gb(dest_dir) < MIN_FREE_GB:
        raise RuntimeError(f"disk almost full on {dest_dir.anchor}")

    files = list_current_files(year, bnname)
    kept = [f for f in files if keep_file(f)]
    failures: list[str] = []
    for name in kept:
        url = f"{BASE}/{year}/{bnname}/current/{name}"
        status = download_file(url, dest_dir / name)
        with _progress_lock:
            if status == "ok":
                _stats["files_ok"] += 1
            elif status == "skip":
                _stats["files_skip"] += 1
            else:
                _stats["files_fail"] += 1
                failures.append(name)
    if failures:
        raise RuntimeError(f"{bnname}: {len(failures)} files failed: {failures[:3]}")
    marker.touch()
    return "ok"


def run(year_from: int, year_to: int, only_bnname: str | None, workers: int) -> None:
    tasks: list[tuple[int, str]] = []
    if only_bnname:
        year = int("20" + only_bnname[2:4])
        tasks = [(year, only_bnname)]
    else:
        for year in range(year_from, year_to + 1):
            bursts = list_year(year)
            log(f"{year}: {len(bursts)} bursts")
            tasks.extend((year, bn) for bn in bursts)

    log(f"total bursts: {len(tasks)}, workers: {workers}")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(download_burst, y, bn): (y, bn) for y, bn in tasks}
        for fut in as_completed(futs):
            y, bn = futs[fut]
            try:
                result = fut.result()
                with _progress_lock:
                    _stats["bursts_done"] += 1
                    done, total = _stats["bursts_done"] + _stats["bursts_skip"], len(tasks)
                    rate = _stats["files_ok"] / max(time.time() - t0, 1)
                if result == "skip":
                    log(f"[{done}/{total}] {bn} already complete, skipped")
                else:
                    log(f"[{done}/{total}] {bn} done ({rate:.1f} files/s)")
            except Exception as exc:  # noqa: BLE001
                with _progress_lock:
                    _stats["files_fail"] += 1
                log(f"ERROR {y}/{bn}: {exc}")

    log(f"finished: {_stats} in {(time.time() - t0) / 3600:.1f} h")


def main() -> None:
    ap = argparse.ArgumentParser(description="GBM bulk downloader")
    ap.add_argument("--year-from", type=int, default=2008)
    ap.add_argument("--year-to", type=int, default=2017)
    ap.add_argument("--bnname", type=str, default=None, help="download a single burst, e.g. bn080714086")
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()
    if args.bnname and not re.fullmatch(r"bn\d{9}", args.bnname):
        sys.exit("invalid bnname")
    run(args.year_from, args.year_to, args.bnname, args.workers)


if __name__ == "__main__":
    main()
