from __future__ import annotations

import argparse
import re
import ssl
import time
from ftplib import FTP_TLS
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

FTP_HOST = "heasarc.gsfc.nasa.gov"
FTP_BASE_PATH = "/fermi/data/gbm/bursts/"
DEFAULT_SAVE_DIR = Path("/home/mxr/lee/gbmtest/GBM_data")
DEFAULT_LAT_CSV = Path("/home/mxr/lee/data/GRB_FermiLAT_filtered.csv")
DEFAULT_GBM_XLS = Path("/home/mxr/lee/data/GBMcatolog.xls")
DEFAULT_JOINT_YEAR_FROM = 2022


def normalize_bnname(value: str) -> str:
    bnname = str(value).strip().lower()
    if not re.fullmatch(r"bn\d{9}", bnname):
        raise ValueError(f"无效的 GRB 名称: {value}")
    return bnname


def extract_year_from_bnname(bnname: str) -> str:
    bnname = normalize_bnname(bnname)
    if len(bnname) < 4 or not bnname.startswith("bn"):
        raise ValueError(f"无效的 GRB 名称: {bnname}")
    yy = bnname[2:4]
    if not yy.isdigit():
        raise ValueError(f"无法从 GRB 名称解析年份: {bnname}")
    return f"20{yy}"


def extract_date_key(value: object) -> Optional[str]:
    match = re.search(r"(\d{6})", str(value).strip().lower())
    return match.group(1) if match else None


def connect_ftp() -> FTP_TLS:
    """Create an anonymous FTPS connection to the HEASARC GBM archive."""
    print(f"正在连接 {FTP_HOST} ...")
    context = ssl.create_default_context()

    ftp = FTP_TLS(host=FTP_HOST, context=context)
    ftp.login()
    ftp.prot_p()
    return ftp


def get_joint_target_list(
    csv_path: Path = DEFAULT_LAT_CSV,
    gbm_xls: Path = DEFAULT_GBM_XLS,
    year_from: int = DEFAULT_JOINT_YEAR_FROM,
    year_to: Optional[int] = None,
) -> List[str]:
    """Read LAT and GBM catalogs and return shared GRB trigger names."""
    df_lat = pd.read_csv(csv_path)
    lat_col = next((c for c in df_lat.columns if "bnname" in c.lower() or "grbname" in c.lower()), None)
    if lat_col is None:
        raise ValueError("LAT CSV 中没有找到 bnname/grbname 列")

    df_gbm = pd.read_excel(gbm_xls, sheet_name="fermigbrst")
    df_gbm.columns = df_gbm.columns.str.strip()
    if "trigger_name" not in df_gbm.columns:
        raise ValueError("GBM catalog 中没有找到 trigger_name 列")

    lat_keys = df_lat[lat_col].map(extract_date_key)
    gbm_keys = df_gbm["trigger_name"].map(extract_date_key)

    df_lat = df_lat.loc[lat_keys.notna()].copy()
    df_gbm = df_gbm.loc[gbm_keys.notna(), ["trigger_name"]].copy()
    df_lat["date_key"] = lat_keys.loc[df_lat.index]
    df_gbm["date_key"] = gbm_keys.loc[df_gbm.index]

    df_merge = pd.merge(df_lat[[lat_col, "date_key"]], df_gbm, on="date_key", how="inner")

    target_list: List[str] = []
    for bn in df_merge["trigger_name"].dropna().astype(str).unique():
        bnname = normalize_bnname(bn)
        try:
            year = int(extract_year_from_bnname(bnname))
        except ValueError:
            continue
        if year < year_from:
            continue
        if year_to is not None and year > year_to:
            continue
        target_list.append(bnname)

    target_list = list(dict.fromkeys(target_list))
    print(f"筛选完成: 找到 {len(target_list)} 个联合分析目标 GRB。")
    return target_list


def load_targets_from_file(list_path: Path) -> List[str]:
    targets: List[str] = []
    with list_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            targets.append(normalize_bnname(line))
    targets = list(dict.fromkeys(targets))
    print(f"从文件读取到 {len(targets)} 个 GRB。")
    return targets


def download_single_burst(ftp: FTP_TLS, bnname: str, save_dir: Path) -> FTP_TLS:
    bnname = normalize_bnname(bnname)
    year = extract_year_from_bnname(bnname)
    remote_dir = f"{FTP_BASE_PATH}{year}/{bnname}/current/"
    local_dir = save_dir / bnname
    local_dir.mkdir(parents=True, exist_ok=True)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            ftp.voidcmd("NOOP")
            ftp.cwd(remote_dir)
            files = ftp.nlst()

            local_existing = {p.name for p in local_dir.iterdir() if p.is_file()}
            for filename in files:
                if filename in local_existing:
                    continue

                local_path = local_dir / filename
                partial_path = local_path.with_name(f"{local_path.name}.part")
                print(f"下载 {bnname}: {filename}")
                try:
                    with partial_path.open("wb") as fh:
                        ftp.retrbinary(f"RETR {filename}", fh.write)
                    partial_path.replace(local_path)
                except Exception:
                    partial_path.unlink(missing_ok=True)
                    raise

            return ftp
        except Exception as exc:  # noqa: BLE001
            if attempt == max_retries - 1:
                print(f"下载失败: {bnname} ({exc})")
                return ftp
            print(f"连接中断 ({exc})，正在进行第 {attempt + 1} 次重连...")
            time.sleep(2)
            try:
                ftp = connect_ftp()
            except Exception as reconnect_exc:  # noqa: BLE001
                print(f"重连失败: {reconnect_exc}")

    return ftp


def download_target_list(burst_list: Iterable[str], save_dir: Path = DEFAULT_SAVE_DIR) -> None:
    burst_list = list(burst_list)
    if not burst_list:
        print("没有可下载的目标 GRB。")
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    ftp: Optional[FTP_TLS] = None
    try:
        ftp = connect_ftp()
        for index, bnname in enumerate(burst_list, start=1):
            print(f"[{index}/{len(burst_list)}] 处理 {bnname}")
            ftp = download_single_burst(ftp, bnname, save_dir)
    finally:
        if ftp is not None:
            try:
                ftp.quit()
                print("FTP 连接已关闭")
            except Exception:
                pass


def add_download_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bnname", type=str, help="指定单个 GRB，例如 bn231129799")
    parser.add_argument("--joint", action="store_true", help="下载可用于 GBM + LAT 联合分析的 GRB 数据")
    parser.add_argument("--list", dest="list_path", type=str, help="从文本文件读取 GRB 列表，每行一个")
    parser.add_argument("--save-dir", type=str, default=str(DEFAULT_SAVE_DIR), help="本地保存目录")
    parser.add_argument("--csv-path", type=str, default=str(DEFAULT_LAT_CSV), help="LAT 过滤列表 CSV 路径")
    parser.add_argument("--gbm-xls", type=str, default=str(DEFAULT_GBM_XLS), help="GBM catalog Excel 路径")
    parser.add_argument("--year-from", type=int, default=DEFAULT_JOINT_YEAR_FROM, help="联合分析目标最早年份")
    parser.add_argument("--year-to", type=int, default=None, help="联合分析目标最晚年份")


def validate_args(args: argparse.Namespace) -> None:
    modes = sum(bool(flag) for flag in (args.bnname, args.list_path, args.joint))
    if modes > 1:
        raise ValueError("--bnname、--list、--joint 只能选择一种模式")


def resolve_targets(args: argparse.Namespace) -> List[str]:
    if args.bnname:
        return [normalize_bnname(args.bnname)]
    if args.list_path:
        return load_targets_from_file(Path(args.list_path))
    if args.joint or (not args.bnname and not args.list_path):
        return get_joint_target_list(
            csv_path=Path(args.csv_path),
            gbm_xls=Path(args.gbm_xls),
            year_from=args.year_from,
            year_to=args.year_to,
        )
    return []


def cli_main_from_args(args: argparse.Namespace) -> None:
    validate_args(args)
    targets = resolve_targets(args)
    if targets:
        download_target_list(targets, save_dir=Path(args.save_dir))
    print("所有 GBM 下载任务处理完毕。")
