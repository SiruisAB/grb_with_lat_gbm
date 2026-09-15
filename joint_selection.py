# -*- coding: utf-8 -*-
"""按数据实况与物理判据一键选择可做 GBM+LAT 联合分析的 GRB 目标。

数据源是 lat_download_targets.csv 交叉表（GBM 目录 × 2FLGC × LLE × GCN 通报，
2026-09 建立），选择时对 GBM_data 与 Extended_data_ex 做磁盘复核，不盲信表内
标志位；每个排除原因逐项记录，供 CLI/Web 展示与复查。本模块只依赖 pandas，
不依赖 threeML，保证任何环境可导入、可单测。
"""

from __future__ import annotations

import json
import os
import re
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from .session import session

BNNAME_RE = re.compile(r"^bn\d{9}$")

REQUIRED_TABLE_COLUMNS = (
    "bnname",
    "grb_name",
    "gbm_held",
    "lat_downloaded",
    "trigger_met",
    "ra",
    "dec",
    "T0",
    "T1",
)

LAT_EVENT_PATTERNS = ("*_EV*.fits", "*_PH*.fits", "*FT1*.fits")
LAT_SC_PATTERN = "*_SC*.fits"


@dataclass(frozen=True)
class JointSelectionCriteria:
    """联合目标选择判据。"""

    year_from: Optional[int] = None
    year_to: Optional[int] = None
    min_lat_ts: Optional[float] = None
    include_lle_only: bool = False
    analysis_mode: str = "gbm+lat"
    verify_on_disk: bool = True


@dataclass(frozen=True)
class JointTarget:
    bnname: str
    grb_name: str
    year: int
    trigger_met: float
    t0: float
    t1: float
    ra: float
    dec: float
    window_source: str
    lat_ts: Optional[float]


@dataclass
class JointSelectionResult:
    criteria: JointSelectionCriteria
    targets: list = field(default_factory=list)
    excluded: dict = field(default_factory=dict)
    table_path: Optional[str] = None

    @property
    def bnnames(self) -> list:
        return [t.bnname for t in self.targets]

    @property
    def grb_names(self) -> list:
        return [t.grb_name for t in self.targets]

    def excluded_total(self) -> int:
        return sum(len(v) for v in self.excluded.values())

    def per_year_counts(self) -> dict:
        counts: dict = {}
        for t in self.targets:
            counts[t.year] = counts.get(t.year, 0) + 1
        return dict(sorted(counts.items()))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "bnname": t.bnname,
                    "grb_name": t.grb_name,
                    "year": t.year,
                    "trigger_met": t.trigger_met,
                    "T0": t.t0,
                    "T1": t.t1,
                    "ra": t.ra,
                    "dec": t.dec,
                    "window_source": t.window_source,
                    "lat_ts": t.lat_ts,
                }
                for t in self.targets
            ]
        )


def _truthy(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in ("true", "1", "yes")


def _numeric(value) -> Optional[float]:
    v = pd.to_numeric(value, errors="coerce")
    if pd.isna(v):
        return None
    return float(v)


def _gbm_dir_ready(bnname: str, data_dir: Path) -> bool:
    burst_dir = data_dir / bnname
    return burst_dir.is_dir() and any(burst_dir.glob("glg_tte*.fit"))


def _lat_dir_ready(grb_name: str, lat_root: Path) -> bool:
    burst_dir = lat_root / grb_name
    if not burst_dir.is_dir():
        return False
    has_events = any(any(burst_dir.glob(p)) for p in LAT_EVENT_PATTERNS)
    has_sc = any(burst_dir.glob(LAT_SC_PATTERN))
    return has_events and has_sc


def load_joint_target_table(path: Optional[str] = None) -> pd.DataFrame:
    """读取联合目标交叉表；缺必需列时报出具体列名。"""
    csv_path = Path(path or session.joint_target_csv).expanduser()
    table = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_TABLE_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f"联合目标交叉表缺列: {', '.join(missing)} ({csv_path})")
    return table


def select_joint_targets(
    criteria: Optional[JointSelectionCriteria] = None,
    *,
    table: Optional[pd.DataFrame] = None,
    data_dir=None,
    lat_root=None,
) -> JointSelectionResult:
    """按判据顺序筛选；每步的排除原因归组记录，先到先得。"""
    criteria = criteria or JointSelectionCriteria()
    if "lat" not in criteria.analysis_mode:
        raise ValueError(
            "联合目标选择的宇宙是 LAT 候选暴，analysis_mode 只接受 gbm+lat / lat；"
            "gbm-only 请走 GBM 目录表流程（--gcn-all 等）。"
        )
    if table is None:
        table = load_joint_target_table()
    gbm_root = Path(data_dir if data_dir is not None else session.data_dir)
    lat_root_path = Path(lat_root if lat_root is not None else session.extended_lat_data_root)
    need_gbm = "gbm" in criteria.analysis_mode

    excluded: dict = {}
    targets: list = []

    def drop(bnname: str, reason: str) -> None:
        excluded.setdefault(reason, []).append(bnname)

    for _, row in table.iterrows():
        raw_bn = str(row.get("bnname", "")).strip()
        bn = raw_bn.lower()
        if not BNNAME_RE.match(bn):
            drop(raw_bn or "<空 bnname>", "bnname 非法")
            continue
        grb_name = str(row.get("grb_name", "")).strip()

        # 1. LAT 对应体有效性（撤回的探测不算）
        if not _truthy(row.get("lat_grb_valid"), default=True):
            drop(bn, "LAT 对应体已撤回")
            continue
        # 2. LLE-only：无标准 >100 MeV 似然探测，默认不进联合样本
        lle_only = _truthy(row.get("requires_lle_data")) or _truthy(row.get("lat_only_lle"))
        if lle_only and not criteria.include_lle_only:
            drop(bn, "LLE-only 探测")
            continue
        # 3. GBM 数据：标志位 + 磁盘复核
        if need_gbm:
            if not _truthy(row.get("gbm_held")):
                drop(bn, "本地无 GBM 数据")
                continue
            if criteria.verify_on_disk and not _gbm_dir_ready(bn, gbm_root):
                drop(bn, "GBM 目录缺失或无 TTE")
                continue
        # 4. LAT 数据：标志位 + 磁盘复核（事件 + SC 文件对）
        if not grb_name:
            drop(bn, "grb_name 为空")
            continue
        if not _truthy(row.get("lat_downloaded")):
            drop(bn, "LAT 数据未下载")
            continue
        if criteria.verify_on_disk and not _lat_dir_ready(grb_name, lat_root_path):
            drop(bn, "LAT 目录缺失或不完整")
            continue
        # 5. 参数完备性
        met = _numeric(row.get("trigger_met"))
        t0 = _numeric(row.get("T0"))
        t1 = _numeric(row.get("T1"))
        ra = _numeric(row.get("ra"))
        dec = _numeric(row.get("dec"))
        if None in (met, t0, t1, ra, dec) or t1 <= t0:
            drop(bn, "参数不完备（MET/RA/Dec/T0/T1）")
            continue
        # 6. 年份区间
        year = 2000 + int(bn[2:4])
        if criteria.year_from is not None and year < criteria.year_from:
            drop(bn, f"年份 < {criteria.year_from}")
            continue
        if criteria.year_to is not None and year > criteria.year_to:
            drop(bn, f"年份 > {criteria.year_to}")
            continue
        # 7. LAT 显著性
        lat_ts = _numeric(row.get("flgc_lat_ts"))
        if criteria.min_lat_ts is not None and (lat_ts is None or lat_ts < criteria.min_lat_ts):
            drop(bn, f"LAT TS < {criteria.min_lat_ts:g}")
            continue

        targets.append(
            JointTarget(
                bnname=bn,
                grb_name=grb_name,
                year=year,
                trigger_met=met,
                t0=t0,
                t1=t1,
                ra=ra,
                dec=dec,
                window_source=str(row.get("window_source", "") or ""),
                lat_ts=lat_ts,
            )
        )

    targets.sort(key=lambda t: t.bnname)
    return JointSelectionResult(
        criteria=criteria, targets=targets, excluded=excluded
    )


def format_selection_report(result: JointSelectionResult, max_listed: int = 20) -> str:
    years = result.per_year_counts()
    lines = [
        f"联合目标选择：可选 {len(result.targets)} 个 / 排除 {result.excluded_total()} 个"
        f"（analysis_mode={result.criteria.analysis_mode}）"
    ]
    if years:
        lines.append("逐年：" + "  ".join(f"{y}:{n}" for y, n in years.items()))
    for reason in sorted(result.excluded):
        lines.append(f"  排除 {reason}: {len(result.excluded[reason])}")
    shown = result.bnnames[:max_listed]
    suffix = " …" if len(result.bnnames) > max_listed else ""
    lines.append("目标：" + " ".join(shown) + suffix)
    return "\n".join(lines)


def restrict_result_targets(
    result: JointSelectionResult, only: Sequence[str]
) -> JointSelectionResult:
    """把选择结果收窄到指定 bn 子集（只能收窄，不能引入未选中的目标）。"""
    known = {t.bnname for t in result.targets}
    wanted = [str(b).strip().lower() for b in dict.fromkeys(only)]
    unknown = [b for b in wanted if b not in known]
    if unknown:
        raise ValueError("以下 bn 不在当前选择结果中: " + ", ".join(unknown))
    keep = set(wanted)
    return JointSelectionResult(
        criteria=result.criteria,
        targets=[t for t in result.targets if t.bnname in keep],
        excluded=result.excluded,
        table_path=result.table_path,
    )


def joint_batch_state_path() -> Path:
    return Path(session.joint_batch_run_state_file).expanduser()


def joint_batch_stop_flag_path() -> Path:
    return Path(session.stop_flag_file).expanduser()


def read_running_batch() -> Optional[dict]:
    path = joint_batch_state_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def batch_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # kill(pid, 0) 对僵尸进程同样成功（父进程未回收时以 Z 状态留在进程表），
    # 读 /proc/<pid>/stat 的状态位，Z 视为已结束。
    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        state = stat_text.rsplit(")", 1)[1].split()[0]
        return state != "Z"
    except (OSError, IndexError):
        return True


def clear_batch_stop_flag() -> None:
    joint_batch_stop_flag_path().unlink(missing_ok=True)


def request_batch_stop() -> dict:
    """优雅停止：写 stop flag，批量进程完成当前暴后自行退出。"""
    state = read_running_batch()
    if not state or not batch_pid_alive(int(state.get("pid", 0))):
        raise RuntimeError("没有运行中的批量任务")
    joint_batch_stop_flag_path().write_text(
        json.dumps(
            {"requested": time.strftime("%Y-%m-%d %H:%M:%S"), "pid": state["pid"]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    state["stop_requested"] = time.strftime("%Y-%m-%d %H:%M:%S")
    joint_batch_state_path().write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return state


def force_stop_batch() -> dict:
    """强制停止：SIGKILL。

    仅当目标本身是进程组长（Web 启动器用 start_new_session 保证）才杀整组，
    否则只杀单个进程——避免误杀手工启动批次所在终端的整个进程组。
    """
    state = read_running_batch()
    if not state:
        raise RuntimeError("没有批量任务记录")
    pid = int(state.get("pid", 0))
    killed = False
    if pid and batch_pid_alive(pid):
        try:
            if os.getpgid(pid) == pid:
                os.killpg(pid, signal.SIGKILL)
            else:
                os.kill(pid, signal.SIGKILL)
            killed = True
        except OSError as exc:
            raise RuntimeError(f"强制停止失败（PID {pid}）: {exc}") from exc
    clear_batch_stop_flag()
    # 进程已结束时不得谎报"已强制停止"——只如实记录真正发生的 kill
    if killed:
        state["stop_requested"] = None
        state["force_stopped"] = time.strftime("%Y-%m-%d %H:%M:%S")
    else:
        state.pop("force_stopped", None)
    state["kill_attempted"] = killed
    joint_batch_state_path().write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return state


def add_selection_arguments(parser) -> None:
    parser.add_argument("--year-from", type=int, default=None, help="起始年份（含）")
    parser.add_argument("--year-to", type=int, default=None, help="截止年份（含）")
    parser.add_argument("--min-lat-ts", type=float, default=None, help="2FLGC LAT TS 下限")
    parser.add_argument("--include-lle-only", action="store_true", help="纳入 LLE-only 探测的暴")
    parser.add_argument(
        "--analysis-mode", choices=("gbm+lat", "lat"), default="gbm+lat", help="联合分析模式"
    )
    parser.add_argument(
        "--table", default=None, help="交叉表 CSV 路径（默认 session.joint_target_csv）"
    )
    parser.add_argument(
        "--no-verify-on-disk", action="store_true", help="跳过磁盘复核，只信交叉表标志位"
    )
    parser.add_argument("--output", default=None, help="选中目标 CSV 输出路径")
    parser.add_argument("--grbs-file", default=None, help="bn 名单输出路径（每行一个）")
    parser.add_argument("--run", action="store_true", help="把选中目标直接交给批量联合分析")
    parser.add_argument("--result-root", default=None)
    parser.add_argument("--summary-csv-name", default=None)
    parser.add_argument("--session-log", action="store_true")
    parser.add_argument("--models", nargs="+", default=None, help="透传给分析的模型列表")
    parser.add_argument(
        "--only", nargs="+", default=None, help="仅保留选中结果里的这些 bn（子集收窄）"
    )
    parser.add_argument(
        "--stop-running",
        action="store_true",
        help="请求运行中的批量分析优雅停止（完成当前暴后退出）",
    )
    parser.add_argument(
        "--force-stop-running",
        action="store_true",
        help="强制停止运行中的批量分析（SIGKILL 进程组）",
    )
    parser.add_argument(
        "--lat-extended-three-ml",
        action="store_true",
        help="为每个暴跑 LAT Extended + GtBurst + threeML 全流程（非常耗时）",
    )


def cli_main_from_args(args):
    if args.stop_running or args.force_stop_running:
        try:
            if args.force_stop_running:
                state = force_stop_batch()
                if state.get("kill_attempted"):
                    print(f"已强制停止批量任务（PID {state.get('pid')}）")
                else:
                    print(f"批量任务（PID {state.get('pid')}）已结束，无需强制停止")
            else:
                state = request_batch_stop()
                print(f"已请求停止批量任务（PID {state.get('pid')}），完成当前暴后退出")
        except RuntimeError as exc:
            print(str(exc))
            return 1
        return 0

    criteria = JointSelectionCriteria(
        year_from=args.year_from,
        year_to=args.year_to,
        min_lat_ts=args.min_lat_ts,
        include_lle_only=args.include_lle_only,
        analysis_mode=args.analysis_mode,
        verify_on_disk=not args.no_verify_on_disk,
    )
    table = load_joint_target_table(args.table) if args.table else None
    result = select_joint_targets(criteria, table=table)
    if args.only:
        result = restrict_result_targets(result, args.only)
    print(format_selection_report(result))

    if args.output:
        result.to_frame().to_csv(args.output, index=False)
        print(f"已写出选中表: {args.output}")
    if args.grbs_file:
        Path(args.grbs_file).write_text("\n".join(result.bnnames) + "\n", encoding="utf-8")
        print(f"已写出 bn 名单: {args.grbs_file}")

    if not args.run:
        return 0
    if not result.targets:
        print("没有选中任何目标，--run 不启动分析")
        return 1

    from .config import GRBRunOverrides
    from .pipeline import main as pipeline_main

    clear_batch_stop_flag()

    overrides = GRBRunOverrides(
        models=list(args.models) if args.models else None,
        lat_three_ml_full=True if getattr(args, "lat_extended_three_ml", False) else None,
    )
    if overrides.is_empty():
        overrides = None
    pipeline_main(
        target_grbs=result.bnnames,
        analysis_mode=args.analysis_mode,
        result_root=args.result_root,
        summary_csv_name=args.summary_csv_name,
        session_log=args.session_log,
        run_overrides=overrides,
    )
    return 0
