#!/bin/bash
# 批次结束后的收尾：把从磁盘产物重建出来的行合并进 summary_results.csv。
#
# 由 cron 定期调用。判定"批次已结束"的两个条件同时成立才动手：
#   1) 状态文件里的 pid 已不是活进程（僵尸 Z 也算结束）
#   2) 日志在 STALE_SEC 秒内没有新写入（避免刚 finished 但收尾还在写）
# 合并是幂等的，重复触发无副作用；成功后把 NOTIFY 标记文件写出来。
set -eo pipefail

ROOT=/home/mxr/lee/gbmtest
RESULT_ROOT=$ROOT/results_joint_batch
ENV=/home/mxr/miniconda3/envs/threeML
STATE=$ROOT/joint_batch_run.json
SUMMARY=$RESULT_ROOT/summary_results.csv
LOG=$RESULT_ROOT/joint_batch_run.log
RECON=$RESULT_ROOT/summary_results.reconstructed.csv
NOTIFY=$RESULT_ROOT/.merge_done
WATCHER_LOG=$RESULT_ROOT/merge_watcher.log

STALE_SEC=${STALE_SEC:-900}
# 已经合并过就不重复做
[ -f "$NOTIFY" ] && exit 0

log() { echo "[$(date '+%F %T')] $*" >> "$WATCHER_LOG"; }

# --- 条件 1: pid 是否还活着 ---
pid=$(sed -n 's/.*"pid": *\([0-9]*\).*/\1/p' "$STATE" 2>/dev/null | head -1)
if [ -z "$pid" ]; then
    log "状态文件里没有 pid，跳过"
    exit 0
fi
if [ -d "/proc/$pid" ]; then
    state=$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null || echo "?")
    if [ "$state" != "Z" ]; then
        exit 0    # 仍在运行，什么都不做（不写日志避免刷屏）
    fi
fi

# --- 条件 2: 日志是否已经静默 ---
if [ -f "$LOG" ]; then
    now=$(date +%s)
    mtime=$(stat -c %Y "$LOG")
    if [ $((now - mtime)) -lt "$STALE_SEC" ]; then
        exit 0    # 还在写，再等等
    fi
fi

log "批次已结束（pid=$pid 非活动，日志静默 >${STALE_SEC}s），开始合并"

export CALDB=$ENV/share/fermitools/data/caldb
export CALDBCONFIG=$CALDB/software/tools/caldb.config
export CALDBALIAS=$CALDB/software/tools/alias_config.fits
export PFILES=$ENV/share/fermitools/syspfiles
export OMP_NUM_THREADS=1
export PYTHONPATH=$ROOT

cd "$ROOT"

# 备份当前汇总（合并前的状态）
[ -f "$SUMMARY" ] && cp -f "$SUMMARY" "$SUMMARY.before_merge" || true

if ! "$ENV/bin/python" grb_project/reconstruct_summary.py \
        --result-root "$RESULT_ROOT" --output "$RECON" >> "$WATCHER_LOG" 2>&1; then
    log "重建失败，保留原汇总不动"
    exit 1
fi

if ! "$ENV/bin/python" grb_project/merge_summary.py \
        --batch-csv "$SUMMARY" --recon-csv "$RECON" --output "$SUMMARY" \
        >> "$WATCHER_LOG" 2>&1; then
    log "合并失败，从备份恢复"
    [ -f "$SUMMARY.before_merge" ] && cp -f "$SUMMARY.before_merge" "$SUMMARY" || true
    exit 1
fi

{
    echo "合并完成于 $(date '+%F %T')"
    echo "汇总: $SUMMARY"
    echo "重建明细: $RECON"
} > "$NOTIFY"
log "合并完成，已写标记 $NOTIFY"
