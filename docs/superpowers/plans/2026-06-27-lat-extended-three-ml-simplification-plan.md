# LAT Extended threeML 精简重构实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在不改变用户可见分析结果的前提下，精简 `lat_extended_three_ml.py` 的主流程，删除重复与低价值分支，提升可读性、可维护性和后续扩展能力。

**架构：** 将单个超长 worker 拆成“时间窗准备 / LAT 事件与建库 / 图像输出 / 拟合执行 / 结果汇总”五个清晰阶段。把重复的绘图、三ML 模型初始化、结果写盘逻辑抽成小函数，统一时间窗与分析段的派生规则，并保留当前主要输出文件与结果字段，避免影响下游脚本。

**技术栈：** Python、NumPy、Pandas、Astropy、threeML、GtBurst、matplotlib。

---

### 任务 1：梳理并固定主流程边界

**文件：**
- 修改：`/home/mxr/lee/gbmtest/grb_project/lat_extended_three_ml.py`

- [ ] **步骤 1：把主流程拆成明确阶段的函数骨架**

在同一文件中新增以下私有辅助函数签名，并让主 worker 只负责串联调用：

```python
def _build_analysis_window(
    selection: Dict[str, Any],
    analysis_bin_start: Optional[float],
    analysis_bin_end: Optional[float],
) -> Dict[str, float]:
    ...


def _build_analysis_segments(
    t0_core: float,
    t1_core: float,
    t95: float,
    analysis_bin_start: Optional[float],
    analysis_bin_end: Optional[float],
) -> list[dict]:
    ...


def _log_and_save_window_stats(
    event_times: np.ndarray,
    energies: np.ndarray,
    t0_analysis: float,
    t1_analysis: float,
) -> Dict[str, Any]:
    ...
```

预期：主 worker 里不再直接散落 `t0/t1/t95/analysis_segments` 的派生逻辑。

- [ ] **步骤 2：运行最小语法检查，确认新函数签名可被引用**

运行：`python -m py_compile /home/mxr/lee/gbmtest/grb_project/lat_extended_three_ml.py`

预期：通过；如果失败，先修正导入、返回类型或未定义引用。

- [ ] **步骤 3：Commit**

```bash
git add /home/mxr/lee/gbmtest/grb_project/lat_extended_three_ml.py
git commit -m "refactor: split LAT analysis window setup"
```

---

### 任务 2：抽出重复绘图逻辑，减少主流程长度

**文件：**
- 修改：`/home/mxr/lee/gbmtest/grb_project/lat_extended_three_ml.py`

- [ ] **步骤 1：抽出事件图与窗口聚焦图的绘制函数**

新增两个私有函数，分别负责通用事件分布图和局部聚焦图；主流程仅传入数据和窗口参数。

```python
def _plot_lat_events_overview(
    event_times: np.ndarray,
    energies: np.ndarray,
    t0_lat: float,
    t1_lat: float,
    t0_analysis: float,
    t1_analysis: float,
    t95: float,
    output_path: str = "events.png",
) -> None:
    ...


def _plot_lat_events_zoom(
    event_times: np.ndarray,
    energies: np.ndarray,
    t0_lat: float,
    t1_lat: float,
    t0_analysis: float,
    t1_analysis: float,
    output_path: str = "events_analyze_single_window.png",
) -> None:
    ...
```

预期：两段重复的 `axvspan/hist/scatter/grid/legend` 逻辑只保留在函数内部一次。

- [ ] **步骤 2：把主 worker 中的绘图代码替换为函数调用**

运行后主流程里只保留：

```python
_plot_lat_events_overview(...)
_plot_lat_events_zoom(...)
```

预期：主函数长度明显缩短，但输出图片文件名和主要视觉信息保持一致。

- [ ] **步骤 3：Commit**

```bash
git add /home/mxr/lee/gbmtest/grb_project/lat_extended_three_ml.py
git commit -m "refactor: extract LAT event plotting helpers"
```

---

### 任务 3：统一三ML 拟合判断与模型初始化

**文件：**
- 修改：`/home/mxr/lee/gbmtest/grb_project/lat_extended_three_ml.py`

- [ ] **步骤 1：把“是否执行二次拟合”的判断收敛到单一函数**

新增一个函数，只负责从 `prob_path`、`source_TS.txt` 和 `lat_plugins` 判定是否需要拟合，并返回明确状态。

```python
def _should_run_lat_fit(
    prob_path: str,
    ts_path: str,
    lat_plugin: Any,
) -> tuple[bool, Optional[float], str]:
    ...
```

预期：不再在主循环中重复处理 `os.path.exists`、读取 `source_TS.txt`、解析 GRB TS、打印相同提示。

- [ ] **步骤 2：抽出三ML 模型构建函数**

新增统一的 GRB 点源模型初始化逻辑，主流程只关心输入插件与参数范围。

```python
def _build_grb_joint_likelihood(
    lat_plugin: Any,
    ra: float,
    dec: float,
) -> JointLikelihood:
    ...
```

预期：`PointSource`、`Model`、参数边界、`JointLikelihood` 初始化只保留一处。

- [ ] **步骤 3：运行单文件语法检查**

运行：`python -m py_compile /home/mxr/lee/gbmtest/grb_project/lat_extended_three_ml.py`

预期：通过；若失败，修正类型名、导入或参数传递。

- [ ] **步骤 4：Commit**

```bash
git add /home/mxr/lee/gbmtest/grb_project/lat_extended_three_ml.py
git commit -m "refactor: simplify LAT threeML fitting gate"
```

---

### 任务 4：收敛结果汇总与输出写盘

**文件：**
- 修改：`/home/mxr/lee/gbmtest/grb_project/lat_extended_three_ml.py`

- [ ] **步骤 1：提取结果汇总函数**

将 `result_data` 的字段整理、全局最高能光子统计、拟合摘要写盘整理到一个函数中。

```python
def _finalize_lat_results(
    result_data: Dict[str, Any],
    my_lat_grb_name: str,
    selection: Dict[str, Any],
) -> Dict[str, Any]:
    ...
```

预期：主流程不再混杂大量 `result_data[...] = ...` 与字符串模板拼接。

- [ ] **步骤 2：统一 CSV / TXT 的写盘逻辑**

把 `all_photon.csv`、`all_high_prob.csv`、`highest_photon_per_interval.csv`、`*_fit_results.txt` 的写盘动作集中到汇总阶段，确保每类输出只由一个入口产生。

预期：避免多个循环里重复 append 写同一文件，减少重复数据和维护成本。

- [ ] **步骤 3：运行最小回归验证**

运行：`python -m py_compile /home/mxr/lee/gbmtest/grb_project/lat_extended_three_ml.py`

预期：通过；如果环境允许，再补一次真实样本输入的短流程 smoke test。

- [ ] **步骤 4：Commit**

```bash
git add /home/mxr/lee/gbmtest/grb_project/lat_extended_three_ml.py
git commit -m "refactor: consolidate LAT result export"
```

---

### 任务 5：做一次维护性自检并清理冗余

**文件：**
- 修改：`/home/mxr/lee/gbmtest/grb_project/lat_extended_three_ml.py`

- [ ] **步骤 1：扫描并删除只服务调试、但不服务用户的重复状态变量**

重点检查并处理：
- 重复派生的 `t0_analysis/t1_analysis`
- 仅用于中间判断的临时列表
- 可由函数返回值直接替代的局部标志位

预期：保留用户可见产物，删掉不必要的临时状态。

- [ ] **步骤 2：复查所有日志信息，合并语义重复但文本不同的提示**

把同一类状态提示统一成少量稳定消息，降低后续排障成本。

预期：日志更少但更关键，用户更容易理解流程进度。

- [ ] **步骤 3：运行最终校验**

运行：`python -m py_compile /home/mxr/lee/gbmtest/grb_project/lat_extended_three_ml.py`

预期：通过。

- [ ] **步骤 4：Commit**

```bash
git add /home/mxr/lee/gbmtest/grb_project/lat_extended_three_ml.py
git commit -m "refactor: reduce LAT pipeline redundancy"
```
