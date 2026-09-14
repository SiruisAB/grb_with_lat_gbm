# 联合分析目标一键选择（joint selection）实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在 `grb_project` 中新增核心模块 `joint_selection.py`——以 2026-09 建立的 `lat_download_targets.csv` 交叉表（GBM 目录 × 2FLGC × LLE × GCN 通报）加磁盘实况为准，**一键选出可做 GBM+LAT 联合分析的 GRB**：支持年份区间（从哪年到哪年）、LAT 探测显著性下限、LLE-only 开关等判据；接口层提供 CLI 子命令 `select`（可 `--run` 直接把选中目标交给现有批量分析、`--models` 等参数透传）与 Web「联合目标选择」页面（表格、逐年统计、排除原因、CSV 导出、生成 CLI 命令）。

**架构：** 三层。核心层 `joint_selection.py` 是纯逻辑（判据 dataclass + 选择函数 + 结果 dataclass），只依赖 pandas/pathlib，**不依赖 threeML**，保证可单测、可在任何环境 import；接口层薄封装：`pipeline.py` 加 `select` 子命令（复用现有 `parse_args` 的 argv[0] 分发模式），`web_app.py` 加侧边栏页面（复用现有 `_run_*_page` 模式，早返回、不加载目录表）。选择时对 `GBM_data/{bnname}` 与 `Extended_data_ex/{grb_name}` 做磁盘复核（TTE 存在性 / 事件+SC 文件对），不盲信交叉表标志位。每个排除原因逐项记录，供界面展示与复查。

**技术栈：** pandas、pathlib、argparse、Streamlit（页面）、unittest（测试，与现有 `tests/` 一致）。

---

## 数据源与判据

### 数据源

`/home/mxr/lee/gbmtest/lat_download_targets.csv`（新增 `session.joint_target_csv` 默认路径），关键列：

| 列 | 含义 |
|----|------|
| `bnname` / `grb_name` | GBM 触发名 / GCN 名（也是 LAT 数据目录名） |
| `gbm_held` / `lat_downloaded` | 本地 GBM / LAT 数据标志（选择时仍要磁盘复核） |
| `lat_grb_valid` | LAT 对应体有效性（GRB140330A 已撤回 = False） |
| `requires_lle_data` / `lat_only_lle` | LLE-only 探测（无标准 >100 MeV 似然探测） |
| `flgc_lat_ts` | 2FLGC LAT 似然 TS（显著性判据） |
| `trigger_met` / `ra` / `dec` / `T0` / `T1` | 联合分析必需的参数完备性检查项 |
| `window_source` | 时间窗来源（flgc_tl / gcn / lle / gcn_manual…，供参考） |

### 判据顺序（每步记录排除原因，先到先得）

1. `lat_grb_valid=False` → 排除「LAT 对应体已撤回」
2. LLE-only 且未开 `include_lle_only` → 排除「LLE-only」
3. `analysis_mode` 含 `gbm`（默认 gbm+lat）：`gbm_held=False` → 「本地无 GBM 数据」；磁盘复核失败 → 「GBM 目录缺 TTE」
4. `analysis_mode` 含 `lat`：`lat_downloaded=False` → 「LAT 数据未下载」；磁盘复核失败 → 「LAT 目录不完整」
5. `trigger_met/ra/dec/T0/T1` 任缺或 `T1<=T0` → 「参数不完备」
6. 年份（取 `bnname[2:4]`）不在 `[year_from, year_to]` → 「年份不符」
7. `min_lat_ts` 设定时：`flgc_lat_ts` 缺失或小于阈值 → 「LAT TS 不足」

约束：本表宇宙是 LAT 候选暴，`analysis_mode` 仅接受 `gbm+lat` / `lat`（`gbm` 应走 GBM 目录表流程，传入即报错）。

---

## 文件结构

### 新增文件
- `grb_project/joint_selection.py`：核心模块。`JointSelectionCriteria` / `JointTarget` / `JointSelectionResult`（含 `bnnames`、`to_frame()`）、`load_joint_target_table()`、`select_joint_targets()`、`format_selection_report()`、`add_selection_arguments()`、`cli_main_from_args()`。
- `grb_project/tests/test_joint_selection.py`：fixture CSV + 临时 GBM/LAT 目录，覆盖全部判据分支。

### 修改文件
- `grb_project/session.py`：`AnalysisSessionState` 增加 `joint_target_csv` 字段。
- `grb_project/pipeline.py`：`_build_selection_parser()`；`parse_args` 增加 `select` 分支；`cli_main` 分发（懒导入）。
- `grb_project/web_app.py`：新增 `_run_joint_selection_page()`；`main()` 侧边栏 radio 增加「联合目标选择」并早返回分发（该页不加载目录表）。
- `grb_project/说明文档.md`：模块一览表加一行；CLI 用法补 `select` 示例。

---

## 任务 1：核心选择逻辑

**文件：** 创建 `grb_project/joint_selection.py`；测试 `grb_project/tests/test_joint_selection.py`

- [ ] **步骤 1：编写失败的测试**——fixture 构造 8–10 行交叉表（覆盖：完全就绪、撤回、LLE-only、GBM 缺目录、LAT 缺文件、参数不全、不同年份、不同 TS），断言各判据的选入/排除与排除原因归组。
- [ ] **步骤 2：运行 `pytest grb_project/tests/test_joint_selection.py -v` 验证失败**（模块不存在）。
- [ ] **步骤 3：实现** `JointSelectionCriteria`（`year_from/year_to/min_lat_ts/include_lle_only/analysis_mode/verify_on_disk`，frozen dataclass）、`load_joint_target_table()`（必需列校验，缺列报 ValueError 列名）、`select_joint_targets(criteria, *, table, data_dir, lat_root)`（按上文判据顺序，磁盘复核：GBM 目录含 `glg_tte*.fit`；LAT 目录含 `*_EV*/*_PH*/*FT1*` 事件文件与 `*_SC*` 文件）、`JointSelectionResult.to_frame()`（列：bnname, grb_name, year, trigger_met, T0, T1, ra, dec, window_source, lat_ts，按 bnname 排序）。
- [ ] **步骤 4：测试转绿**；`analysis_mode="gbm"` 与非法模式断言 ValueError。

## 任务 2：CLI 子命令

**文件：** 修改 `grb_project/pipeline.py`；`joint_selection.py` 补 CLI 函数

- [ ] `add_selection_arguments()`：`--year-from/--year-to/--min-lat-ts/--include-lle-only/--analysis-mode{gbm+lat,lat}/--table/--no-verify-on-disk/--output/--grbs-file/--run/--result-root/--summary-csv-name/--session-log/--models`。
- [ ] `cli_main_from_args()`：打印报告（总数、逐年统计、排除原因、前若干 bn）；`--output` 写选中表 CSV；`--grbs-file` 写 bn 名单（每行一个，可直接 `xargs`）；`--run` 且选中非空时懒导入 `pipeline.main`，`--models` 经 `GRBRunOverrides(models=...)` 透传，其余参数原样交接。
- [ ] `parse_args` / `cli_main` 增加 `select` 分支（沿用 `download-lat` 的 argv[0] 模式，懒导入）。
- [ ] 冒烟：`python -m grb_project select --year-from 2008 --year-to 2010` 输出统计；`--run` 分支用 `--help` 与空选择保护（空选择报错退出码 1，不启动分析）。

## 任务 3：Web 页面

**文件：** 修改 `grb_project/web_app.py`

- [ ] `_run_joint_selection_page(st, base_cfg)`：四控件（起始年/截止年/最小 TS/包含 LLE-only）→ `select_joint_targets`；三个 metric（可选目标/表内总数/被排除数）+ 逐年 `st.bar_chart` + `st.dataframe(to_frame())`；expander 展示排除原因明细；`st.download_button` 导出 CSV；`st.code` 生成等价 CLI 命令。
- [ ] `main()` 的 radio options 加入「联合目标选择」，与「GBM 数据下载」同样早返回分发。
- [ ] 页面不加载目录表、不触发分析（批量执行留给 CLI `--run` 或后续任务队列，见「不在本计划」）。

## 任务 4：文档与提交

- [ ] `说明文档.md`：模块一览表加 `joint_selection.py` 一行；第 8 节 CLI 用法补 `select` 示例。
- [ ] 全量测试 `pytest grb_project/tests/ -q` 无回归。
- [ ] 按仓库提交规范单 commit：`feat(selection): 联合分析目标一键选择核心 + CLI/Web 接口`。

---

## 不在本计划（明确划出）

- **Web 端批量执行/任务队列**：2026-06-24 计划中的 task_store/worker 体系尚未落地；本计划的 Web 页面只做选择与导出，不阻塞、不启动分析。
- **gbm-only 模式选择**：本表宇宙是 LAT 候选暴；GBM 全目录选择走既有 `--gcn-all`/目录表流程。
- **修改既有分析流程**（analyze_single/pipeline.main 的行为不动，`--run` 只是复用）。
