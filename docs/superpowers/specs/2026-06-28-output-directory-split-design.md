# 输出目录拆分实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 将 GRB 单次分析的输出目录按职责拆分为 `run_root / grb_name / model_str`、`run_root / grb_name / gbm` 和 `run_root / grb_name / lat`，避免模型结果覆盖、GBM/LAT 中间文件混放，并修复当前因 `result_dir` 调整引发的保存与分析问题。

**架构：** `project.py` 负责定义 burst 级目录结构，并在每个模型运行时单独创建模型输出目录；`bayesian_fit.py` 只接收明确的模型输出目录，负责把 corner 图、fits、spectrum 图和 SED 图写入该模型目录；`gbm_core.py` 和 `lat_processing.py` 分别接收 GBM/LAT 工作目录，确保它们产生的中间文件都写入 `gbm`/`lat` 子目录，而不是污染模型结果目录。这样保留现有分析流程，但把“分析工作目录”和“最终结果目录”彻底解耦。

**技术栈：** Python、Pathlib、pandas、threeML、GtBurst、astropy、matplotlib。

---

### 任务 1：在项目入口层引入 burst/model/工作目录分层

**文件：**
- 修改：`/home/mxr/lee/gbmtest/grb_project/project.py`
- 测试：`/home/mxr/lee/gbmtest/grb_project/tests/test_project_flow.py`

- [ ] **步骤 1：定义 burst 级目录的派生规则**

在 `run_single_analysis(...)` 中，把单个事件的根目录统一为 `base_dir = run_root / grb_name`，并从中派生：

```python
base_dir = run_root / grb_name
model_dir = base_dir / model_str
gbm_dir = base_dir / "gbm"
lat_dir = base_dir / "lat"
```

预期：`result_dir` 不再同时承担“burst 根目录”和“模型结果目录”两种职责。

- [ ] **步骤 2：把模型输出目录传给贝叶斯拟合层**

在调用 `_run_bayesian_analysis_for_model(...)` 时，传入 `model_dir` 作为 `result_dir`，并确保每个模型在自己的目录下创建输出文件：

```python
row = _run_bayesian_analysis_for_model(
    model_str=model_str,
    grb_name=grb_name,
    bnname=bnname,
    ra=float(ra),
    dec=float(dec),
    datalist=datalist,
    plugins=plugins,
    lat_plugin=lat_plugin,
    dets=list(dets),
    result_dir=str(model_dir),
    bin_start=bin_start,
    bin_end=bin_end,
    duration=duration,
    analysis_mode=analysis_mode,
)
```

预期：不同模型不会再共享同一个 `my_results_*.fits` 或图像文件输出位置。

- [ ] **步骤 3：把 GBM/LAT 工作目录从模型目录中剥离**

在 `_run_gbm_analysis(...)` 和 `_run_lat_analysis(...)` 的调用链中，分别显式传入 `gbm_dir` 和 `lat_dir`，避免它们继续使用模型目录作为工作目录。

预期：GBM/LAT 产生的中间文件与最终模型结果分开存储。

- [ ] **步骤 4：运行项目流的最小回归检查**

运行：`python -m py_compile /home/mxr/lee/gbmtest/grb_project/project.py`

预期：语法通过；若失败，优先修复路径变量命名和函数调用参数不一致的问题。

---

### 任务 2：让贝叶斯拟合输出写入模型专属目录

**文件：**
- 修改：`/home/mxr/lee/gbmtest/grb_project/bayesian_fit.py`
- 测试：`/home/mxr/lee/gbmtest/grb_project/tests/test_project_flow.py`

- [ ] **步骤 1：把所有结果文件路径改为基于传入的 `result_dir`**

将 corner 图、FITS、频谱图和 SED 图的路径统一改为“模型目录 + 文件名”模式：

```python
corner_fig_path = os.path.join(
    result_dir,
    f"bs_{bnname}_{model_str}_{bin_start}-{bin_end}_{suffix}_corner_plot.png",
)
result_fits_path = os.path.join(
    result_dir,
    f"my_results_{bin_start}-{bin_end}.fits",
)
spec_fig_path = os.path.join(
    result_dir,
    f"bs_{bnname}_{model_str}_counts_{suffix}_spectrum_{bin_start}-{bin_end}.png",
)
sed_path = os.path.join(
    result_dir,
    f"bs_{bnname}_{model_str}_{suffix}_spectrum_{bin_start}-{bin_end}_total.png",
)
```

预期：不再拼接 `result_dir / model_str / ...`，因为调用方已经把 `result_dir` 设为模型专属目录。

- [ ] **步骤 2：保留下游 summary 字段不变**

确认 `summary_entry` 仍按原字段返回，不因目录拆分而改变统计列名、数值来源或分析模式逻辑。

预期：现有 summary CSV 读取方不需要修改。

- [ ] **步骤 3：运行单文件语法检查**

运行：`python -m py_compile /home/mxr/lee/gbmtest/grb_project/bayesian_fit.py`

预期：通过；如果失败，修复路径拼接或未使用变量告警导致的错误。

---

### 任务 3：让 GBM 中间文件落在 `run_root / grb_name / gbm`

**文件：**
- 修改：`/home/mxr/lee/gbmtest/grb_project/project.py`
- 修改：`/home/mxr/lee/gbmtest/grb_project/gbm_core.py`
- 测试：`/home/mxr/lee/gbmtest/grb_project/tests/test_gbm_core.py`

- [ ] **步骤 1：在 GBM 分析入口传入独立的 GBM 工作目录**

把 `_run_gbm_analysis(...)` 的工作目录参数改成明确的 `gbm_dir`，并在调用 `_build_gbm_plugin_for_detector(...)` 前把它作为 `grb_dir` 传入。

预期：GBM detector 选择、PHA 生成、背景文件、时间序列临时文件都在 `gbm` 子目录内工作。

- [ ] **步骤 2：确保 GBM 核心函数在工作目录内生成文件**

`_build_gbm_plugin_for_detector(...)` 当前会生成这些文件：

```python
ts_cspec.save_background(f"{det}_bkg.h5", overwrite=True)
ts_tte.write_pha_from_binner(
    file_name=f"gbm_tte_{det}",
    overwrite=True,
    force_rsp_write=True,
)
```

需要确保这些文件实际写入传入的 GBM 目录，而不是默认当前工作目录。

预期：`gbm_tte_*.pha`、`gbm_tte_*_bak.pha`、`gbm_tte_*.rsp`、`*_bkg.h5` 都可在 `run_root / grb_name / gbm` 中找到。

- [ ] **步骤 3：补一条针对路径归属的测试**

在 `tests/test_gbm_core.py` 或 `tests/test_project_flow.py` 中加入对 `gbm_dir` 归属的断言，确认 GBM 输出路径不再指向模型目录。

预期：路径拆分回归时测试会直接失败，而不是等到拟合阶段才暴露。

- [ ] **步骤 4：运行语法与关键测试**

运行：

```bash
python -m py_compile /home/mxr/lee/gbmtest/grb_project/gbm_core.py /home/mxr/lee/gbmtest/grb_project/project.py
pytest /home/mxr/lee/gbmtest/grb_project/tests/test_gbm_core.py -q
```

预期：语法通过；测试在当前可用依赖范围内通过，或者至少能明确报出路径断言是否生效。

---

### 任务 4：让 LAT 处理中间文件落在 `run_root / grb_name / lat`

**文件：**
- 修改：`/home/mxr/lee/gbmtest/grb_project/project.py`
- 修改：`/home/mxr/lee/gbmtest/grb_project/lat_processing.py`
- 测试：`/home/mxr/lee/gbmtest/grb_project/tests/test_project_flow.py`

- [ ] **步骤 1：把 LAT 处理入口改成接收独立的 LAT 工作目录**

将 `process_lat_data(bnname, t0, t1, _result_dir)` 的 `_result_dir` 语义固定为 LAT 工作目录，并从项目入口传入 `lat_dir`。

预期：`gtselect`、`gtbin`、`gtrspgen` 产物全部进入 `lat` 子目录。

- [ ] **步骤 2：确保 LAT 目录在创建前存在**

在进入 LAT 处理前创建目录：

```python
lat_dir.mkdir(parents=True, exist_ok=True)
```

预期：即使没有预先创建 burst 目录，LAT 处理也不会因为目录不存在而失败。

- [ ] **步骤 3：保持 LAT 返回对象行为不变**

`process_lat_data(...)` 仍应返回 `OGIPLike` 或 `None`，不改变上层联合拟合逻辑。

预期：仅变更文件位置，不变更拟合数据对象接口。

- [ ] **步骤 4：运行单文件语法检查**

运行：`python -m py_compile /home/mxr/lee/gbmtest/grb_project/lat_processing.py`

预期：通过；若失败，优先修复路径参数名和类型提示。

---

### 任务 5：补一个端到端目录结构回归测试

**文件：**
- 修改：`/home/mxr/lee/gbmtest/grb_project/tests/test_project_flow.py`

- [ ] **步骤 1：构造一个最小目录分层断言**

添加一个测试，断言单次分析会派生出：

```python
base_dir = run_root / grb_name
model_dir = base_dir / model_str
gbm_dir = base_dir / "gbm"
lat_dir = base_dir / "lat"
```

并且模型结果文件只出现在 `model_dir`，GBM/LAT 中间文件只出现在对应工作目录。

预期：目录职责混放会被测试直接捕获。

- [ ] **步骤 2：运行测试并确认回归保护有效**

运行：`pytest /home/mxr/lee/gbmtest/grb_project/tests/test_project_flow.py -q`

预期：通过；如果失败，先修正断言与当前实际输出路径的差异。

---

### 任务 6：做一次整体自检，确保没有目录回流

**文件：**
- 修改：`/home/mxr/lee/gbmtest/grb_project/project.py`
- 修改：`/home/mxr/lee/gbmtest/grb_project/bayesian_fit.py`
- 修改：`/home/mxr/lee/gbmtest/grb_project/gbm_core.py`
- 修改：`/home/mxr/lee/gbmtest/grb_project/lat_processing.py`

- [ ] **步骤 1：全局搜索旧的目录拼接模式**

检查是否还存在把模型目录同时拿来做 GBM/LAT 工作目录的逻辑，重点搜索类似：

```python
os.path.join(result_dir, model_str, ...)
```

或把 `result_dir` 同时传给模型输出和 LAT/GBM 工作函数的调用。

预期：所有职责清晰分离，且没有旧路径残留。

- [ ] **步骤 2：运行最小语法与测试回归**

运行：

```bash
python -m py_compile \
  /home/mxr/lee/gbmtest/grb_project/project.py \
  /home/mxr/lee/gbmtest/grb_project/bayesian_fit.py \
  /home/mxr/lee/gbmtest/grb_project/gbm_core.py \
  /home/mxr/lee/gbmtest/grb_project/lat_processing.py
pytest /home/mxr/lee/gbmtest/grb_project/tests/test_project_flow.py -q
```

预期：通过，或者暴露出仍然引用旧目录职责的地方。

- [ ] **步骤 3：准备交付总结**

总结本次变更时明确说明：
- 模型结果现在在 `run_root / grb_name / model_str`
- GBM 中间文件在 `run_root / grb_name / gbm`
- LAT 中间文件在 `run_root / grb_name / lat`

预期：用户可直接据此理解新目录结构。
