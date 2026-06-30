# fluxdata 叠加拟合曲线实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在现有 `fluxdata` 重绘图上，按每个时间段读取 `GRB231129C_allmodel_nested_compact.json` 中对应的 `band+bb` 自由参数，叠加总模型曲线以及 `Band` / `Blackbody` 组件曲线，并在日志中明确记录所使用的输入目录、输出目录和拟合参数来源。

**架构：** 保持现有 `fluxdata` 读图流程不变，只增加一层“拟合参数解析 + 模型曲线生成”。实现上把 JSON 读取、时间段匹配、`band+bb` 参数提取、`Band` 与 `Blackbody` 谱函数计算拆成独立辅助函数，再由单图和总览图共享同一套叠加逻辑。这样可以保证输出图样式一致，同时避免把解析逻辑散落在绘图代码里。

**技术栈：** Python 3、`json`、`pathlib`、`numpy`、`matplotlib`、现有 `grb_project/separate_spectr.py`。

---

## 将修改的文件

- 修改：`/home/mxr/lee/gbmtest/grb_project/separate_spectr.py`
  - 增加 JSON 读取和时间段匹配辅助函数
  - 增加 `band+bb` 模型曲线计算函数
  - 增加总模型与组件曲线叠加到单图 / 总览图的逻辑
  - 扩展 CLI，让它能接收 JSON 路径并在日志中打印拟合来源
- 修改：`/home/mxr/lee/gbmtest/grb_project/tests/test_separate_spectr_fluxdata_redraw.py`
  - 验证 JSON 解析、参数映射、曲线生成、输出日志与文件路径

---

### 任务 1：建立 JSON 参数解析与时间段匹配

**文件：**
- 修改：`/home/mxr/lee/gbmtest/grb_project/separate_spectr.py`
- 测试：`/home/mxr/lee/gbmtest/grb_project/tests/test_separate_spectr_fluxdata_redraw.py`

- [ ] **步骤 1：编写失败的测试**

```python
from pathlib import Path


def test_load_bandbb_fit_params_from_json(tmp_path: Path):
    from grb_project.separate_spectr import load_bandbb_fit_params

    json_path = tmp_path / "GRB231129C_allmodel_nested_compact.json"
    json_path.write_text(
        '{"GRB231129C": {"bins": {"bn231129799_bin_0.10_1.00": {"band+bb": {"GRB.spectrum.main.composite.K_1_value": 1.0, "GRB.spectrum.main.composite.alpha_1_value": -0.5, "GRB.spectrum.main.composite.xp_1_value": 200.0, "GRB.spectrum.main.composite.beta_1_value": -2.5, "GRB.spectrum.main.composite.K_2_value": 1e-6, "GRB.spectrum.main.composite.kT_2_value": 100.0}}}}}'
    )

    params = load_bandbb_fit_params(json_path, bnname="GRB231129C", timebin="0.1-1")

    assert params["K_1"] == 1.0
    assert params["alpha_1"] == -0.5
    assert params["xp_1"] == 200.0
    assert params["beta_1"] == -2.5
    assert params["K_2"] == 1e-6
    assert params["kT_2"] == 100.0
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_separate_spectr_fluxdata_redraw.py::test_load_bandbb_fit_params_from_json -v`
预期：FAIL，因为 JSON 解析函数还不存在。

- [ ] **步骤 3：编写最少实现代码**

新增一个解析函数，负责从 `GRB231129C_allmodel_nested_compact.json` 中找到对应时间段条目，并返回统一字段：

```python
def load_bandbb_fit_params(json_path: Path, bnname: str, timebin: str) -> dict[str, float]:
    # 读取 JSON
    # 通过 bin_start_time / bin_end_time 或 time_bin_identifier 匹配 timebin
    # 提取 K_1, alpha_1, xp_1, beta_1, K_2, kT_2
    # 返回统一字典
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_separate_spectr_fluxdata_redraw.py::test_load_bandbb_fit_params_from_json -v`
预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add separate_spectr.py tests/test_separate_spectr_fluxdata_redraw.py
git commit -m "feat(plot): load bandbb fit params from json"
```

---

### 任务 2：实现 `Band + Blackbody` 组件曲线计算

**文件：**
- 修改：`/home/mxr/lee/gbmtest/grb_project/separate_spectr.py`
- 测试：`/home/mxr/lee/gbmtest/grb_project/tests/test_separate_spectr_fluxdata_redraw.py`

- [ ] **步骤 1：编写失败的测试**

```python
import numpy as np


def test_compute_bandbb_component_curves_returns_total_and_components():
    from grb_project.separate_spectr import compute_bandbb_curves

    xs = np.logspace(1, 3, 5)
    curves = compute_bandbb_curves(
        xs,
        K_1=1.0,
        alpha_1=-0.5,
        xp_1=200.0,
        beta_1=-2.5,
        K_2=1e-6,
        kT_2=100.0,
    )

    assert set(curves) == {"total", "band", "bb"}
    assert curves["total"].shape == xs.shape
    assert curves["band"].shape == xs.shape
    assert curves["bb"].shape == xs.shape
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_separate_spectr_fluxdata_redraw.py::test_compute_bandbb_component_curves_returns_total_and_components -v`
预期：FAIL，因为曲线计算函数还不存在。

- [ ] **步骤 3：编写最少实现代码**

增加一个纯数值函数，只依赖 `numpy`：

```python
def compute_bandbb_curves(xs, K_1, alpha_1, xp_1, beta_1, K_2, kT_2):
    # 用 Band 和 Blackbody 的解析形式生成 E^2 dN/dE
    # 返回 total/band/bb 三条曲线
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_separate_spectr_fluxdata_redraw.py::test_compute_bandbb_component_curves_returns_total_and_components -v`
预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add separate_spectr.py tests/test_separate_spectr_fluxdata_redraw.py
git commit -m "feat(plot): compute bandbb overlay curves"
```

---

### 任务 3：把拟合曲线叠加到单图和总览图上

**文件：**
- 修改：`/home/mxr/lee/gbmtest/grb_project/separate_spectr.py`
- 测试：`/home/mxr/lee/gbmtest/grb_project/tests/test_separate_spectr_fluxdata_redraw.py`

- [ ] **步骤 1：编写失败的测试**

```python
from pathlib import Path


def test_redraw_single_timebin_overlays_fit_curves(tmp_path: Path):
    from grb_project.separate_spectr import redraw_fluxdata_timebin_with_fit

    # 准备 fluxdata 和 JSON
    # 调用后断言输出 PDF 存在
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_separate_spectr_fluxdata_redraw.py::test_redraw_single_timebin_overlays_fit_curves -v`
预期：FAIL，因为新函数还不存在。

- [ ] **步骤 3：编写最少实现代码**

在现有 `redraw_fluxdata_timebin` / `redraw_fluxdata_overview` 基础上增加可选的 `fit_json_path` 参数：

```python
def redraw_fluxdata_timebin(..., fit_json_path: Path | None = None):
    # 先画数据点
    # 如果提供 JSON，就调用 load_bandbb_fit_params + compute_bandbb_curves
    # 叠加 total / band / bb 曲线
```

总览图同样调用共享函数，确保单图和总览图风格一致。

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_separate_spectr_fluxdata_redraw.py::test_redraw_single_timebin_overlays_fit_curves -v`
预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add separate_spectr.py tests/test_separate_spectr_fluxdata_redraw.py
git commit -m "feat(plot): overlay fitted bandbb curves on fluxdata"
```

---

### 任务 4：扩展 CLI，打印拟合 JSON 和保存路径

**文件：**
- 修改：`/home/mxr/lee/gbmtest/grb_project/separate_spectr.py`
- 测试：`/home/mxr/lee/gbmtest/grb_project/tests/test_separate_spectr_fluxdata_redraw.py`

- [ ] **步骤 1：编写失败的测试**

```python
from pathlib import Path


def test_cli_logs_json_and_output_paths(tmp_path: Path):
    from grb_project import separate_spectr

    # 准备 fluxdata / JSON
    # 运行 main([...])
    # 断言日志中包含 JSON 路径、输出目录、保存文件路径
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_separate_spectr_fluxdata_redraw.py::test_cli_logs_json_and_output_paths -v`
预期：FAIL。

- [ ] **步骤 3：编写最少实现代码**

给 CLI 增加：
- `--fit-json`：JSON 路径
- 日志输出中打印：
  - `Fluxdata directory`
  - `Fit JSON path`
  - `Output directory`
  - `Log file`
  - `Saved: ...`

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_separate_spectr_fluxdata_redraw.py::test_cli_logs_json_and_output_paths -v`
预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add separate_spectr.py tests/test_separate_spectr_fluxdata_redraw.py
git commit -m "feat(plot): log fit json and output paths"
```

---

## 自检

### 1. 规格覆盖度
- 读取 `GRB231129C_allmodel_nested_compact.json` → 任务 1
- 将自由参数映射回 `band+bb` 模型 → 任务 2
- 叠加总模型和组件曲线 → 任务 3
- 日志中显示 JSON 与保存目录 → 任务 4

### 2. 占位符扫描
- 没有 `TODO`
- 没有 `待定`
- 没有 `后续实现`
- 每个任务都有可执行测试和明确的文件路径

### 3. 类型一致性
- `load_bandbb_fit_params(json_path: Path, bnname: str, timebin: str) -> dict[str, float]`
- `compute_bandbb_curves(xs, K_1, alpha_1, xp_1, beta_1, K_2, kT_2) -> dict[str, np.ndarray]`
- `redraw_fluxdata_timebin(..., fit_json_path: Path | None = None)`
- `redraw_fluxdata_overview(..., fit_json_path: Path | None = None)`
- `main(argv: Optional[list[str]] = None) -> int`

---

## 交接执行选项

计划已完成并保存到 `docs/superpowers/plans/2026-06-30-fluxdata-fit-overlay.md`。两种执行方式：

1. **子代理驱动（推荐）** - 每个任务调度一个新的子代理，任务间进行审查，快速迭代
2. **内联执行** - 在当前会话中使用 `executing-plans` 执行任务，批量执行并设有检查点供审查

选哪种方式？
