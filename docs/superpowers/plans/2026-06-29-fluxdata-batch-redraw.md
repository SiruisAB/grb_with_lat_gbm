# fluxdata 批量重绘谱图 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 直接利用 `results_test/GRB231129C/band+bb/fluxdata` 里的导出数据，批量重绘每个时间段的谱图，并同时生成单独图和多子图总览图，保持当前 `band+bb` 的配色、误差棒和出版风格。

**架构：** 在 `separate_spectr.py` 中增加一个只负责“从 `fluxdata` 文件恢复并绘图”的通用入口，把数据发现、文件解析、单图绘制、多子图排版拆成小函数。主绘图逻辑仍复用现有的轴样式、对数坐标和保存约定，只把数据源从三ML 插件输出改成 `*.txt` 文件。这样既不影响现有拟合流程，也能为后续其它 burst 的 `fluxdata` 重绘复用同一套代码。

**技术栈：** Python 3、`pathlib`/`glob`、`numpy`、`matplotlib`、现有项目的 `separate_spectr.py`。

---

## 先决条件与输入约定

### `fluxdata` 文件命名规则

本次重绘只面向 `band+bb` 的导出结果，目录里已经存在如下四类文件：

- `band+bb_nai_n3_data_point_<timebin>.txt`
- `band+bb_nai_n7_data_point_<timebin>.txt`
- `band+bb_bgo_b0_data_point_<timebin>.txt`
- `band+bb_lat_data_point_<timebin>.txt`

每个文件每行六列：

1. 能量中心 `E`
2. 下界误差 `E_low`
3. 上界误差 `E_high`
4. 谱通量 `E^2 dN/dE`
5. 下误差 `flux_err_low`
6. 上误差 `flux_err_high`

这些数据可直接喂给 `matplotlib.errorbar` 重绘，无需重新跑 `threeML` 或重新拟合。

---

## 将修改的文件

- 修改：`/home/mxr/lee/gbmtest/grb_project/separate_spectr.py`
  - 增加从 `fluxdata` 读取数据的辅助函数
  - 增加单时间段重绘函数
  - 增加多时间段总览图函数
  - 为批量重绘增加一个公开入口
- 新增测试：`/home/mxr/lee/gbmtest/grb_project/tests/test_separate_spectr_fluxdata_redraw.py`
  - 验证文件发现、时间段分组、数据读取、输出文件名和图层组织

---

### 任务 1：整理 `separate_spectr.py` 中可复用的绘图边界

**文件：**
- 修改：`/home/mxr/lee/gbmtest/grb_project/separate_spectr.py:1-40`

- [ ] **步骤 1：编写失败的测试**

```python
def test_public_axis_style_hides_top_and_right_ticks_and_spines():
    import matplotlib.pyplot as plt
    from grb_project.separate_spectr import _style_publication_axes

    fig, ax = plt.subplots()
    _style_publication_axes(ax)

    assert not ax.spines["top"].get_visible()
    assert not ax.spines["right"].get_visible()
    assert ax.spines["left"].get_visible()
    assert ax.spines["bottom"].get_visible()
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_separate_spectr_fluxdata_redraw.py::test_public_axis_style_hides_top_and_right_ticks_and_spines -v`
预期：FAIL，因为还没有这个测试文件，也没有针对重绘入口的组织方式。

- [ ] **步骤 3：编写最少实现代码**

把当前的 `_style_publication_axes` 明确成“只保留左/下边框”，并让它成为后续重绘函数的统一轴样式来源：

```python
def _style_publication_axes(ax):
    ax.tick_params(axis="both", which="major", direction="in", top=False, right=False, length=6, width=1.6, labelsize=14)
    ax.tick_params(axis="both", which="minor", direction="in", top=False, right=False, length=3, width=1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.8)
    ax.spines["bottom"].set_linewidth(1.8)
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_separate_spectr_fluxdata_redraw.py::test_public_axis_style_hides_top_and_right_ticks_and_spines -v`
预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add separate_spectr.py tests/test_separate_spectr_fluxdata_redraw.py
git commit -m "refactor(plot): reuse publication axes style for fluxdata redraw"
```

---

### 任务 2：增加 `fluxdata` 文件发现与时间段分组

**文件：**
- 修改：`/home/mxr/lee/gbmtest/grb_project/separate_spectr.py`
- 测试：`/home/mxr/lee/gbmtest/grb_project/tests/test_separate_spectr_fluxdata_redraw.py`

- [ ] **步骤 1：编写失败的测试**

```python
from pathlib import Path


def test_group_fluxdata_files_by_timebin(tmp_path: Path):
    from grb_project.separate_spectr import discover_fluxdata_groups

    flux_dir = tmp_path / "fluxdata"
    flux_dir.mkdir()
    (flux_dir / "band+bb_nai_n3_data_point_0.1-1.txt").write_text("1 0.1 0.1 2 0.2 0.2\n")
    (flux_dir / "band+bb_bgo_b0_data_point_0.1-1.txt").write_text("1 0.1 0.1 2 0.2 0.2\n")
    (flux_dir / "band+bb_lat_data_point_0.1-1.txt").write_text("1 0.1 0.1 2 0.2 0.2\n")
    (flux_dir / "band+bb_nai_n7_data_point_0.1-1.txt").write_text("1 0.1 0.1 2 0.2 0.2\n")

    groups = discover_fluxdata_groups(flux_dir)

    assert list(groups) == ["0.1-1"]
    assert set(groups["0.1-1"].keys()) == {"nai_n3", "nai_n7", "bgo_b0", "lat"}
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_separate_spectr_fluxdata_redraw.py::test_group_fluxdata_files_by_timebin -v`
预期：FAIL，提示 `discover_fluxdata_groups` 不存在。

- [ ] **步骤 3：编写最少实现代码**

在 `separate_spectr.py` 里新增一个分组函数，使用 `Path.glob("*.txt")` 和文件名解析，把同一时间段的四个探测器文件归为一组：

```python
from pathlib import Path
import re

_FLUXDATA_RE = re.compile(r"^band\+bb_(nai_[^_]+|bgo_[^_]+|lat)_data_point_(.+)\.txt$")


def discover_fluxdata_groups(fluxdata_dir: Path) -> dict[str, dict[str, Path]]:
    groups: dict[str, dict[str, Path]] = {}
    for path in sorted(fluxdata_dir.glob("*.txt")):
        m = _FLUXDATA_RE.match(path.name)
        if not m:
            continue
        detector_tag, timebin = m.groups()
        groups.setdefault(timebin, {})[detector_tag] = path
    return groups
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_separate_spectr_fluxdata_redraw.py::test_group_fluxdata_files_by_timebin -v`
预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add separate_spectr.py tests/test_separate_spectr_fluxdata_redraw.py
git commit -m "feat(plot): group fluxdata files by time bin"
```

---

### 任务 3：实现单个时间段的谱图重绘

**文件：**
- 修改：`/home/mxr/lee/gbmtest/grb_project/separate_spectr.py`
- 测试：`/home/mxr/lee/gbmtest/grb_project/tests/test_separate_spectr_fluxdata_redraw.py`

- [ ] **步骤 1：编写失败的测试**

```python
from pathlib import Path


def test_redraw_single_timebin_creates_pdf(tmp_path: Path):
    from grb_project.separate_spectr import redraw_fluxdata_timebin

    flux_dir = tmp_path / "fluxdata"
    flux_dir.mkdir()
    for name in [
        "band+bb_nai_n3_data_point_0.1-1.txt",
        "band+bb_nai_n7_data_point_0.1-1.txt",
        "band+bb_bgo_b0_data_point_0.1-1.txt",
        "band+bb_lat_data_point_0.1-1.txt",
    ]:
        (flux_dir / name).write_text("1 0.1 0.1 2 0.2 0.2\n")

    out_path = redraw_fluxdata_timebin(
        timebin="0.1-1",
        fluxdata_dir=flux_dir,
        output_dir=tmp_path / "out",
        bnname="GRB231129C",
    )

    assert out_path.name == "bs_GRB231129C_gbm_lat_spectra_band+bb_0.1-1.pdf"
    assert out_path.exists()
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_separate_spectr_fluxdata_redraw.py::test_redraw_single_timebin_creates_pdf -v`
预期：FAIL，因为重绘函数还不存在。

- [ ] **步骤 3：编写最少实现代码**

实现一个以 `fluxdata` 为输入的单图函数，读取四个探测器数据后调用统一绘图逻辑：

```python
def redraw_fluxdata_timebin(timebin: str, fluxdata_dir: Path, output_dir: Path, bnname: str) -> Path:
    groups = discover_fluxdata_groups(fluxdata_dir)
    files = groups[timebin]
    fig, ax = plt.subplots(figsize=(12, 8))
    _plot_fluxdata_group(ax, files, timebin=timebin, bnname=bnname)
    out_path = output_dir / f"bs_{bnname}_gbm_lat_spectra_band+bb_{timebin}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    return out_path
```

其中 `_plot_fluxdata_group` 负责：
- 读取 txt 数据
- 绘制 `errorbar`
- 绘制 `band+bb` 风格的总模型线（如果仅用导出数据，也可以先只画观测点，作为最小可用版本）
- 设置标题、坐标轴、图例、轴样式

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_separate_spectr_fluxdata_redraw.py::test_redraw_single_timebin_creates_pdf -v`
预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add separate_spectr.py tests/test_separate_spectr_fluxdata_redraw.py
git commit -m "feat(plot): redraw a single spectrum from fluxdata"
```

---

### 任务 4：实现所有时间段的批量重绘与多子图总览图

**文件：**
- 修改：`/home/mxr/lee/gbmtest/grb_project/separate_spectr.py`
- 测试：`/home/mxr/lee/gbmtest/grb_project/tests/test_separate_spectr_fluxdata_redraw.py`

- [ ] **步骤 1：编写失败的测试**

```python
from pathlib import Path


def test_batch_redraw_creates_individual_and_overview_outputs(tmp_path: Path):
    from grb_project.separate_spectr import redraw_all_fluxdata_spectra

    flux_dir = tmp_path / "fluxdata"
    flux_dir.mkdir()
    for timebin in ["0.1-1", "1-3"]:
        for prefix in ["nai_n3", "nai_n7", "bgo_b0", "lat"]:
            (flux_dir / f"band+bb_{prefix}_data_point_{timebin}.txt").write_text("1 0.1 0.1 2 0.2 0.2\n")

    outputs = redraw_all_fluxdata_spectra(
        fluxdata_dir=flux_dir,
        output_dir=tmp_path / "out",
        bnname="GRB231129C",
    )

    assert (tmp_path / "out" / "bs_GRB231129C_gbm_lat_spectra_band+bb_0.1-1.pdf").exists()
    assert (tmp_path / "out" / "bs_GRB231129C_gbm_lat_spectra_band+bb_1-3.pdf").exists()
    assert (tmp_path / "out" / "bs_GRB231129C_gbm_lat_spectra_band+bb_overview.pdf").exists()
    assert len(outputs["single_plots"]) == 2
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_separate_spectr_fluxdata_redraw.py::test_batch_redraw_creates_individual_and_overview_outputs -v`
预期：FAIL，因为批量入口不存在。

- [ ] **步骤 3：编写最少实现代码**

增加一个批量入口，先逐个生成单图，再用 `matplotlib.subplots` 把每个时间段排成多子图：

```python
def redraw_all_fluxdata_spectra(fluxdata_dir: Path, output_dir: Path, bnname: str) -> dict[str, list[Path]]:
    groups = discover_fluxdata_groups(fluxdata_dir)
    single_plots = []
    for timebin in groups:
        single_plots.append(redraw_fluxdata_timebin(timebin, fluxdata_dir, output_dir, bnname))
    overview_path = redraw_fluxdata_overview(groups, fluxdata_dir, output_dir, bnname)
    return {"single_plots": single_plots, "overview": [overview_path]}
```

`redraw_fluxdata_overview` 负责：
- 自动计算子图布局（例如按 2 列或 3 列排版）
- 每个子图绘制一个时间段
- 共享坐标轴范围，保证横向可比性
- 在标题或角标里显示时间段标签

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_separate_spectr_fluxdata_redraw.py::test_batch_redraw_creates_individual_and_overview_outputs -v`
预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add separate_spectr.py tests/test_separate_spectr_fluxdata_redraw.py
git commit -m "feat(plot): batch redraw fluxdata spectra and overview"
```

---

### 任务 5：把入口接到现有分析约定里，并补充使用说明

**文件：**
- 修改：`/home/mxr/lee/gbmtest/grb_project/separate_spectr.py`
- 可选修改：`/home/mxr/lee/gbmtest/grb_project/README.md` 或项目现有说明文档（如果仓库里已有相应文档）

- [ ] **步骤 1：编写失败的测试**

```python
def test_redraw_entrypoint_uses_default_bandbb_fluxdata_path(monkeypatch, tmp_path):
    from grb_project.separate_spectr import redraw_bandbb_fluxdata_outputs

    called = {}

    def fake_redraw_all(fluxdata_dir, output_dir, bnname):
        called["fluxdata_dir"] = fluxdata_dir
        called["output_dir"] = output_dir
        called["bnname"] = bnname
        return {"single_plots": [], "overview": []}

    monkeypatch.setattr("grb_project.separate_spectr.redraw_all_fluxdata_spectra", fake_redraw_all)
    redraw_bandbb_fluxdata_outputs(
        bnname="GRB231129C",
        result_root=tmp_path,
    )

    assert called["bnname"] == "GRB231129C"
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_separate_spectr_fluxdata_redraw.py::test_redraw_entrypoint_uses_default_bandbb_fluxdata_path -v`
预期：FAIL，因为入口函数还不存在。

- [ ] **步骤 3：编写最少实现代码**

增加一个便于上层调用的入口，默认指向本次用户给定的目录结构：

```python
def redraw_bandbb_fluxdata_outputs(bnname: str, result_root: Path) -> dict[str, list[Path]]:
    fluxdata_dir = result_root / bnname / "band+bb" / "fluxdata"
    output_dir = result_root / bnname / "band+bb" / "fluxdata"
    return redraw_all_fluxdata_spectra(fluxdata_dir=fluxdata_dir, output_dir=output_dir, bnname=bnname)
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_separate_spectr_fluxdata_redraw.py::test_redraw_entrypoint_uses_default_bandbb_fluxdata_path -v`
预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add separate_spectr.py tests/test_separate_spectr_fluxdata_redraw.py README.md
if git diff --cached --quiet; then
  git add separate_spectr.py tests/test_separate_spectr_fluxdata_redraw.py
fi
git commit -m "docs(plot): document fluxdata redraw entrypoint"
```

---

## 计划自检

### 1. 规格覆盖度

- “利用 `fluxdata` 里的数据重新绘制图像” → 任务 2、3、4、5 覆盖
- “这里面的数据”指 `band+bb` 导出的四类探测器文件 → 任务 2 的分组规则覆盖
- “重新绘制图像”且保持当前样式 → 任务 1、3、4 复用现有轴样式与绘图约定
- “所有时间段” → 任务 4 的批量处理覆盖
- “单张图 + 总览图” → 任务 4 明确覆盖

### 2. 占位符扫描

已检查以下高风险词：
- 没有 `TODO`
- 没有 `待定`
- 没有 `后续实现`
- 没有空泛的“添加适当错误处理”之类描述
- 每个步骤都给出可执行的测试/实现/提交内容

### 3. 类型一致性

- `discover_fluxdata_groups(fluxdata_dir: Path) -> dict[str, dict[str, Path]]`
- `redraw_fluxdata_timebin(timebin: str, fluxdata_dir: Path, output_dir: Path, bnname: str) -> Path`
- `redraw_all_fluxdata_spectra(fluxdata_dir: Path, output_dir: Path, bnname: str) -> dict[str, list[Path]]`
- `redraw_bandbb_fluxdata_outputs(bnname: str, result_root: Path) -> dict[str, list[Path]]`

这些签名在后续任务中保持一致，避免前后不一致的问题。

---

## 交接执行选项

计划已完成并保存到 `docs/superpowers/plans/2026-06-29-fluxdata-batch-redraw.md`。两种执行方式：

1. **子代理驱动（推荐）** - 每个任务调度一个新的子代理，任务间进行审查，快速迭代
2. **内联执行** - 在当前会话中使用 `executing-plans` 执行任务，批量执行并设有检查点供审查

选哪种方式？
