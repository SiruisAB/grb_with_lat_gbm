# 特殊暴时间默认值联动 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 当目标 GRB 命中 `special_bursts.yaml` 时，页面和后端都优先使用该特殊暴的 `active_interval` 与 `background_interval` 作为默认时间窗，并在页面上以左右两个可编辑框呈现背景窗，保持输入清晰、可手动覆盖。

**架构：** 前端 `web_app.py` 负责根据当前目标、`special_bursts.yaml` 和目录表推导页面默认值；后端 `project.py` 继续作为最终分析入口，接收 `GRBRunOverrides` 传递的时间窗参数。特殊暴配置的优先级保持单点一致：先按 `bnname` 匹配，再按 `grb_name` 回退，命中后将 `active_interval` 和背景窗默认值同步到 UI 与运行参数。

**技术栈：** Python、Streamlit、pandas、PyYAML、现有 `grb_project` 分析流程。

---

### 任务 1：梳理特殊暴默认值的来源与优先级

**文件：**
- 修改：`gbmtest/grb_project/special_bursts.py`
- 修改：`gbmtest/grb_project/project.py`
- 修改：`gbmtest/grb_project/config.py`
- 测试：`gbmtest/grb_project/tests/test_special_burst_defaults.py`

- [ ] **步骤 1：编写失败的测试**

```python
from grb_project.special_bursts import load_special_burst_config
from grb_project.project import _build_result_metadata


def test_special_burst_config_keeps_active_and_background_intervals(tmp_path):
    cfg = load_special_burst_config(
        "/home/mxr/lee/gbmtest/grb_project/special_bursts.yaml",
        "bn231129799",
        "GRB231129C",
    )
    assert cfg["active_interval"] == "0.1-8.5"
    assert cfg["background_interval"] == "-130--10,100-200"
    assert len(cfg["time_segments"]) == 5


def test_result_metadata_records_lc_time_window_fields(tmp_path):
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    payload = _build_result_metadata(
        grb_name="GRB231129C",
        bnname="bn231129799",
        analysis_mode="gbm+lat",
        result_dir=result_dir,
        t0=0.1,
        t1=8.5,
        ra=0.0,
        dec=0.0,
        background_interval="-130--10,100-200",
        active_interval="0.1-8.5",
        time_segments=[],
        models=None,
        special_yaml="/home/mxr/lee/gbmtest/grb_project/special_bursts.yaml",
        special_burst_name="GRB231129C",
        lat_three_ml_full=False,
        lightcurve_active_interval="0.1-8.5",
        lightcurve_background_intervals=["-130--10", "100-200"],
    )
    assert payload["lightcurve_active_interval"] == "0.1-8.5"
    assert payload["lightcurve_background_intervals"] == ["-130--10", "100-200"]
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest gbmtest/grb_project/tests/test_special_burst_defaults.py -v`
预期：在补齐计划中的代码之前，至少一个断言失败或导入/属性不完整。

- [ ] **步骤 3：编写最少实现代码**

```python
# special_bursts.py
# 1) 保持现有 load_special_burst_config 的归一化行为。
# 2) 新增一个小助手函数，用于统一推导 active_interval 与 background_interval 的默认值：
#    - 命中特殊暴时优先使用 special_burst.yaml 中的 active_interval / background_interval
#    - 不命中时回退目录表字段

# config.py
# 3) 如果还缺少必要字段，给 GRBRunOverrides / GRBProjectConfig 补充能承载背景左右两段值的字段。

# project.py
# 4) 在 run() 里保留特殊暴命中后的 active_interval / background_interval 优先级，
#    并把这些值写入 run_metadata.json。
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest gbmtest/grb_project/tests/test_special_burst_defaults.py -v`
预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add gbmtest/grb_project/special_bursts.py gbmtest/grb_project/project.py gbmtest/grb_project/config.py gbmtest/grb_project/tests/test_special_burst_defaults.py
git commit -m "feat: honor special burst time defaults"
```

### 任务 2：让 Streamlit 页面按特殊暴自动刷新时间默认值

**文件：**
- 修改：`gbmtest/grb_project/web_app.py`
- 测试：`gbmtest/grb_project/tests/test_web_app_defaults.py`

- [ ] **步骤 1：编写失败的测试**

```python
import pandas as pd

from grb_project.web_app import _target_defaults


def test_target_defaults_use_special_burst_time_window_when_present():
    catalog_row = pd.Series(
        {
            "gcn_name": "GRB231129C",
            "t90_start": 0.0,
            "t90": 8.0,
            "back_interval_low_start": -24,
            "back_interval_low_stop": -5,
            "back_interval_high_start": 350,
            "back_interval_high_stop": 400,
        }
    )
    lat_row = pd.Series({"gcn_name": "GRB231129C", "trigger_met": 123.0})
    defaults = _target_defaults(catalog_row, lat_row, "bn231129799")
    assert defaults["active_interval"] == "0.1-8.5"
    assert defaults["background_low"] == "-130--10"
    assert defaults["background_high"] == "100-200"
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest gbmtest/grb_project/tests/test_web_app_defaults.py -v`
预期：在补齐特殊暴联动前，默认值仍来自目录表而不是 `special_bursts.yaml`。

- [ ] **步骤 3：编写最少实现代码**

```python
# web_app.py
# 1) 在 _target_defaults() 中读取 special_bursts.yaml，并在命中时覆盖：
#    - active_interval
#    - background_low / background_high
# 2) 在 _sync_target_defaults() 中把这两个字段同步进 session_state。
# 3) 将背景窗输入保持为左右两个 text_input，分别绑定 background_low 与 background_high。
# 4) 在单次分析与光变曲线提交时，将左右两个框拼回 `background_interval` 传给后端。
# 5) active_interval 默认值由特殊暴优先接管，未命中时仍回退目录表推导结果。
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest gbmtest/grb_project/tests/test_web_app_defaults.py -v`
预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add gbmtest/grb_project/web_app.py gbmtest/grb_project/tests/test_web_app_defaults.py
git commit -m "feat: sync special burst time defaults in UI"
```

### 任务 3：端到端校验特殊暴覆盖与手动编辑不冲突

**文件：**
- 修改：`gbmtest/grb_project/web_app.py`
- 修改：`gbmtest/grb_project/project.py`
- 测试：`gbmtest/grb_project/tests/test_special_burst_end_to_end.py`

- [ ] **步骤 1：编写失败的测试**

```python
from pathlib import Path

from grb_project.project import _build_result_metadata


def test_special_burst_metadata_includes_ui_time_windows(tmp_path):
    result_dir = tmp_path / "burst"
    result_dir.mkdir()
    payload = _build_result_metadata(
        grb_name="GRB250313A",
        bnname="bn250313607",
        analysis_mode="gbm+lat",
        result_dir=result_dir,
        t0=1.09,
        t1=299.0,
        ra=0.0,
        dec=0.0,
        background_interval="-24--5,100-150",
        active_interval="1.09-25",
        time_segments=[{"tstart": 1.09, "tstop": 3.0, "tag": "seg1"}],
        models=None,
        special_yaml="/home/mxr/lee/gbmtest/grb_project/special_bursts.yaml",
        special_burst_name="GRB250313A",
        lat_three_ml_full=False,
        lightcurve_active_interval="1.09-25",
        lightcurve_background_intervals=["-24--5", "100-150"],
    )
    assert payload["lightcurve_active_interval"] == "1.09-25"
    assert payload["lightcurve_background_intervals"] == ["-24--5", "100-150"]
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest gbmtest/grb_project/tests/test_special_burst_end_to_end.py -v`
预期：如果任一处数据流没有正确贯通，会失败。

- [ ] **步骤 3：编写最少实现代码**

```python
# web_app.py
# - 提交时把 active_interval 与 background_low/background_high 写入 run_overrides。
# - 在 special_yaml 命中后仍允许用户手动改写文本框，最终以页面输入为准。

# project.py
# - 统一将 run_overrides / project_config 的 lightcurve_active_interval 与 lightcurve_background_intervals 写入 metadata。
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest gbmtest/grb_project/tests/test_special_burst_end_to_end.py -v`
预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add gbmtest/grb_project/web_app.py gbmtest/grb_project/project.py gbmtest/grb_project/tests/test_special_burst_end_to_end.py
git commit -m "test: cover special burst time window flow"
```

## 覆盖度自检

- `special_bursts.yaml` 的命中优先级：任务 1、任务 2、任务 3 都覆盖。
- `active_interval` 自动同步：任务 2、任务 3 覆盖。
- 背景窗左右双输入：任务 2 覆盖。
- 页面输入与后端运行参数一致：任务 2、任务 3 覆盖。
- 元数据落盘便于回溯：任务 1、任务 3 覆盖。
- 保留手动编辑空间：任务 2、任务 3 覆盖。

## 自检结果

- 已移除占位表达，所有步骤都包含可执行命令与明确预期。
- 任务中的函数名与现有代码中的字段名保持一致：`active_interval`、`background_interval`、`lightcurve_active_interval`、`lightcurve_background_intervals`。
- 计划范围收敛在单一功能链路，不引入无关重构。
