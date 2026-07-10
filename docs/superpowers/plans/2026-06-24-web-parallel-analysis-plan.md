# `web` 并行分析改造实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 让 `grb_project` 的 `web` 页面中“一次性分析”和“一键批量分析”可以同时运行，且互不阻塞、互不覆盖结果。

**架构：** `web_app.py` 只负责提交任务和展示状态；新增 SQLite 任务存储记录任务元数据与状态；新增独立 worker 进程从数据库取出待执行任务并调用现有分析入口。每个任务使用独立输出目录与任务级锁，保留现有 `GRBProject.run()` 和 `run_joint_band_batch()` 的核心逻辑，避免重写分析管线。

**技术栈：** Streamlit、SQLite、Python `multiprocessing` / `subprocess`、现有 `grb_project` 分析模块、现有日志与文件输出约定。

---

## 文件结构

### 新增文件
- `grb_project/task_store.py`：SQLite 任务表、任务读写、状态迁移、日志字段更新。
- `grb_project/task_worker.py`：后台 worker 主循环，拉取任务并执行单次/批量分析。
- `grb_project/task_models.py`：任务数据结构与状态常量，避免在多个模块里重复字符串。
- `grb_project/task_paths.py`：任务目录规范、任务级锁路径、输出目录构造函数。
- `grb_project/web_tasks.py`：`web_app.py` 使用的任务提交与轮询封装，保持 UI 代码薄。
- `tests/test_task_store.py`：任务存储与状态迁移测试。
- `tests/test_task_paths.py`：任务目录与锁路径生成测试。
- `tests/test_web_task_flow.py`：任务提交逻辑与状态轮询逻辑的轻量测试。

### 修改文件
- `grb_project/web_app.py`：将单次分析/批量分析从同步执行改为提交任务；新增任务列表与刷新显示。
- `grb_project/batch_analyze.py`：保留批量分析核心逻辑，但允许由 worker 传入任务级 result root 或任务上下文。
- `grb_project/project.py`：补充任务级输出目录支持所需的最小接口，不改变核心分析算法。
- `grb_project/logging_utils.py`：如现有日志适合扩展，补充按任务 ID 写日志的最小能力。
- `grb_project/session.py`：如需要，增加任务队列根目录与任务数据库路径的会话默认值。
- `grb_project/说明文档.md`：补充 web 并行与任务化运行方式说明。

---

## 任务 1：定义任务模型与 SQLite 存储

**文件：**
- 创建：`grb_project/task_models.py`
- 创建：`grb_project/task_store.py`
- 测试：`tests/test_task_store.py`

- [ ] **步骤 1：编写失败的测试**

```python
from pathlib import Path

from grb_project.task_store import TaskStore
from grb_project.task_models import TaskStatus, TaskType


def test_create_update_and_load_task(tmp_path: Path):
    db_path = tmp_path / "tasks.sqlite"
    store = TaskStore(db_path)

    task_id = store.create_task(
        task_type=TaskType.SINGLE_ANALYSIS,
        payload={"target": "bn221009888"},
        result_root=str(tmp_path / "results"),
    )

    task = store.get_task(task_id)
    assert task.task_type == TaskType.SINGLE_ANALYSIS
    assert task.status == TaskStatus.PENDING
    assert task.payload["target"] == "bn221009888"

    store.update_status(task_id, TaskStatus.RUNNING)
    task = store.get_task(task_id)
    assert task.status == TaskStatus.RUNNING
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_task_store.py -v`
预期：FAIL，提示 `TaskStore` / `TaskStatus` / `TaskType` 未定义。

- [ ] **步骤 3：编写最少实现代码**

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
import json
import sqlite3
from typing import Any


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class TaskType(StrEnum):
    SINGLE_ANALYSIS = "single_analysis"
    BATCH_ANALYSIS = "batch_analysis"


@dataclass(slots=True)
class TaskRecord:
    task_id: str
    task_type: TaskType
    status: TaskStatus
    payload: dict[str, Any]
    result_root: str


class TaskStore:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    result_root TEXT NOT NULL
                )
                """
            )

    def create_task(self, *, task_type: TaskType, payload: dict[str, Any], result_root: str) -> str:
        import uuid

        task_id = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO tasks(task_id, task_type, status, payload_json, result_root) VALUES (?, ?, ?, ?, ?)",
                (task_id, task_type.value, TaskStatus.PENDING.value, json.dumps(payload, ensure_ascii=False), result_root),
            )
        return task_id

    def get_task(self, task_id: str) -> TaskRecord:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT task_id, task_type, status, payload_json, result_root FROM tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        if row is None:
            raise KeyError(task_id)
        return TaskRecord(
            task_id=row[0],
            task_type=TaskType(row[1]),
            status=TaskStatus(row[2]),
            payload=json.loads(row[3]),
            result_root=row[4],
        )

    def update_status(self, task_id: str, status: TaskStatus) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE tasks SET status = ? WHERE task_id = ?", (status.value, task_id))
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_task_store.py -v`
预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add grb_project/task_models.py grb_project/task_store.py tests/test_task_store.py
git commit -m "feat: add sqlite-backed task store"
```

---

## 任务 2：定义任务目录、锁与输出隔离规则

**文件：**
- 创建：`grb_project/task_paths.py`
- 修改：`grb_project/project.py`
- 修改：`grb_project/batch_analyze.py`
- 测试：`tests/test_task_paths.py`

- [ ] **步骤 1：编写失败的测试**

```python
from pathlib import Path

from grb_project.task_paths import task_root_dir, task_lock_path


def test_task_root_and_lock_path(tmp_path: Path):
    root = task_root_dir(tmp_path / "results", "task_123")
    assert root == tmp_path / "results" / "tasks" / "task_123"
    assert task_lock_path(tmp_path / "results", "task_123") == root / ".lock"
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_task_paths.py -v`
预期：FAIL，提示 `task_root_dir` / `task_lock_path` 未定义。

- [ ] **步骤 3：编写最少实现代码**

```python
from pathlib import Path


def task_root_dir(result_root: Path | str, task_id: str) -> Path:
    return Path(result_root) / "tasks" / task_id


def task_lock_path(result_root: Path | str, task_id: str) -> Path:
    return task_root_dir(result_root, task_id) / ".lock"
```

并在 `project.py` / `batch_analyze.py` 中把任务输出目录作为可注入参数传递，默认仍保留旧行为，不破坏现有 CLI。

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_task_paths.py -v`
预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add grb_project/task_paths.py grb_project/project.py grb_project/batch_analyze.py tests/test_task_paths.py
git commit -m "feat: isolate task output directories"
```

---

## 任务 3：实现后台 worker 执行单次与批量任务

**文件：**
- 创建：`grb_project/task_worker.py`
- 修改：`grb_project/batch_analyze.py`
- 修改：`grb_project/project.py`
- 测试：`tests/test_task_worker.py`

- [ ] **步骤 1：编写失败的测试**

```python
from pathlib import Path

from grb_project.task_store import TaskStore
from grb_project.task_models import TaskType, TaskStatus
from grb_project.task_worker import run_one_task


def test_worker_marks_task_success(monkeypatch, tmp_path: Path):
    store = TaskStore(tmp_path / "tasks.sqlite")
    task_id = store.create_task(
        task_type=TaskType.SINGLE_ANALYSIS,
        payload={"target": "bn221009888"},
        result_root=str(tmp_path / "results"),
    )

    monkeypatch.setattr("grb_project.task_worker.execute_single_analysis", lambda *_args, **_kwargs: None)
    run_one_task(store, task_id)

    task = store.get_task(task_id)
    assert task.status == TaskStatus.SUCCESS
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_task_worker.py -v`
预期：FAIL，提示 `run_one_task` / `execute_single_analysis` 未定义。

- [ ] **步骤 3：编写最少实现代码**

```python
from __future__ import annotations

from .task_models import TaskStatus, TaskType
from .task_store import TaskStore


def execute_single_analysis(*_args, **_kwargs) -> None:
    raise NotImplementedError


def execute_batch_analysis(*_args, **_kwargs) -> None:
    raise NotImplementedError


def run_one_task(store: TaskStore, task_id: str) -> None:
    task = store.get_task(task_id)
    store.update_status(task_id, TaskStatus.RUNNING)
    try:
        if task.task_type == TaskType.SINGLE_ANALYSIS:
            execute_single_analysis(task)
        elif task.task_type == TaskType.BATCH_ANALYSIS:
            execute_batch_analysis(task)
        else:
            raise ValueError(task.task_type)
        store.update_status(task_id, TaskStatus.SUCCESS)
    except Exception:
        store.update_status(task_id, TaskStatus.FAILED)
        raise
```

然后把真实执行逻辑接到：

```python
# single analysis
GRBProject(cfg).run(...)

# batch analysis
run_joint_band_batch(...)
```

并确保 worker 使用独立任务目录作为 `result_root`。

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_task_worker.py -v`
预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add grb_project/task_worker.py grb_project/batch_analyze.py grb_project/project.py tests/test_task_worker.py
git commit -m "feat: add background task worker"
```

---

## 任务 4：改造 `web_app.py` 为“提交任务 + 查询状态”

**文件：**
- 创建：`grb_project/web_tasks.py`
- 修改：`grb_project/web_app.py`
- 修改：`grb_project/session.py`
- 测试：`tests/test_web_task_flow.py`

- [ ] **步骤 1：编写失败的测试**

```python
from grb_project.web_tasks import submit_single_analysis_task


def test_submit_single_analysis_task_returns_task_id(monkeypatch):
    monkeypatch.setattr("grb_project.web_tasks.get_task_store", lambda: FakeStore())
    task_id = submit_single_analysis_task(target="bn221009888", result_root="/tmp/results")
    assert task_id == "task-1"
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_web_task_flow.py -v`
预期：FAIL，提示 `submit_single_analysis_task` 未定义。

- [ ] **步骤 3：编写最少实现代码**

```python
from .task_models import TaskType
from .task_store import TaskStore


def get_task_store() -> TaskStore:
    raise NotImplementedError


def submit_single_analysis_task(*, target: str, result_root: str, payload: dict | None = None) -> str:
    store = get_task_store()
    task_payload = {"target": target, **(payload or {})}
    return store.create_task(task_type=TaskType.SINGLE_ANALYSIS, payload=task_payload, result_root=result_root)
```

`web_app.py` 中把原来的“直接执行”改成：

- 写任务
- 展示 task_id
- 定时刷新任务状态
- 在页面上显示最近任务列表

批量按钮改成批量创建任务或创建父任务 + 子任务列表，先保证最小版本可用：**每个目标一个独立任务**。

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_web_task_flow.py -v`
预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add grb_project/web_tasks.py grb_project/web_app.py grb_project/session.py tests/test_web_task_flow.py
git commit -m "feat: submit web analyses as background tasks"
```

---

## 任务 5：补充任务状态展示、日志与失败恢复说明

**文件：**
- 修改：`grb_project/web_app.py`
- 修改：`grb_project/logging_utils.py`
- 修改：`grb_project/说明文档.md`
- 测试：`tests/test_task_store.py` 或新增针对状态读取的测试

- [ ] **步骤 1：编写失败的测试**

```python
from pathlib import Path

from grb_project.task_store import TaskStore
from grb_project.task_models import TaskStatus, TaskType


def test_store_persists_failure_reason(tmp_path: Path):
    store = TaskStore(tmp_path / "tasks.sqlite")
    task_id = store.create_task(
        task_type=TaskType.SINGLE_ANALYSIS,
        payload={"target": "bn221009888"},
        result_root=str(tmp_path / "results"),
    )
    store.update_status(task_id, TaskStatus.FAILED)
    task = store.get_task(task_id)
    assert task.status == TaskStatus.FAILED
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_task_store.py -v`
预期：如果状态未正确持久化则失败。

- [ ] **步骤 3：编写最少实现代码**

在 SQLite 表中增加：

- `error_text`
- `started_at`
- `finished_at`
- `updated_at`

并在 `web_app.py` 中增加任务列表与状态刷新区域，至少展示：

- `task_id`
- `task_type`
- `target`
- `status`
- `result_root`
- `error_text`

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_task_store.py -v`
预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add grb_project/web_app.py grb_project/logging_utils.py grb_project/说明文档.md tests/test_task_store.py
git commit -m "docs: describe web task execution flow"
```

---

## 验证清单

在所有任务完成后，按以下顺序验证：

1. `pytest tests/test_task_store.py -v`
2. `pytest tests/test_task_paths.py -v`
3. `pytest tests/test_task_worker.py -v`
4. `pytest tests/test_web_task_flow.py -v`
5. 手动启动 `streamlit run grb_project/web_app.py`，在同一页面同时提交一次单次分析和一组批量任务，确认：
   - 两类任务都进入 `pending` / `running`
   - 页面不会卡死在单个请求上
   - 不同任务输出目录互不覆盖
   - 失败任务能在 UI 上看到错误信息

---

## 实施顺序建议

1. 先做 `task_store`，把状态持久化打牢。
2. 再做 `task_worker`，保证能在后台跑起来。
3. 再改 `web_app.py`，让 UI 只提交任务。
4. 最后补状态展示、日志与文档。

这样即使中途停掉，也能保留已经落库的任务状态，不会把现有分析入口一次性打坏。
