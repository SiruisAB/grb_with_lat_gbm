from __future__ import annotations

from pathlib import Path
from unittest import mock
import importlib
import unittest


FORBIDDEN_TOKENS = {
    "task_worker",
    "web_tasks",
    "task_store",
    "task_models",
    "batch_analyze",
    "cli.py",
    "__main__.py",
    "task_paths",
    "TaskStore",
    "TaskStatus",
    "TaskType",
    "submit_single_analysis_task",
    "submit_batch_analysis_task",
    "run_worker_loop",
    "task_db_path",
    "run_gcn_batch",
}


class SmokeTests(unittest.TestCase):
    def test_pipeline_imports_and_dispatches(self) -> None:
        with mock.patch.dict("sys.modules", {"threeML": mock.MagicMock()}):
            module = importlib.import_module("grb_project.pipeline")
        self.assertTrue(callable(module.main))

    def test_web_app_imports_and_exposes_main(self) -> None:
        with mock.patch.dict("sys.modules", {"threeML": mock.MagicMock()}):
            module = importlib.import_module("grb_project.web_app")
        self.assertTrue(callable(module.main))

    def test_worker_era_modules_and_symbols_are_removed(self) -> None:
        package_root = Path(__file__).resolve().parents[1]
        source_files = sorted(package_root.glob("*.py"))
        source_text = "\n".join(path.read_text(encoding="utf-8") for path in source_files)

        for token in FORBIDDEN_TOKENS:
            self.assertNotIn(token, source_text, msg=f"unexpected leftover token: {token}")

        for filename in (
            "task_worker.py",
            "web_tasks.py",
            "task_store.py",
            "task_models.py",
            "batch_analyze.py",
            "cli.py",
            "__main__.py",
            "task_paths.py",
        ):
            self.assertFalse((package_root / filename).exists(), msg=f"unexpected leftover file: {filename}")

    def test_special_burst_loader_normalizes_time_fields(self) -> None:
        from grb_project.special_bursts import load_special_burst_config

        burst = load_special_burst_config(None, "bn231129799", "GRB231129C")

        self.assertEqual(burst["active_interval"], "0.1-8.5")
        self.assertEqual(burst["background_interval"], "-130--10,100-200")
        self.assertEqual(burst["time_segments"][0]["start"], 0.1)
        self.assertEqual(burst["time_segments"][-1]["stop"], 8.5)

    def test_special_burst_time_fields_override_analysis_windows(self) -> None:
        from grb_project.analyze_single import _segments_from_special_burst

        burst = {
            "active_interval": "0.1-8.5",
            "background_interval": "-130--10,100-200",
            "time_segments": [
                {"name": "seg1", "start": 0.1, "stop": 1.0},
                {"name": "seg2", "start": 1.0, "stop": 8.5},
            ],
        }
        fallback = [{"tstart": 9.0, "tstop": 10.0, "tag": "fallback"}]

        segments = _segments_from_special_burst(burst, fallback)

        self.assertEqual(segments[0]["tstart"], 0.1)
        self.assertEqual(segments[-1]["tstop"], 8.5)

    def test_used_segments_records_special_burst_time_fields(self) -> None:
        from grb_project.special_bursts import _load_special_burst_config

        burst = _load_special_burst_config(None, "bn231129799", "GRB231129C")
        self.assertEqual(burst["active_interval"], "0.1-8.5")
        self.assertEqual(burst["background_interval"], "-130--10,100-200")

    def test_special_burst_loader_imports_without_threeML(self) -> None:
        module = importlib.import_module("grb_project.special_bursts")
        self.assertTrue(callable(module._load_special_burst_config))

    def test_lat_processing_resolves_grb_name_directory(self) -> None:
        module = importlib.import_module("grb_project.lat_processing")
        resolved = module._resolve_lat_data_dir("GRB231129C", "/tmp/extended")
        self.assertEqual(resolved, "/tmp/extended/GRB231129C")


if __name__ == "__main__":
    unittest.main()
