from __future__ import annotations

import json
from pathlib import Path
import time
import unittest

from grb_project.joint_selection import (
    JointSelectionCriteria,
    JointSelectionResult,
    JointTarget,
    restrict_result_targets,
)
from grb_project.web_app import _build_joint_batch_command


def _target(bn, grb):
    return JointTarget(bn, grb, 2000 + int(bn[2:4]), 6.0e8, 0.0, 100.0, 10.0, 20.0, "flgc_tl", 25.0)


class RestrictResultTargetsTests(unittest.TestCase):
    def _result(self):
        return JointSelectionResult(
            criteria=JointSelectionCriteria(),
            targets=[_target("bn220101215", "GRB220101A"), _target("bn081001234", "GRB081001A")],
            excluded={"LLE-only 探测": ["bnX"]},
        )

    def test_restrict_keeps_subset_in_sorted_order(self):
        result = restrict_result_targets(self._result(), ["bn081001234"])
        self.assertEqual(result.bnnames, ["bn081001234"])

    def test_restrict_accepts_uppercase_and_dedupes(self):
        result = restrict_result_targets(self._result(), ["BN081001234", "bn081001234"])
        self.assertEqual(result.bnnames, ["bn081001234"])

    def test_restrict_rejects_unknown(self):
        with self.assertRaises(ValueError) as ctx:
            restrict_result_targets(self._result(), ["bn999999999"])
        self.assertIn("bn999999999", str(ctx.exception))


class BuildJointBatchCommandTests(unittest.TestCase):
    def test_minimal_command_shape(self):
        cmd = _build_joint_batch_command(
            analysis_mode="gbm+lat",
            result_root="/tmp/results_joint",
            summary_csv_name="summary_results.csv",
            models=["band", "comp"],
        )
        self.assertIn("select", cmd)
        self.assertIn("--run", cmd)
        self.assertEqual(cmd[cmd.index("--result-root") + 1], "/tmp/results_joint")
        self.assertEqual(cmd[cmd.index("--models") + 1], "band")
        self.assertEqual(cmd[cmd.index("--models") + 2], "comp")
        self.assertNotIn("--year-from", cmd)
        self.assertNotIn("--only", cmd)

    def test_full_command_flags(self):
        cmd = _build_joint_batch_command(
            analysis_mode="lat",
            result_root="/tmp/r",
            summary_csv_name="s.csv",
            models=["band"],
            year_from=2008,
            year_to=2017,
            min_lat_ts=25.0,
            include_lle_only=True,
            only=["bn081001234", "bn130502345"],
        )
        self.assertEqual(cmd[cmd.index("--year-from") + 1], "2008")
        self.assertEqual(cmd[cmd.index("--year-to") + 1], "2017")
        self.assertEqual(cmd[cmd.index("--min-lat-ts") + 1], "25")
        self.assertIn("--include-lle-only", cmd)
        self.assertEqual(
            cmd[cmd.index("--only") + 1: cmd.index("--only") + 3],
            ["bn081001234", "bn130502345"],
        )
        self.assertIn("--session-log", cmd)


class JointBatchPidAliveTests(unittest.TestCase):
    def test_nonexistent_pid_is_dead(self):
        from grb_project.joint_selection import batch_pid_alive

        self.assertFalse(batch_pid_alive(999999999))

    def test_self_pid_is_alive(self):
        import os

        from grb_project.joint_selection import batch_pid_alive

        self.assertTrue(batch_pid_alive(os.getpid()))

    def test_zombie_pid_is_dead(self):
        import os
        import time

        from grb_project.joint_selection import batch_pid_alive

        pid = os.fork()
        if pid == 0:
            os._exit(0)
        try:
            # fork 到子进程真正进入 Z 状态有微秒级窗口，稍等它完成退出
            deadline = time.time() + 2.0
            while time.time() < deadline and batch_pid_alive(pid):
                time.sleep(0.01)
            self.assertFalse(batch_pid_alive(pid))
        finally:
            os.waitpid(pid, 0)


class BatchStopTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        from grb_project.session import session

        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self._old_state = session.joint_batch_run_state_file
        self._old_flag = session.stop_flag_file
        session.joint_batch_run_state_file = str(root / "joint_batch_run.json")
        session.stop_flag_file = str(root / "joint_batch_stop.flag")

    def tearDown(self):
        from grb_project.session import session

        session.joint_batch_run_state_file = self._old_state
        session.stop_flag_file = self._old_flag
        self._tmp.cleanup()

    def _write_state(self, pid):
        from grb_project.joint_selection import joint_batch_state_path

        state = {"pid": pid, "cmd": ["sleep"], "targets": 1}
        joint_batch_state_path().write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return state

    def test_request_stop_writes_flag_and_updates_state(self):
        import os

        from grb_project.joint_selection import (
            joint_batch_stop_flag_path,
            read_running_batch,
            request_batch_stop,
        )

        self._write_state(os.getpid())
        state = request_batch_stop()
        self.assertTrue(joint_batch_stop_flag_path().exists())
        self.assertTrue(state["stop_requested"])
        self.assertTrue(read_running_batch()["stop_requested"])

    def test_request_stop_without_running_batch_raises(self):
        from grb_project.joint_selection import request_batch_stop

        with self.assertRaises(RuntimeError):
            request_batch_stop()

    def test_force_stop_kills_group_leader_process(self):
        import subprocess

        from grb_project.joint_selection import (
            batch_pid_alive,
            force_stop_batch,
            joint_batch_stop_flag_path,
        )

        proc = subprocess.Popen(
            ["sleep", "30"], stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            self._write_state(proc.pid)
            joint_batch_stop_flag_path().write_text("{}", encoding="utf-8")
            state = force_stop_batch()
            self.assertEqual(state["pid"], proc.pid)
            self.assertTrue(state["force_stopped"])
            deadline = time.time() + 3.0
            while time.time() < deadline and batch_pid_alive(proc.pid):
                time.sleep(0.05)
            self.assertFalse(batch_pid_alive(proc.pid))
            self.assertFalse(joint_batch_stop_flag_path().exists())
        finally:
            proc.kill()
            proc.wait()

    def test_force_stop_on_dead_pid_reports_no_kill(self):
        from grb_project.joint_selection import (
            force_stop_batch,
            joint_batch_state_path,
        )

        self._write_state(999999999)
        state = force_stop_batch()
        self.assertFalse(state["kill_attempted"])
        self.assertNotIn("force_stopped", state)
        saved = json.loads(joint_batch_state_path().read_text(encoding="utf-8"))
        self.assertNotIn("force_stopped", saved)

    def test_check_stop_requested_raises_on_flag(self):
        from grb_project.joint_selection import joint_batch_stop_flag_path
        from grb_project.session import AnalysisCancelled, check_stop_requested

        joint_batch_stop_flag_path().write_text("{}", encoding="utf-8")
        with self.assertRaises(AnalysisCancelled):
            check_stop_requested()


class SelectedBnnamesTests(unittest.TestCase):
    def test_default_all_selected(self):
        from grb_project.web_app import _selected_bnnames

        self.assertEqual(
            _selected_bnnames(["bn081001234", "bn130502345"], {}),
            ["bn081001234", "bn130502345"],
        )

    def test_unchecked_excluded_and_order_preserved(self):
        from grb_project.web_app import _selected_bnnames

        state = {"joint_pick_bn081001234": False, "joint_pick_bn130502345": True}
        self.assertEqual(
            _selected_bnnames(["bn130502345", "bn081001234", "bn220101215"], state),
            ["bn130502345", "bn220101215"],
        )

    def test_default_false_respects_missing_keys(self):
        from grb_project.web_app import _selected_bnnames

        state = {"joint_pick_bn081001234": True}
        self.assertEqual(
            _selected_bnnames(["bn081001234", "bn220101215"], state, default=False),
            ["bn081001234"],
        )


if __name__ == "__main__":
    unittest.main()
