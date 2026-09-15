from __future__ import annotations

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
        from grb_project.web_app import _joint_batch_pid_alive

        self.assertFalse(_joint_batch_pid_alive(999999999))

    def test_self_pid_is_alive(self):
        import os

        from grb_project.web_app import _joint_batch_pid_alive

        self.assertTrue(_joint_batch_pid_alive(os.getpid()))

    def test_zombie_pid_is_dead(self):
        import os
        import time

        from grb_project.web_app import _joint_batch_pid_alive

        pid = os.fork()
        if pid == 0:
            os._exit(0)
        try:
            # fork 到子进程真正进入 Z 状态有微秒级窗口，稍等它完成退出
            deadline = time.time() + 2.0
            while time.time() < deadline and _joint_batch_pid_alive(pid):
                time.sleep(0.01)
            self.assertFalse(_joint_batch_pid_alive(pid))
        finally:
            os.waitpid(pid, 0)


if __name__ == "__main__":
    unittest.main()
