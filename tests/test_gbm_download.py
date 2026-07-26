from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import mock
import unittest

import pandas as pd


class FakeFTP:
    def __init__(self) -> None:
        self.cwd_calls: list[str] = []
        self.retrieved: list[str] = []

    def voidcmd(self, command: str) -> str:
        assert command == "NOOP"
        return "OK"

    def cwd(self, remote_dir: str) -> None:
        self.cwd_calls.append(remote_dir)

    def nlst(self) -> list[str]:
        return ["existing.fit", "new.fit"]

    def retrbinary(self, command: str, callback) -> None:
        self.retrieved.append(command)
        callback(b"payload")


class InterruptedFTP(FakeFTP):
    def __init__(self) -> None:
        super().__init__()
        self.failed_once = False

    def nlst(self) -> list[str]:
        return ["new.fit"]

    def retrbinary(self, command: str, callback) -> None:
        self.retrieved.append(command)
        if not self.failed_once:
            self.failed_once = True
            callback(b"partial")
            raise ConnectionError("connection lost")
        callback(b"complete")


class GBMDownloadTests(unittest.TestCase):
    def test_joint_targets_compare_four_digit_year_range(self) -> None:
        from grb_project.gbm_download import get_joint_target_list

        lat_catalog = pd.DataFrame({"GRBname": ["GRB080825C", "GRB231129C"]})
        gbm_catalog = pd.DataFrame(
            {"trigger_name": ["bn080825593", "bn231129799"]}
        )
        with mock.patch(
            "grb_project.gbm_download.pd.read_csv", return_value=lat_catalog
        ), mock.patch(
            "grb_project.gbm_download.pd.read_excel", return_value=gbm_catalog
        ):
            targets = get_joint_target_list(
                Path("/tmp/lat.csv"),
                Path("/tmp/gbm.xls"),
                year_from=2008,
                year_to=2022,
            )

        self.assertEqual(targets, ["bn080825593"])

    def test_normalize_bnname_rejects_path_traversal(self) -> None:
        from grb_project.gbm_download import normalize_bnname

        with self.assertRaises(ValueError):
            normalize_bnname("bn23/../../outside")

    def test_connect_ftp_keeps_certificate_verification_enabled(self) -> None:
        import ssl

        from grb_project.gbm_download import connect_ftp

        context = ssl.create_default_context()
        ftp = mock.MagicMock()
        with mock.patch("grb_project.gbm_download.ssl.create_default_context", return_value=context), mock.patch(
            "grb_project.gbm_download.FTP_TLS", return_value=ftp
        ):
            connect_ftp()

        self.assertTrue(context.check_hostname)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)

    def test_interrupted_download_does_not_treat_partial_file_as_complete(self) -> None:
        from grb_project.gbm_download import download_single_burst

        fake_ftp = InterruptedFTP()
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch(
            "grb_project.gbm_download.connect_ftp", return_value=fake_ftp
        ), mock.patch("grb_project.gbm_download.time.sleep"):
            download_single_burst(fake_ftp, "bn231129799", Path(tmpdir))

            burst_dir = Path(tmpdir) / "bn231129799"
            self.assertEqual((burst_dir / "new.fit").read_bytes(), b"complete")
            self.assertFalse((burst_dir / "new.fit.part").exists())

    def test_download_single_burst_uses_fermi_burst_path_and_skips_existing_files(self) -> None:
        from grb_project.gbm_download import download_single_burst

        fake_ftp = FakeFTP()
        with tempfile.TemporaryDirectory() as tmpdir:
            burst_dir = Path(tmpdir) / "bn231129799"
            burst_dir.mkdir(parents=True)
            (burst_dir / "existing.fit").write_bytes(b"already here")

            returned_ftp = download_single_burst(fake_ftp, "BN231129799", Path(tmpdir))

            self.assertIs(returned_ftp, fake_ftp)
            self.assertEqual(fake_ftp.cwd_calls, ["/fermi/data/gbm/bursts/2023/bn231129799/current/"])
            self.assertEqual(fake_ftp.retrieved, ["RETR new.fit"])
            self.assertEqual((burst_dir / "new.fit").read_bytes(), b"payload")

    def test_cli_parser_accepts_gbm_download_subcommand(self) -> None:
        from grb_project.pipeline import parse_args

        args = parse_args(["download-gbm", "--bnname", "bn231129799", "--save-dir", "/tmp/gbm"])

        self.assertEqual(args.command, "download-gbm")
        self.assertEqual(args.bnname, "bn231129799")
        self.assertEqual(args.save_dir, "/tmp/gbm")

    def test_cli_parser_reads_download_subcommand_from_sys_argv(self) -> None:
        from grb_project.pipeline import parse_args

        with mock.patch("sys.argv", ["grb-project", "download-gbm", "--bnname", "bn231129799"]):
            args = parse_args()

        self.assertEqual(args.command, "download-gbm")
        self.assertEqual(args.bnname, "bn231129799")

    def test_download_cli_dispatches_to_resolved_targets(self) -> None:
        from grb_project import gbm_download

        args = mock.MagicMock(
            bnname="bn231129799",
            list_path=None,
            joint=False,
            save_dir="/tmp/gbm",
            csv_path="/tmp/lat.csv",
            gbm_xls="/tmp/gbm.xls",
            year_from=2022,
            year_to=None,
        )

        with mock.patch.object(gbm_download, "download_target_list") as mock_download:
            gbm_download.cli_main_from_args(args)

        mock_download.assert_called_once_with(["bn231129799"], save_dir=Path("/tmp/gbm"))


if __name__ == "__main__":
    unittest.main()
