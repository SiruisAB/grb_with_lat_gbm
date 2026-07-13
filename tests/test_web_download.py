from __future__ import annotations

from pathlib import Path
from unittest import mock
import unittest
import types


class WebDownloadTests(unittest.TestCase):
    def test_web_lat_download_dispatches_selected_targets(self) -> None:
        from grb_project import web_app

        expected = [mock.sentinel.result]
        with mock.patch.object(web_app.lat_download, "run_download", return_value=expected) as run_download:
            result = web_app._run_lat_download_request(
                bnnames=["bn250313607"],
                year=None,
                month_from=1,
                catalog_xls="/tmp/fermilat.xls",
                data_root="/tmp/lat",
            )

        self.assertEqual(result, expected)
        run_download.assert_called_once_with(
            bnnames=["bn250313607"],
            year=None,
            month_from=1,
            catalog_xls=Path("/tmp/fermilat.xls"),
            data_root=Path("/tmp/lat"),
        )

    def test_web_lat_gcn_refresh_uses_configured_paths(self) -> None:
        from grb_project import web_app

        with mock.patch.object(web_app.lat_gcn_extract, "refresh_gcn_lat_data") as refresh:
            web_app._run_lat_gcn_refresh_request(
                archive_url="https://example.test/archive.json.tar.gz",
                archive_tar="/tmp/archive.json.tar.gz",
                archive_dir="/tmp/archive.json",
                output_csv="/tmp/lat.csv",
            )

        refresh.assert_called_once_with(
            archive_url="https://example.test/archive.json.tar.gz",
            archive_tar_path=Path("/tmp/archive.json.tar.gz"),
            archive_dir=Path("/tmp/archive.json"),
            output_csv=Path("/tmp/lat.csv"),
        )

    def test_web_download_request_dispatches_single_bnname(self) -> None:
        from grb_project import web_app

        with mock.patch.object(web_app.gbm_download, "download_target_list") as mock_download:
            targets = web_app._run_gbm_download_request(
                mode="单个 GRB",
                bnname="BN231129799",
                list_text="",
                save_dir="/tmp/gbm",
                csv_path="/tmp/lat.csv",
                gbm_xls="/tmp/gbm.xls",
                year_from=2022,
                year_to=None,
            )

        self.assertEqual(targets, ["bn231129799"])
        mock_download.assert_called_once_with(["bn231129799"], save_dir=Path("/tmp/gbm"))

    def test_web_download_request_parses_inline_list(self) -> None:
        from grb_project import web_app

        with mock.patch.object(web_app.gbm_download, "download_target_list") as mock_download:
            targets = web_app._run_gbm_download_request(
                mode="GRB 列表",
                bnname="",
                list_text="bn231129799\n# comment\nBN240101123\nbn231129799",
                save_dir="/tmp/gbm",
                csv_path="/tmp/lat.csv",
                gbm_xls="/tmp/gbm.xls",
                year_from=2022,
                year_to=None,
            )

        self.assertEqual(targets, ["bn231129799", "bn240101123"])
        mock_download.assert_called_once_with(["bn231129799", "bn240101123"], save_dir=Path("/tmp/gbm"))

    def test_download_page_is_available_when_catalog_loading_fails(self) -> None:
        from grb_project import web_app

        fake_sidebar = mock.MagicMock()
        fake_sidebar.radio.return_value = "GBM 数据下载"
        fake_st = types.SimpleNamespace(
            set_page_config=mock.MagicMock(),
            title=mock.MagicMock(),
            caption=mock.MagicMock(),
            sidebar=fake_sidebar,
        )

        with mock.patch.dict("sys.modules", {"streamlit": fake_st}), mock.patch.object(
            web_app,
            "_load_catalog",
            side_effect=RuntimeError("catalog missing"),
        ), mock.patch.object(web_app, "_run_gbm_download_page") as mock_download_page:
            web_app.main()

        mock_download_page.assert_called_once()


if __name__ == "__main__":
    unittest.main()
