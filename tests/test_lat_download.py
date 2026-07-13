from __future__ import annotations

from pathlib import Path
from http.client import RemoteDisconnected
import tempfile
from unittest import mock
import unittest

import pandas as pd


class LatDownloadTests(unittest.TestCase):
    def test_http_downloader_retries_remote_disconnect(self) -> None:
        import requests
        from grb_project.lat_download import download_lat_data_http

        query_url = "https://fermi.gsfc.nasa.gov/cgi-bin/ssc/LAT/QueryResults.cgi?id=L123ABC"
        query_response = mock.Mock(text=f'<a href="{query_url}">results</a>')
        query_response.raise_for_status.return_value = None
        result_response = mock.Mock(
            text="wget https://fermi.gsfc.nasa.gov/FTP/fermi/data/lat/queries/L123_EV00.fits"
        )
        result_response.raise_for_status.return_value = None
        file_response = mock.Mock(
            iter_content=mock.Mock(return_value=[b"events"]),
            raise_for_status=mock.Mock(),
        )
        session = mock.Mock()
        session.post.side_effect = [
            requests.ConnectionError("Connection aborted", RemoteDisconnected("closed")),
            query_response,
        ]
        session.get.side_effect = [result_response, file_response]

        with tempfile.TemporaryDirectory() as tmp:
            paths = download_lat_data_http(
                ra=1.0,
                dec=2.0,
                radius=12.0,
                tstart=10,
                tstop=20,
                destination=Path(tmp),
                session=session,
                sleep=lambda _: None,
                request_retries=2,
            )

        self.assertEqual(len(paths), 1)
        self.assertEqual(session.post.call_count, 2)

    def test_http_downloader_submits_query_and_downloads_fits_files(self) -> None:
        from grb_project.lat_download import download_lat_data_http

        query_url = "https://fermi.gsfc.nasa.gov/cgi-bin/ssc/LAT/QueryResults.cgi?id=L123ABC"
        responses = [
            mock.Mock(text=f'<a href="{query_url}">results</a>'),
            mock.Mock(text="Estimated completion time: 5 seconds"),
            mock.Mock(
                text=(
                    "wget https://fermi.gsfc.nasa.gov/FTP/fermi/data/lat/queries/L123_EV00.fits\n"
                    "wget https://fermi.gsfc.nasa.gov/FTP/fermi/data/lat/queries/L123_SC00.fits"
                )
            ),
        ]
        for response in responses:
            response.raise_for_status.return_value = None
        file_responses = {
            "https://fermi.gsfc.nasa.gov/FTP/fermi/data/lat/queries/L123_EV00.fits": b"events",
            "https://fermi.gsfc.nasa.gov/FTP/fermi/data/lat/queries/L123_SC00.fits": b"spacecraft",
        }
        session = mock.Mock()
        session.post.return_value = responses[0]
        session.get.side_effect = [
            responses[1],
            responses[2],
            *[
                mock.Mock(
                    headers={"content-length": str(len(content))},
                    iter_content=mock.Mock(return_value=[content]),
                    raise_for_status=mock.Mock(),
                )
                for content in file_responses.values()
            ],
        ]

        with tempfile.TemporaryDirectory() as tmp:
            paths = download_lat_data_http(
                ra=278.53,
                dec=53.72,
                radius=12.0,
                tstart=763568278,
                tstop=763572478,
                destination=Path(tmp),
                session=session,
                sleep=lambda _: None,
                max_polls=2,
            )
            contents = {path.name: path.read_bytes() for path in paths}

        self.assertEqual(contents, {"L123_EV00.fits": b"events", "L123_SC00.fits": b"spacecraft"})
        payload = session.post.call_args.kwargs["data"]
        self.assertEqual(payload["coordfield"], "278.53,53.72")
        self.assertEqual(payload["timefield"], "763568278,763572478")
        self.assertEqual(payload["photonOrExtendedOrNone"], "Extended")

    def test_default_downloader_source_does_not_import_three_ml(self) -> None:
        import inspect
        import grb_project.lat_download as module

        self.assertNotIn("threeML", inspect.getsource(module))

    def test_download_target_uses_gcn_position_and_padded_met_window(self) -> None:
        from grb_project.lat_download import LatDownloadTarget, download_target

        target = LatDownloadTarget(
            bnname="bn250313607",
            grb_name="GRB250313A",
            trigger_met=763569278.0,
            t0=0.0,
            t1=1200.0,
            ra=278.53,
            dec=53.72,
        )
        downloader = mock.MagicMock()

        with tempfile.TemporaryDirectory() as tmp:
            result = download_target(target, data_root=Path(tmp), downloader=downloader)

        downloader.assert_called_once_with(
            ra=278.53,
            dec=53.72,
            radius=12.0,
            tstart=763568278,
            tstop=763572478,
            destination=result.destination,
        )
        self.assertEqual(result.destination.name, "GRB250313A")

    def test_catalog_selection_supports_single_target_and_year_month_filter(self) -> None:
        from grb_project.lat_download import select_targets

        frame = pd.DataFrame(
            [
                {
                    "trigname": "bn250206827",
                    "gcn_name": "GRB 250206A",
                    "trigger_met": 760564296.0,
                    "ra,dec": "225.31,-62.26",
                    "T0": 0.0,
                    "T1": 90.0,
                },
                {
                    "trigname": "bn250313607",
                    "gcn_name": "GRB250313A",
                    "trigger_met": 763569278.0,
                    "ra,dec": "278.53,53.72",
                    "T0": 0.0,
                    "T1": 1200.0,
                },
                {
                    "trigname": "bn240101123",
                    "gcn_name": "GRB240101A",
                    "trigger_met": 700000000.0,
                    "ra,dec": "10.0,20.0",
                    "T0": 0.0,
                    "T1": 100.0,
                },
            ]
        )

        single = select_targets(frame, bnnames=["BN250206827"])
        filtered = select_targets(frame, year=2025, month_from=3)

        self.assertEqual([item.bnname for item in single], ["bn250206827"])
        self.assertEqual([item.bnname for item in filtered], ["bn250313607"])

    def test_cli_parser_accepts_lat_download_subcommand(self) -> None:
        from grb_project.pipeline import parse_args

        args = parse_args(["download-lat", "--bnname", "bn250313607", "--data-root", "/tmp/lat"])

        self.assertEqual(args.command, "download-lat")
        self.assertEqual(args.bnname, ["bn250313607"])
        self.assertEqual(args.data_root, "/tmp/lat")


if __name__ == "__main__":
    unittest.main()
