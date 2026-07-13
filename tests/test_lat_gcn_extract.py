from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock


def _archive_bytes(records: dict[str, dict]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for filename, record in records.items():
            payload = json.dumps(record).encode("utf-8")
            info = tarfile.TarInfo(f"archive.json/{filename}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


class LatGcnExtractTests(unittest.TestCase):
    def test_accepts_fermi_lat_subject_without_hyphen(self) -> None:
        from grb_project.lat_gcn_extract import _parse_record

        record = _parse_record(
            "12345.json",
            {
                "circularId": 12345,
                "subject": "GRB 120711A: Fermi LAT detection",
                "body": "time interval 0 - 10 s; trigger 123456789; RA, Dec = 1.0, 2.0",
            },
        )

        self.assertIsNotNone(record)

    def test_mixed_interval_units_are_converted_independently(self) -> None:
        from grb_project.lat_gcn_extract import _extract_interval

        self.assertEqual(_extract_interval("T0+100 s - T0+2 ks"), ("100", "2000"))

    def test_parses_parenthesized_position_and_decimal_gbm_trigger(self) -> None:
        from grb_project.lat_gcn_extract import _parse_record

        record = _parse_record(
            "30062.json",
            {
                "circularId": 30062,
                "subject": "GRB 210520A: Fermi-LAT detection",
                "body": (
                    "Fermi-GBM (trigger 643230427.939988 / 210520797, GCN 30059).\n"
                    "The best LAT on-ground location is found to be\n"
                    "(RA, Dec) = 123.0, -69.4 (degrees, J2000)\n"
                    "The photon flux above 100 MeV in the time interval 0-100s after the GBM trigger is measured."
                ),
            },
        )

        self.assertIsNotNone(record)
        self.assertEqual(record["Trigger Time"], "643230427.939988")
        self.assertEqual(record["T0 (s)"], "0")
        self.assertEqual(record["T1 (s)"], "100")
        self.assertEqual(record["RA"], "123.0")
        self.assertEqual(record["DEC"], "-69.4")

    def test_cli_parser_accepts_update_lat_gcn_subcommand(self) -> None:
        from grb_project.pipeline import parse_args

        args = parse_args(["update-lat-gcn", "--output-csv", "/tmp/lat.csv"])

        self.assertEqual(args.command, "update-lat-gcn")
        self.assertEqual(args.output_csv, "/tmp/lat.csv")

    def test_cli_dispatches_to_lat_gcn_refresh(self) -> None:
        from grb_project import lat_gcn_extract
        from grb_project.pipeline import cli_main

        expected = mock.sentinel.result
        with mock.patch.object(lat_gcn_extract, "cli_main_from_args", return_value=expected) as dispatch:
            result = cli_main(["update-lat-gcn", "--output-csv", "/tmp/lat.csv"])

        self.assertIs(result, expected)
        dispatch.assert_called_once()

    def test_refresh_downloads_every_run_and_replaces_stale_archive(self) -> None:
        from grb_project.lat_gcn_extract import refresh_gcn_lat_data

        payload = _archive_bytes(
            {
                "40001.json": {
                    "circularId": 40001,
                    "subject": "GRB 250313A: Fermi-LAT Detection",
                    "body": "time interval 0 - 1200 s; trigger 763000001; RA, Dec = 12.34, -45.67",
                }
            }
        )
        calls: list[str] = []

        def downloader(url: str, destination: Path) -> None:
            calls.append(url)
            destination.write_bytes(payload)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_dir = root / "archive.json"
            archive_dir.mkdir()
            (archive_dir / "stale.json").write_text("stale", encoding="utf-8")

            result = refresh_gcn_lat_data(
                archive_url="https://example.test/archive.json.tar.gz",
                archive_tar_path=root / "archive.json.tar.gz",
                archive_dir=archive_dir,
                output_csv=root / "lat.csv",
                downloader=downloader,
            )

            self.assertEqual(calls, ["https://example.test/archive.json.tar.gz"])
            self.assertFalse((archive_dir / "stale.json").exists())
            self.assertTrue((archive_dir / "40001.json").exists())
            self.assertEqual(result.record_count, 1)

    def test_refresh_extracts_expected_lat_columns(self) -> None:
        import pandas as pd

        from grb_project.lat_gcn_extract import refresh_gcn_lat_data

        payload = _archive_bytes(
            {
                "40002.json": {
                    "circularId": 40002,
                    "subject": "GRB 250314B: Fermi-LAT detection",
                    "body": (
                        "The photon flux above 100 MeV in the time interval 1 - 2 ks after the GBM trigger "
                        "is measured. The highest-energy photon is a 1.5 GeV event observed 27 seconds "
                        "after the GBM trigger 763000002. RA, Dec = 123.45, +6.78"
                    ),
                }
            }
        )

        def downloader(_url: str, destination: Path) -> None:
            destination.write_bytes(payload)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = refresh_gcn_lat_data(
                archive_tar_path=root / "archive.json.tar.gz",
                archive_dir=root / "archive.json",
                output_csv=root / "lat.csv",
                downloader=downloader,
            )
            frame = pd.read_csv(result.output_csv)

        self.assertEqual(
            frame.columns.tolist(),
            [
                "Circular ID",
                "GRB Name",
                "Trigger Time",
                "T0 (s)",
                "T1 (s)",
                "Photon Energy",
                "HE Photon Time (s)",
                "RA",
                "DEC",
            ],
        )
        self.assertEqual(frame.loc[0, "GRB Name"], "GRB250314B")
        self.assertEqual(frame.loc[0, "T0 (s)"], 1000)
        self.assertEqual(frame.loc[0, "T1 (s)"], 2000)
        self.assertEqual(frame.loc[0, "Photon Energy"], "1.5 GEV")


if __name__ == "__main__":
    unittest.main()
