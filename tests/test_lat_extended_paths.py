from __future__ import annotations

from pathlib import Path
import inspect
import tempfile
import unittest


class LatExtendedPathTests(unittest.TestCase):
    def test_extended_files_are_copied_only_into_lat_bn_directory(self) -> None:
        from grb_project.io_utils import copy_extended_lat_to_bn_dir

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "extended" / "GRB123"
            destination = root / "results" / "GRB123" / "lat" / "bn123"
            source.mkdir(parents=True)
            (source / "L123_EV00.fits").write_bytes(b"events")
            (source / "L123_SC00.fits").write_bytes(b"spacecraft")

            copied = copy_extended_lat_to_bn_dir(
                "GRB123",
                str(destination),
                str(root / "extended"),
            )

            self.assertTrue(copied)
            self.assertEqual((destination / "gll_ft1_tr_bn123_v00.fit").read_bytes(), b"events")
            self.assertEqual((destination / "gll_ft2_tr_bn123_v00.fit").read_bytes(), b"spacecraft")
            self.assertFalse((destination.parent / "gll_ft1_tr_bn123_v00.fit").exists())
            self.assertFalse((destination.parent / "gll_ft2_tr_bn123_v00.fit").exists())

    def test_pipeline_uses_lat_root_for_builder_and_bn_directory_for_dataset(self) -> None:
        import grb_project.lat_extended_three_ml as module

        source = inspect.getsource(module._run_lat_extended_three_ml_impl)

        self.assertIn("gtburst_data_repository = lat_dir", source)
        self.assertIn("_resolve_ft_paths(lat_dir, bn_name, extended_data_dir)", source)
        self.assertIn("dataset_dir = os.path.join(lat_dir, bn_name)", source)
        self.assertIn("selection[\"trigger_time\"],\n            dataset_dir,", source)
        self.assertIn("gtburst_data_repository=gtburst_data_repository", source)

    def test_gtburst_dataset_directory_contains_expected_four_file_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "lat" / "bn123"
            dataset_dir.mkdir(parents=True)
            expected = {
                "gll_ft1_tr_bn123_v00.fit",
                "gll_ft2_tr_bn123_v00.fit",
                "gll_cspec_tr_bn123_v00.pha",
                "gll_cspec_tr_bn123_v00.rsp",
            }
            for name in expected:
                (dataset_dir / name).touch()

            actual = {path.name for path in dataset_dir.iterdir()}

        self.assertTrue(expected.issubset(actual))

    def test_resolve_ft_paths_stores_data_under_bn_directory(self) -> None:
        from grb_project.lat_extended_three_ml import _resolve_ft_paths

        with tempfile.TemporaryDirectory() as tmp:
            lat_dir = Path(tmp) / "lat"
            bn_dir = lat_dir / "bn123"
            bn_dir.mkdir(parents=True)
            ft1 = bn_dir / "gll_ft1_tr_bn123_v00.fit"
            ft2 = bn_dir / "gll_ft2_tr_bn123_v00.fit"
            ft1.write_bytes(b"ft1")
            ft2.write_bytes(b"ft2")

            resolved = _resolve_ft_paths(str(lat_dir), "bn123", str(Path(tmp) / "extended"))

        self.assertEqual(resolved, (str(ft1), str(ft2)))


if __name__ == "__main__":
    unittest.main()
