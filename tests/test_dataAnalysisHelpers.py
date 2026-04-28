import importlib.util
import os
import unittest
from pathlib import Path


class TestDataAnalysisHelpers(unittest.TestCase):
    def test_default_data_path_uses_cwd_when_env_missing(self):
        original = os.environ.pop("MATRIX_HMC_DATA", None)
        try:
            spec = importlib.util.spec_from_file_location(
                "dataAnalysisHelpers_test",
                Path(__file__).resolve().parents[1] / "dataAnalysisHelpers.py",
            )
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)

            self.assertEqual(module.DATA_PATH, Path.cwd())
        finally:
            if original is not None:
                os.environ["MATRIX_HMC_DATA"] = original
