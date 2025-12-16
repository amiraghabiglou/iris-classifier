from __future__ import annotations

import unittest

import pandas as pd

from src.data_loader import DataConfig, IrisDataLoader


class TestIrisDataLoader(unittest.TestCase):
    def test_load_has_expected_columns(self) -> None:
        loader = IrisDataLoader(DataConfig())
        df = loader.load()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("target", df.columns)
        self.assertEqual(df.shape[0], 150)

    def test_split_shapes_and_stratification(self) -> None:
        cfg = DataConfig(test_size=0.2, random_state=0, stratify=True)
        loader = IrisDataLoader(cfg)
        df = loader.load()
        X_train, X_test, y_train, y_test = loader.split(df)

        self.assertEqual(len(X_train) + len(X_test), len(df))
        # With stratify=True, each class should appear in both splits for Iris sizes.
        self.assertEqual(set(y_train.unique()), {0, 1, 2})
        self.assertEqual(set(y_test.unique()), {0, 1, 2})
