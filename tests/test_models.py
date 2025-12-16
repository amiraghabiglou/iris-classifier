from __future__ import annotations

import unittest

from sklearn.base import BaseEstimator

from src.data_loader import DataConfig, IrisDataLoader
from src.trainer import ModelTrainer, TrainConfig


class TestModels(unittest.TestCase):
    def test_build_models_returns_required_two(self) -> None:
        trainer = ModelTrainer(TrainConfig())
        models = trainer.build_models()
        self.assertEqual(set(models.keys()), {"logistic_regression", "decision_tree"})
        for m in models.values():
            self.assertIsInstance(m, BaseEstimator)

    def test_fit_produces_fitted_estimators(self) -> None:
        loader = IrisDataLoader(DataConfig())
        df = loader.load()
        X_train, X_test, y_train, y_test = loader.split(df)

        trainer = ModelTrainer(TrainConfig())
        models = trainer.build_models()
        fitted = trainer.fit(models, X_train, y_train)

        # Ensure they can predict without error
        for name, model in fitted.items():
            preds = model.predict(X_test)
            self.assertEqual(len(preds), len(X_test))
