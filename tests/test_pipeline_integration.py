import unittest

from src.data_loader import IrisDataLoader
from src.trainer import ModelTrainer
from src.evaluator import ModelEvaluator
from src.config import AppConfig
from typing import Dict


class TestPipelineIntegration(unittest.TestCase):
    def test_full_training_and_evaluation_pipeline(self):
        # Arrange
        config = AppConfig()
        loader = IrisDataLoader(config.data)
        df = loader.load()
        trainer = ModelTrainer(config.train)
        evaluator = ModelEvaluator(config.eval)

        # Act
        X_train, X_test, y_train, y_test = loader.split(df)
        models = trainer.build_models()
        fitted = trainer.fit(models,X_train, y_train)
        metrics: Dict[str, Dict[str, float]] = {}
        for name, model in fitted.items():
            metrics = evaluator.evaluate(model, X_test, y_test)

        # Assert
        self.assertIn("accuracy", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)
