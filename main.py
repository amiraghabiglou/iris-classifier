from __future__ import annotations

import logging
from dataclasses import asdict
from src.config import AppConfig
from typing import Dict

import pandas as pd
from sklearn.base import BaseEstimator

from src.data_loader import DataConfig, IrisDataLoader
from src.evaluator import EvalConfig, ModelEvaluator
from src.trainer import ModelTrainer, TrainConfig


def setup_logging() -> None:
    """Configure application-wide logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def run(config: AppConfig) -> Dict[str, Dict[str, float]]:
    """Run the end-to-end Iris classification pipeline.

    Args:
        config: Application configuration.

    Returns:
        Nested dict with model-level metrics (currently just accuracy).
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting run with config=%s", asdict(config))

    loader = IrisDataLoader(config.data)
    df: pd.DataFrame = loader.load()
    X_train, X_test, y_train, y_test = loader.split(df)

    trainer = ModelTrainer(config.train)
    models: Dict[str, BaseEstimator] = trainer.build_models()
    fitted = trainer.fit(models, X_train, y_train)

    evaluator = ModelEvaluator(config.eval)

    metrics: Dict[str, Dict[str, float]] = {}
    # Save ONE required confusion matrix PNG. We'll use the better-performing model's confusion matrix.
    best_model_name = None
    best_acc = -1.0
    best_cm = None

    for name, model in fitted.items():
        logger.info("Evaluating model=%s", name)
        result = evaluator.evaluate(model, X_test, y_test)
        metrics[name] = {"accuracy": float(result["accuracy"])}

        logger.info("\n%s\n%s", name, result["classification_report"])

        if float(result["accuracy"]) > best_acc:
            best_acc = float(result["accuracy"])
            best_model_name = name
            best_cm = result["confusion_matrix"]

    if best_cm is None or best_model_name is None:
        raise RuntimeError("No model evaluation results to plot")

    evaluator.save_confusion_matrix_heatmap(best_cm, title=f"Confusion Matrix ({best_model_name})")
    logger.info("Done. Best model=%s accuracy=%.4f", best_model_name, best_acc)
    return metrics


if __name__ == "__main__":
    setup_logging()
    run(AppConfig())
