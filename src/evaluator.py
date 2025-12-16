from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation configuration.

    Attributes:
        output_dir: Directory where plots will be saved.
        confusion_matrix_filename: Filename for the confusion matrix PNG.
        class_names: Ordered class names corresponding to label ids.
    """
    output_dir: Path = Path(".")
    confusion_matrix_filename: str = "confusion_matrix.png"
    class_names: Tuple[str, str, str] = ("setosa", "versicolor", "virginica")


class ModelEvaluator:
    """Computes metrics and generates plots for classifiers."""

    def __init__(self, config: EvalConfig) -> None:
        self._config = config

    def evaluate(
        self, model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, Any]:
        """Evaluate a model on the test set.

        Args:
            model: Fitted estimator.
            X_test: Test features.
            y_test: Test labels.

        Returns:
            Dict containing accuracy, classification report text, confusion matrix array,
            and predicted labels.
        """
        y_pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        report = classification_report(
            y_test, y_pred, target_names=list(self._config.class_names), digits=4
        )
        cm = confusion_matrix(y_test, y_pred)

        logger.info("Accuracy=%.4f", acc)
        return {
            "accuracy": acc,
            "classification_report": report,
            "confusion_matrix": cm,
            "y_pred": y_pred,
        }

    def save_confusion_matrix_heatmap(
        self, cm: Any, title: str = "Confusion Matrix"
    ) -> Path:
        """Save a confusion matrix heatmap as a PNG.

        Args:
            cm: Confusion matrix array-like (shape [n_classes, n_classes]).
            title: Plot title.

        Returns:
            Path to the saved image.
        """
        output_path = self._config.output_dir / self._config.confusion_matrix_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=self._config.class_names,
            yticklabels=self._config.class_names,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        logger.info("Saved confusion matrix heatmap to %s", output_path)
        return output_path
