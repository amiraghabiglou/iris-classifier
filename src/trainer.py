from __future__ import annotations

import logging
from src.config import TrainConfig
from typing import Dict

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains the required classification models."""

    def __init__(self, config: TrainConfig) -> None:
        self._config = config

    def build_models(self) -> Dict[str, BaseEstimator]:
        """Construct the two required models.

        Returns:
            Mapping from model name to unfitted estimator.
        """
        models: Dict[str, BaseEstimator] = {
            "logistic_regression": LogisticRegression(max_iter=self._config.logreg_max_iter),
            "decision_tree": DecisionTreeClassifier(random_state=self._config.random_state),
        }
        return models

    def fit(
        self, models: Dict[str, BaseEstimator], X_train: pd.DataFrame, y_train: pd.Series
    ) -> Dict[str, BaseEstimator]:
        """Fit all provided models.

        Args:
            models: Mapping of model name to unfitted estimator.
            X_train: Training features.
            y_train: Training labels.

        Returns:
            Mapping of model name to fitted estimator.
        """
        fitted: Dict[str, BaseEstimator] = {}
        for name, model in models.items():
            logger.info("Training model=%s", name)
            model.fit(X_train, y_train)
            fitted[name] = model
        return fitted
