from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataConfig:
    """Configuration for dataset loading and splitting.

    Attributes:
        test_size: Fraction of samples used for the test split.
        random_state: Random seed used for reproducible splitting.
        stratify: Whether to stratify the split by the target label.
    """
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True


class IrisDataLoader:
    """Loads the Iris dataset and produces train/test splits."""

    def __init__(self, config: DataConfig) -> None:
        self._config = config

    def load(self) -> pd.DataFrame:
        """Load Iris as a pandas DataFrame.

        Returns:
            DataFrame with 4 feature columns and a 'target' column.

        Raises:
            RuntimeError: If the dataset cannot be loaded.
        """
        try:
            iris = load_iris(as_frame=True)
            df: pd.DataFrame = iris.frame.copy()
            logger.info("Loaded Iris dataset with shape=%s", df.shape)
            return df
        except Exception as exc:
            raise RuntimeError("Failed to load Iris dataset") from exc

    def split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split dataframe into train/test features and labels.

        Args:
            df: Iris dataframe containing features and a 'target' column.

        Returns:
            X_train, X_test, y_train, y_test

        Raises:
            ValueError: If required columns are missing.
        """
        if "target" not in df.columns:
            raise ValueError("Input dataframe must contain a 'target' column")

        X = df.drop(columns=["target"])
        y = df["target"]

        strat = y if self._config.stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self._config.test_size,
            random_state=self._config.random_state,
            stratify=strat,
        )

        logger.info(
            "Split data: X_train=%s, X_test=%s", X_train.shape, X_test.shape
        )
        return X_train, X_test, y_train, y_test
