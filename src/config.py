from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

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

@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters for the two required models.

    Attributes:
        logreg_max_iter: Maximum iterations for LogisticRegression.
        random_state: Random seed for reproducible DecisionTree behavior.
    """
    logreg_max_iter: int = 1000
    random_state: int = 42

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

@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""
    data: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()
    eval: EvalConfig = EvalConfig(output_dir=Path("."))
    class_names: Tuple[str, str, str] = ("setosa", "versicolor", "virginica")
