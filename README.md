# Iris Classifier (Production-Style Take-Home)

## How to run
1. `python -m venv .venv && source .venv/bin/activate` (Windows: `\.venv\Scripts\activate`)
2. `pip install -r requirements.txt`
3. `python main.py` (prints metrics and saves `confusion_matrix.png` to the repo root)

## Notes
- Code is organized as a small “repo” with typed modules for loading, training, and evaluation, plus unit tests under `tests/`.
- Only the allowed libraries are used for the ML workflow (pandas/numpy/scikit-learn/matplotlib/seaborn).

## Required 4–6 sentence analysis
**Logistic Regression performed better** (96.67% vs 93.33%), likely because the Iris dataset is small and the decision boundary between classes is relatively simple. While Decision Trees can model complex non-linear relationships, they are prone to overfitting on small data, whereas Logistic Regression's linear boundary generalized better to this test set.
The most frequent confusion is between **versicolor** and **virginica** (specifically, 1 versicolor sample was misclassified as virginica).
This happens because these two species overlap significantly in petal dimensions (length vs. width), unlike **setosa**, which is linearly separable from both.
A realistic improvement would be to use **Cross-Validation** rather than a single train/test split to get a more robust estimate of model performance given the small sample size.