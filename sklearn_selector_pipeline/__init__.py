"""
sklearn-selector-pipeline: Meta-estimators for combining feature selectors with classifiers and regressors.
"""

from ._feature_selector_classifier import FeatureSelectorClassifier
from ._feature_selector_regressor import FeatureSelectorRegressor
from ._version import __version__

__all__ = ["FeatureSelectorClassifier", "FeatureSelectorRegressor", "__version__"]
