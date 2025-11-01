import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
)


class FeatureSelectorRegressor(BaseEstimator, RegressorMixin):
    """
    A meta-estimator that composes a feature selector and a regressor into a single
    scikit-learn compatible estimator.

    Features:
    - Supports selectors that implement fit_transform(X, y) or fit(X, y) + transform(X).
    - Forwards fit-time parameters to selector and regressor using prefixes:
        selector__param and regressor__param.
      Unprefixed params are attempted to be passed first to the selector then to the regressor.
    - Exposes transform(X) and fit_transform(X, y).
    - Supports incremental training via partial_fit when the underlying regressor
      (and optionally selector) implement partial_fit.
    - Supports sample_weight in score method.

    Parameters
    ----------
    feature_selector : estimator
        Must implement `fit` and `transform` (e.g. SelectKBest, RFE, or a custom selector).
    regressor : estimator
        Must implement `fit` and `predict` (any sklearn-compatible regressor).

    Attributes
    ----------
    feature_selector_ : estimator
        The fitted feature selector.
    regressor_ : estimator
        The fitted regressor.
    """

    def __init__(self, feature_selector, regressor):
        self.feature_selector = feature_selector
        self.regressor = regressor

    def _split_fit_params(self, fit_params):
        selector_fit_params = {}
        regressor_fit_params = {}
        generic_params = {}

        for key, val in fit_params.items():
            if key.startswith("selector__"):
                selector_fit_params[key[len("selector__"):]] = val
            elif key.startswith("regressor__"):
                regressor_fit_params[key[len("regressor__"):]] = val
            else:
                generic_params[key] = val

        return selector_fit_params, regressor_fit_params, generic_params

    def _fit_selector(self, X, y, selector, selector_fit_params):
        """
        Fit the selector and return transformed X.
        Prefer fit_transform; fallback to fit+transform.
        """
        if hasattr(selector, "fit_transform"):
            try:
                return selector.fit_transform(X, y, **selector_fit_params)
            except TypeError:
                # fit_transform didn't accept y or provided params -> fallback
                try:
                    return selector.fit_transform(X, **selector_fit_params)
                except TypeError:
                    # fit_transform didn't accept any params -> use default
                    return selector.fit_transform(X)

        if hasattr(selector, "fit"):
            try:
                selector.fit(X, y, **selector_fit_params)
            except TypeError:
                # fit didn't accept y or some params -> try without y
                try:
                    selector.fit(X, **selector_fit_params)
                except TypeError:
                    # fit didn't accept params -> use default
                    selector.fit(X, y)
            
            if not hasattr(selector, "transform"):
                raise AttributeError(
                    f"The provided feature_selector ({selector.__class__.__name__}) "
                    "does not implement transform after fitting."
                )
            return selector.transform(X)

        raise AttributeError(
            "feature_selector must implement either fit_transform(X, y) or fit(X, y) + transform(X)."
        )

    def fit(self, X, y, **fit_params):
        """
        Fit the feature selector then the regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.
        **fit_params : dict
            Additional fit parameters. Use selector__* and regressor__* prefixes 
            to target parameters to specific components.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, multi_output=True)

        # fresh clones for each fit
        self.feature_selector_ = clone(self.feature_selector)
        self.regressor_ = clone(self.regressor)

        selector_specific, regressor_specific, generic = self._split_fit_params(fit_params)

        selector_params = {**generic, **selector_specific}
        X_selected = self._fit_selector(X, y, self.feature_selector_, selector_params)

        regressor_params = {**generic, **regressor_specific}
        self.regressor_.fit(X_selected, y, **regressor_params)

        return self

    def partial_fit(self, X, y, **fit_params):
        """
        Incremental fit for regressors that support partial_fit.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.
        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X)
        X, y = check_X_y(X, y, multi_output=True)

        # Ensure regressor supports partial_fit
        if not hasattr(self.regressor, "partial_fit") and not (
            hasattr(self, "regressor_") and hasattr(self.regressor_, "partial_fit")
        ):
            raise AttributeError(f"{self.regressor.__class__.__name__} does not support partial_fit.")

        # First call: create clones and fit selector
        if not hasattr(self, "feature_selector_") or not hasattr(self, "regressor_"):
            self.feature_selector_ = clone(self.feature_selector)
            self.regressor_ = clone(self.regressor)

            selector_specific, regressor_specific, generic = self._split_fit_params(fit_params)
            selector_params = {**generic, **selector_specific}
            X_selected = self._fit_selector(X, y, self.feature_selector_, selector_params)

            regressor_params = {**generic, **regressor_specific}
            self.regressor_.partial_fit(X_selected, y, **regressor_params)

            return self

        # Subsequent calls
        selector_specific, regressor_specific, generic = self._split_fit_params(fit_params)
        selector_params = {**generic, **selector_specific}
        regressor_params = {**generic, **regressor_specific}

        if hasattr(self.feature_selector_, "partial_fit"):
            try:
                self.feature_selector_.partial_fit(X, y, **selector_params)
            except TypeError:
                self.feature_selector_.partial_fit(X, **selector_params)
            X_selected = self.feature_selector_.transform(X)
        else:
            X_selected = self.feature_selector_.transform(X)

        self.regressor_.partial_fit(X_selected, y, **regressor_params)

        return self

    def transform(self, X):
        """
        Transform X using the fitted feature selector.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        X_transformed : array-like of shape (n_samples, n_selected_features)
            Transformed data with selected features.
        """
        check_is_fitted(self, ["feature_selector_"])
        X = check_array(X)
        if not hasattr(self.feature_selector_, "transform"):
            raise AttributeError(f"{self.feature_selector_.__class__.__name__} does not implement transform.")
        return self.feature_selector_.transform(X)

    def fit_transform(self, X, y, **fit_params):
        """
        Fit the feature selector and regressor, then return the transformed features.

        This is equivalent to calling fit(X, y, **fit_params) followed by transform(X),
        but may be more efficient for some feature selectors that implement fit_transform.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.
        **fit_params : dict
            Additional fit parameters. Use selector__* and regressor__* prefixes
            to target parameters to specific components.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_selected_features)
            The selected features from the training data.
        """
        # Fit the estimator
        self.fit(X, y, **fit_params)
        # Return the transformed features
        return self.transform(X)

    def predict(self, X):
        """
        Predict target values for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_pred : array-like of shape (n_samples,) or (n_samples, n_targets)
            Predicted target values.
        """
        check_is_fitted(self, ["feature_selector_", "regressor_"])
        X = check_array(X)
        X_selected = self.feature_selector_.transform(X)
        return self.regressor_.predict(X_selected)

    def score(self, X, y, sample_weight=None):
        """
        Return the coefficient of determination R² of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            True target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            R² score.
        """
        check_is_fitted(self, ["feature_selector_", "regressor_"])
        X, y = check_X_y(X, y, multi_output=True)
        X_selected = self.feature_selector_.transform(X)
        return self.regressor_.score(X_selected, y, sample_weight=sample_weight)
