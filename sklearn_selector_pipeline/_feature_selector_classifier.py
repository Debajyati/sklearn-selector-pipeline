import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
)
from sklearn.utils.multiclass import unique_labels


class FeatureSelectorClassifier(BaseEstimator, ClassifierMixin):
    """
    A meta-estimator that composes a feature selector and a classifier into a single
    scikit-learn compatible estimator.

    Features:
    - Supports selectors that implement fit_transform(X, y) or fit(X, y) + transform(X).
    - Forwards fit-time parameters to selector and classifier using prefixes:
        selector__param and classifier__param.
      Unprefixed params are attempted to be passed first to the selector then to the classifier.
    - Exposes transform(X).
    - Proxies predict_proba and decision_function when available.
    - Supports incremental training via partial_fit when the underlying classifier
      (and optionally selector) implement partial_fit.

    Parameters
    ----------
    feature_selector : estimator
        Must implement `fit` and `transform` (e.g. SelectKBest, RFE, or a custom selector).
    classifier : estimator
        Must implement `fit` and `predict` (any sklearn-compatible classifier).

    Attributes
    ----------
    classes_ : ndarray
        Class labels.
    feature_selector_ : estimator
        The fitted feature selector.
    classifier_ : estimator
        The fitted classifier.
    """

    def __init__(self, feature_selector, classifier):
        self.feature_selector = feature_selector
        self.classifier = classifier

    def _split_fit_params(self, fit_params):
        selector_fit_params = {}
        classifier_fit_params = {}
        generic_params = {}

        for key, val in fit_params.items():
            if key.startswith("selector__"):
                selector_fit_params[key[len("selector__"):]] = val
            elif key.startswith("classifier__"):
                classifier_fit_params[key[len("classifier__"):]] = val
            else:
                generic_params[key] = val

        return selector_fit_params, classifier_fit_params, generic_params

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
                pass

        if hasattr(selector, "fit"):
            selector.fit(X, y, **selector_fit_params)
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
        Fit the feature selector then the classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        **fit_params : dict
            Additional fit parameters. Use selector__* and classifier__* prefixes 
            to target parameters to specific components.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        # fresh clones for each fit
        self.feature_selector_ = clone(self.feature_selector)
        self.classifier_ = clone(self.classifier)

        selector_specific, classifier_specific, generic = self._split_fit_params(fit_params)

        selector_params = {**generic, **selector_specific}
        X_selected = self._fit_selector(X, y, self.feature_selector_, selector_params)

        classifier_params = {**generic, **classifier_specific}
        self.classifier_.fit(X_selected, y, **classifier_params)

        return self

    def partial_fit(self, X, y, classes=None, **fit_params):
        """
        Incremental fit for classifiers that support partial_fit.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        classes : array-like of shape (n_classes,), default=None
            Classes across all calls to partial_fit.
        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X)
        X, y = check_X_y(X, y)

        # Ensure classifier supports partial_fit
        if not hasattr(self.classifier, "partial_fit") and not (
            hasattr(self, "classifier_") and hasattr(self.classifier_, "partial_fit")
        ):
            raise AttributeError(f"{self.classifier.__class__.__name__} does not support partial_fit.")

        # First call: create clones and fit selector
        if not hasattr(self, "feature_selector_") or not hasattr(self, "classifier_"):
            self.feature_selector_ = clone(self.feature_selector)
            self.classifier_ = clone(self.classifier)

            if classes is not None:
                self.classes_ = np.asarray(classes)
            else:
                self.classes_ = unique_labels(y)

            selector_specific, classifier_specific, generic = self._split_fit_params(fit_params)
            selector_params = {**generic, **selector_specific}
            X_selected = self._fit_selector(X, y, self.feature_selector_, selector_params)

            classifier_params = {**generic, **classifier_specific}
            # Many classifiers require classes on the first partial_fit
            try:
                if classes is not None:
                    self.classifier_.partial_fit(X_selected, y, classes=self.classes_, **classifier_params)
                else:
                    try:
                        self.classifier_.partial_fit(X_selected, y, **classifier_params)
                    except TypeError:
                        self.classifier_.partial_fit(X_selected, y, classes=self.classes_, **classifier_params)
            except TypeError:
                # fallback to always passing classes_
                self.classifier_.partial_fit(X_selected, y, classes=self.classes_, **classifier_params)

            return self

        # Subsequent calls
        selector_specific, classifier_specific, generic = self._split_fit_params(fit_params)
        selector_params = {**generic, **selector_specific}
        classifier_params = {**generic, **classifier_specific}

        if hasattr(self.feature_selector_, "partial_fit"):
            try:
                self.feature_selector_.partial_fit(X, y, **selector_params)
            except TypeError:
                self.feature_selector_.partial_fit(X, **selector_params)
            X_selected = self.feature_selector_.transform(X)
        else:
            X_selected = self.feature_selector_.transform(X)

        if classes is not None:
            self.classifier_.partial_fit(X_selected, y, classes=classes, **classifier_params)
        else:
            self.classifier_.partial_fit(X_selected, y, **classifier_params)

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
        Fit the feature selector and classifier, then return the transformed features.

        This is equivalent to calling fit(X, y, **fit_params) followed by transform(X),
        but may be more efficient for some feature selectors that implement fit_transform.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        **fit_params : dict
            Additional fit parameters. Use selector__* and classifier__* prefixes
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
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self, ["feature_selector_", "classifier_"])
        X = check_array(X)
        X_selected = self.feature_selector_.transform(X)
        return self.classifier_.predict(X_selected)

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_proba : array-like of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        check_is_fitted(self, ["feature_selector_", "classifier_"])
        X = check_array(X)
        X_selected = self.feature_selector_.transform(X)
        if hasattr(self.classifier_, "predict_proba"):
            return self.classifier_.predict_proba(X_selected)
        raise AttributeError(f"{self.classifier_.__class__.__name__} does not support predict_proba.")

    def decision_function(self, X):
        """
        Predict confidence scores for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        scores : array-like of shape (n_samples,) or (n_samples, n_classes)
            Confidence scores.
        """
        check_is_fitted(self, ["feature_selector_", "classifier_"])
        X = check_array(X)
        if not hasattr(self.classifier_, "decision_function"):
            raise AttributeError(f"{self.classifier_.__class__.__name__} does not support decision_function.")
        X_selected = self.feature_selector_.transform(X)
        return self.classifier_.decision_function(X_selected)

    def score(self, X, y,sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        y : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        score : float
            Mean accuracy.
        """
        check_is_fitted(self, ["feature_selector_", "classifier_"])
        X, y = check_X_y(X, y)
        X_selected = self.feature_selector_.transform(X)
        return self.classifier_.score(X_selected, y,sample_weight=sample_weight)
