"""
Example: Using a simple Genetic Algorithm based feature selector with
FeatureSelectorClassifier and a RandomForestClassifier.

"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils import check_random_state

from sklearn_selector_pipeline import FeatureSelectorClassifier


class GeneticFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Minimal genetic algorithm feature selector.

    Parameters
    ----------
    n_features_to_select : int
    population_size : int
    generations : int
    crossover_rate : float
    mutation_rate : float
    fitness_estimator : estimator or None
    cv : int
    random_state : int, RandomState instance or None
        Seed or RandomState; accepted values are forwarded to sklearn.utils.check_random_state.
    """

    def __init__(
        self,
        n_features_to_select,
        population_size=20,
        generations=10,
        crossover_rate=0.8,
        mutation_rate=0.02,
        fitness_estimator=None,
        cv=3,
        random_state=None,
    ):
        self.n_features_to_select = int(n_features_to_select)
        self.population_size = int(population_size)
        self.generations = int(generations)
        self.crossover_rate = float(crossover_rate)
        self.mutation_rate = float(mutation_rate)
        self.fitness_estimator = fitness_estimator
        self.cv = int(cv)
        self.random_state = random_state

    def _init_population(self, n_features, rng):
        pop = []
        for _ in range(self.population_size):
            mask = np.zeros(n_features, dtype=bool)
            idx = rng.choice(n_features, self.n_features_to_select, replace=False)
            mask[idx] = True
            pop.append(mask)
        return np.array(pop, dtype=bool)

    def _score_mask(self, X, y, mask, estimator):
        if mask.sum() == 0:
            return 0.0
        Xsub = X[:, mask]
        try:
            scores = cross_val_score(estimator, Xsub, y, cv=self.cv, scoring="accuracy")
            return float(scores.mean())
        except Exception:
            return 0.0

    def _tournament_selection(self, population, fitnesses, rng, k=3):
        idx = rng.choice(len(population), size=k, replace=False)
        best = idx[np.argmax(fitnesses[idx])]
        return population[best].copy()

    def _crossover(self, parent_a, parent_b, rng):
        if rng.rand() > self.crossover_rate:
            return parent_a.copy(), parent_b.copy()
        mask = rng.rand(parent_a.shape[0]) < 0.5
        child1 = parent_a.copy()
        child2 = parent_b.copy()
        child1[mask] = parent_b[mask]
        child2[mask] = parent_a[mask]
        for child in (child1, child2):
            diff = child.sum() - self.n_features_to_select
            if diff > 0:
                ones_idx = np.where(child)[0]
                off_idx = rng.choice(ones_idx, size=diff, replace=False)
                child[off_idx] = False
            elif diff < 0:
                zeros_idx = np.where(~child)[0]
                on_idx = rng.choice(zeros_idx, size=-diff, replace=False)
                child[on_idx] = True
        return child1, child2

    def _mutate(self, individual, rng):
        flips = rng.rand(individual.shape[0]) < self.mutation_rate
        if not flips.any():
            return individual
        ind = individual.copy()
        ind[flips] = ~ind[flips]
        diff = ind.sum() - self.n_features_to_select
        if diff > 0:
            ones_idx = np.where(ind)[0]
            off_idx = rng.choice(ones_idx, size=diff, replace=False)
            ind[off_idx] = False
        elif diff < 0:
            zeros_idx = np.where(~ind)[0]
            on_idx = rng.choice(zeros_idx, size=-diff, replace=False)
            ind[on_idx] = True
        return ind

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        n_features = X.shape[1]

        # create RNG from the stored parameter in a safe way
        rng = check_random_state(self.random_state)

        if self.fitness_estimator is None:
            base_est = RandomForestClassifier(n_estimators=50, random_state=0)
        else:
            base_est = clone(self.fitness_estimator)

        population = self._init_population(n_features, rng)
        fitnesses = np.array([self._score_mask(X, y, ind, base_est) for ind in population], dtype=float)

        for gen in range(self.generations):
            new_pop = []
            best_idx = np.argmax(fitnesses)
            best_ind = population[best_idx].copy()
            new_pop.append(best_ind)

            while len(new_pop) < self.population_size:
                parent_a = self._tournament_selection(population, fitnesses, rng)
                parent_b = self._tournament_selection(population, fitnesses, rng)
                child1, child2 = self._crossover(parent_a, parent_b, rng)
                child1 = self._mutate(child1, rng)
                if len(new_pop) < self.population_size:
                    new_pop.append(child1)
                if len(new_pop) < self.population_size:
                    child2 = self._mutate(child2, rng)
                    new_pop.append(child2)

            population = np.array(new_pop, dtype=bool)
            fitnesses = np.array([self._score_mask(X, y, ind, base_est) for ind in population], dtype=float)
            best_score = fitnesses.max()
            print(f"GA generation {gen+1}/{self.generations}, best CV accuracy: {best_score:.4f}")

        best_idx = np.argmax(fitnesses)
        self.support_ = population[best_idx].copy()
        self.n_features_in_ = n_features
        return self

    def transform(self, X):
        check_array(X)
        if not hasattr(self, "support_"):
            raise ValueError("GeneticFeatureSelector is not fitted yet.")
        return X[:, self.support_]

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X)


if __name__ == "__main__":
    # Create synthetic dataset
    X, y = make_classification(
        n_samples=500,
        n_features=30,
        n_informative=6,
        n_redundant=2,
        random_state=42,
    )

    # GA selector: choose 8 features
    ga_selector = GeneticFeatureSelector(
        n_features_to_select=8,
        population_size=20,
        generations=6,
        crossover_rate=0.8,
        mutation_rate=0.05,
        fitness_estimator=RandomForestClassifier(n_estimators=30, random_state=0),
        cv=3,
        random_state=42,  # can be int, None, or a RandomState instance; handled safely
    )

    rf = RandomForestClassifier(n_estimators=100, random_state=0)

    wrapped = FeatureSelectorClassifier(feature_selector=ga_selector, classifier=rf)

    print("Fitting FeatureSelectorClassifier with GA selector + RandomForest...")
    wrapped.fit(X, y)

    # The selector instance was cloned inside the wrapper; inspect the fitted selector on the wrapper:
    fitted_selector = wrapped.feature_selector_
    selected_mask = getattr(fitted_selector, "support_", None)
    if selected_mask is None:
        print("Selector did not store support_.")
    else:
        selected_indices = np.where(selected_mask)[0]
        print("Selected feature indices:", selected_indices)

    acc = wrapped.score(X, y)
    print(f"Accuracy on training data (wrapped.score): {acc:.4f}")

    Xt = wrapped.transform(X)
    print("Transformed X shape:", Xt.shape)

