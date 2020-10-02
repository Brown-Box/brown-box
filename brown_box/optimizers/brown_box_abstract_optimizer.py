import numpy as np

from bayesmark.abstract_optimizer import AbstractOptimizer


class BrownBoxAbstractOptimizer(AbstractOptimizer):
    """Base class for Brown Box optimizers"""

    def __init__(self, api_config):
        super().__init__(api_config)

        self.known_points = []
        self.known_values = []

    def observe(self, X, y):
        """Feed the observations back to hyperopt.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated.
        """
        obs_y = []
        for _X, _y in zip(X, y):
            self.known_points.append(_X)
            if np.isfinite(_y):
                obs_y.append(_y)
            else:
                obs_y.append(np.iinfo(np.int32).max)
        self.known_values = np.concatenate([self.known_values, obs_y])
