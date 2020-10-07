import numpy as np

import bayesmark.random_search as rs
from bayesmark import np_util
from bayesmark.abstract_optimizer import AbstractOptimizer

from ..utils import HyperTransformer


class BrownBoxAbstractOptimizer(AbstractOptimizer):
    """Base class for Brown Box optimizers"""

    def __init__(self, api_config, random=np_util.random):
        super().__init__(api_config)

        self._random_state = random
        self.tr = HyperTransformer(api_config)

        self.known_points = []
        self.known_values = []

        self.current_iteration = 0

    def random_suggestion(self, n_suggestions):
        x_guess = rs.suggest_dict(
            [],
            [],
            self.api_config,
            n_suggestions=n_suggestions,
            random=self._random_state,
        )
        return x_guess

    def suggest(self, n_suggestions=1):
        """Make `n_suggestions` suggestions for what to evaluate next.

        This requires the user observe all previous suggestions before calling
        again.

        Parameters
        ----------
        n_suggestions : int
            The number of suggestions to return.

        Returns
        -------
        next_guess : list of dict
            List of `n_suggestions` suggestions to evaluate the objective
            function. Each suggestion is a dictionary where each key
            corresponds to a parameter being optimized.
        """
        return self.random_suggestion(n_suggestions)

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

        self.current_iteration += 1
