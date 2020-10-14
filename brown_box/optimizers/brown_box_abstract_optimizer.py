import numpy as np
import random

import bayesmark.random_search as rs
from bayesmark import np_util
from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.space import JointSpace

from ..utils import HyperTransformer


class BrownBoxAbstractOptimizer(AbstractOptimizer):
    """Base class for Brown Box optimizers"""

    def __init__(self, api_config, random=np_util.random):
        super().__init__(api_config)

        self._random_state = random
        self.tr = HyperTransformer(api_config)

        self.known_points = []
        self.known_values = []  # normalized to (0, 1). inf -> 2
        self.known_values_real = []

        self.current_iteration = 0

    def _remove_known_points(self, guess):
        x_guess = guess[:]
        for point in self.known_points:
            while point in x_guess:
                x_guess.remove(point)
        return x_guess

    def _grid_suggestion(self, n_suggestions):
        space = JointSpace(self.api_config)
        grid = space.grid(n_suggestions)

        # make sure grid has enough items
        for key in grid:
            while 0 < len(grid[key]) < n_suggestions:
                grid[key] += grid[key]
            random.shuffle(grid[key])
            grid[key] = grid[key][:n_suggestions]

        # select from the grid
        suggestions = []
        for i in range(n_suggestions):
            guess = dict()
            for key in grid:
                guess[key] = grid[key][i]
            suggestions.append(guess)

        return suggestions

    def _random_suggestion(self, n_suggestions):
        return rs.suggest_dict(
            [],
            [],
            self.api_config,
            n_suggestions=n_suggestions,
            random=self._random_state,
        )

    def random_suggestion(self, n_suggestions, grid=True):
        want_suggestions = len(self.known_points) + n_suggestions + 2
        if grid:
            x_guess = self._grid_suggestion(want_suggestions)
        else:
            x_guess = self._random_suggestion(want_suggestions)
        reduced_guess = self._remove_known_points(x_guess)
        random.shuffle(reduced_guess)
        return (reduced_guess + x_guess)[:n_suggestions]

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

    @staticmethod
    def _normalize_values(values, inf=np.iinfo(np.int32).max, new_inf=2):
        """Transform values to (0, 1) range. Inf is transformed to new_inf"""
        is_valid_value = lambda v: v != inf
        min_value = min(filter(is_valid_value, values), default=new_inf)
        max_value = max(filter(is_valid_value, values), default=new_inf)

        if min_value == max_value:  # cannot normalize
            return values

        difference = max_value - min_value
        transform = (
            lambda v: (v - min_value) / difference if is_valid_value(v) else new_inf
        )
        return list(map(transform, values))

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
        self.known_values_real = np.concatenate([self.known_values_real, obs_y])
        self.known_values = self._normalize_values(self.known_values_real)

        self.current_iteration += 1
