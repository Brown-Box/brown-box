import numpy as np
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm

import bayesmark.random_search as rs
from bayesmark import np_util
from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main

from ..utils import HyperTransformer
from ..utils import DiscreteKernel

def minimum(cost, n_suggestions, api_config, random_state, seeds=1000):
    x_guess = rs.suggest_dict([], [], api_config, n_suggestions=seeds, random=random_state)
    dict_lists  = {k: [dic[k] for dic in x_guess] for k in x_guess[0]}
    y = cost(**dict_lists)
    idx = np.argsort(y)
    return [x_guess[i] for i in idx[:n_suggestions]]


def _neg_ei(gp, tr, max_y=0):
    def cost(**kwargs):
        X = tr.to_real_space(**kwargs)
        mean, std = gp.predict(X, return_std=True)
        z = (mean - max_y) / std
        return -(mean * norm.cdf(z) + std * norm.pdf(z))
    return cost

class MultiGaussianProcessExpectedImprovement(AbstractOptimizer):
    primary_import = "bayesmark"

    def __init__(self, api_config, random=np_util.random):
        """This optimizes samples multiple suggestions from Gaussian Process.

        Cost function is set to maximixe expected improvement.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        """
        AbstractOptimizer.__init__(self, api_config)
        self._api_config = api_config
        self._random_state = random
        self.tr = HyperTransformer(api_config)
        self.known_points = []
        self.known_values = []

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
        if len(self.known_points) < 2:
            x_guess = rs.suggest_dict([], [], self._api_config, n_suggestions=n_suggestions, random=self._random_state)
            return x_guess
        
        gp = GaussianProcessRegressor(
            kernel=DiscreteKernel(Matern(nu=2.5), self.tr),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self._random_state,
        )
        known_points  = {k: [dic[k] for dic in self.known_points] for k in self.known_points[0]}
        gp.fit(self.tr.to_real_space(**known_points), self.known_values)
        cost = _neg_ei(gp, self.tr, max(self.known_values))
        return minimum(cost, n_suggestions, self._api_config, self._random_state)

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
        self.known_points += X
        self.known_values = np.concatenate([self.known_values, y])


if __name__ == "__main__":
    experiment_main(BrownBoxOptimizer)
