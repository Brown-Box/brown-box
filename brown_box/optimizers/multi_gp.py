import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from bayesmark import np_util
from bayesmark.experiment import experiment_main

from .brown_box_abstract_optimizer import BrownBoxAbstractOptimizer
from ..cost_functions import ei
from ..meta_optimizers import RandomOptimizer
from ..utils import DiscreteKernel


class MultiGaussianProcess(BrownBoxAbstractOptimizer):
    primary_import = "bayesmark"

    def __init__(self, api_config, random=np_util.random, meta_optimizer=RandomOptimizer, cost=ei):
        """This optimizes samples multiple suggestions from Gaussian Process.

        Cost function is set to maximixe expected improvement.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        """
        super().__init__(api_config, random)
        self._cost = cost
        self._meta_optimizer = meta_optimizer

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
            return self.random_suggestion(n_suggestions)

        gp = GaussianProcessRegressor(
            kernel=DiscreteKernel(Matern(nu=2.5), self.tr),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self._random_state,
        )
        known_points = {k: [dic[k] for dic in self.known_points] for k in self.known_points[0]}
        gp.fit(self.tr.to_real_space(**known_points), self.known_values)

        cost_f = self._cost(gp, self.tr, max_y=max(self.known_values), x=0.01, kappa=2.6)
        meta_minimizer = self._meta_optimizer(self.api_config, self._random_state, cost_f)
        return meta_minimizer.suggest(n_suggestions, timeout=30)


if __name__ == "__main__":
    experiment_main(MultiGaussianProcess)
