import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

import bayesmark.random_search as rs
from bayesmark import np_util
from bayesmark.experiment import experiment_main

from ..cost_functions import ucb_real
from ..meta_optimizers.geneticSearch import GeneticSearch
from ..utils import DiscreteKernel
from .markov_gp_real import MarkovGaussianProcessReal


class GAMarkovGaussianProcessReal(MarkovGaussianProcessReal):
    primary_import = "bayesmark"

    def __init__(
        self,
        api_config,
        random=np_util.random,
        meta_optimizer=GeneticSearch,
        cost=ucb_real,
    ):
        super().__init__(api_config, random, meta_optimizer, cost)

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
            x_guess = rs.suggest_dict(
                [],
                [],
                self._api_config,
                n_suggestions=n_suggestions,
                random=self._random_state,
            )
            return x_guess

        new_points = []
        new_values = []
        for k in range(n_suggestions):
            gp = GaussianProcessRegressor(
                kernel=DiscreteKernel(Matern(nu=2.5), self.tr),
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=self._random_state,
            )
            all_points = self.known_points[:]
            all_points += new_points
            all_values = np.concatenate([self.known_values, new_values])
            all_known_points = {
                k: [dic[k] for dic in all_points] for k in all_points[0]
            }
            gp.fit(self.tr.to_real_space(**all_known_points), all_values)

            cost_f = self._cost(
                gp,
                self.tr,
                max_y=max(all_values),
                min_y=min(all_values),
                xi=0.1,
                kappa=2.6,
            )
            meta_minimizer = self._meta_optimizer(self.tr)

            min_point = [meta_minimizer.suggest(cost_f, k, timeout=4.0)]
            new_points += min_point

            _p = {k: [dic[k] for dic in min_point] for k in min_point[0]}
            X = self.tr.to_real_space(**_p)
            min_value = gp.predict(X)[0]
            new_values.append(min_value)
            # print(min_value, min_point)
        return new_points


if __name__ == "__main__":
    experiment_main(GAMarkovGaussianProcessReal)
