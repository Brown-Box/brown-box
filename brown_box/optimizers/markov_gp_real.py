import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from bayesmark import np_util
from bayesmark.experiment import experiment_main

from .brown_box_abstract_optimizer import BrownBoxAbstractOptimizer
from ..cost_functions import ei_real
from ..meta_optimizers import SciPyOptimizer
from ..utils import DiscreteKernel


class MarkovGaussianProcessReal(BrownBoxAbstractOptimizer):
    primary_import = "bayesmark"

    def __init__(
        self,
        api_config,
        random=np_util.random,
        meta_optimizer=SciPyOptimizer,
        kernel=Matern(nu=2.5),
        cost=ei_real,
        iter_timeout=40.0
    ):
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
        self.kernel=kernel
        self.iter_timeout = iter_timeout

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

        new_points = []
        new_values = []
        for k in range(n_suggestions):
            gp = self._gp()
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
                xi=0.11,
                kappa=2.6,
            )
            meta_minimizer = self._meta_optimizer(
                self.tr, self._random_state, cost_f
            )
            meta_minimizer.observe(all_points, all_values)
            min_point = [meta_minimizer.suggest(timeout=self.iter_timeout*0.9/n_suggestions)]
            new_points += min_point

            _p = {k: [dic[k] for dic in min_point] for k in min_point[0]}
            X = self.tr.to_real_space(**_p)
            min_value = gp.predict(X)[0]
            new_values.append(min_value)
            # print(min_value, min_point)
        return new_points

    def _gp(self):
        return GaussianProcessRegressor(
            kernel=DiscreteKernel(self.kernel, self.tr),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self._random_state,
        )

if __name__ == "__main__":
    experiment_main(MarkovGaussianProcess)
