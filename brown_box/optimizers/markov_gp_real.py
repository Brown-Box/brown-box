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
        xi=5.0,
        r_xi=1,
        kappa=2.6,
        iter_timeout=40.0,
        min_known=2,
        normalize_y=True,
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
        self.xi = xi
        self.r_xi = r_xi
        self.kappa = kappa
        self._iter = 0
        self.min_known = min_known
        self.normalize_y = normalize_y

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
        if len(self.known_points) < self.min_known:
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
                xi=self.xi,
                kappa=self.kappa,
            )
            meta_minimizer = self._meta_optimizer(
                self.tr, self._random_state, cost_f, step=self._iter
            )
            # meta_minimizer.observe(self.known_points, self.known_values)
            meta_minimizer.observe(all_points, all_values)
            min_point = [meta_minimizer.suggest(timeout=self.iter_timeout*0.9/n_suggestions)]
            new_points += min_point

            _p = {k: [dic[k] for dic in min_point] for k in min_point[0]}
            X = self.tr.to_real_space(**_p)
            min_value = gp.predict(X)[0]
            new_values.append(min_value)
            # print(min_value, min_point)
        self.xi *= self.r_xi
        self._iter += 1
        return new_points

    def _gp(self):
        return GaussianProcessRegressor(
            kernel=DiscreteKernel(self.kernel, self.tr),
            # kernel=self.kernel,
            alpha=1e-6,
            normalize_y=self.normalize_y,
            n_restarts_optimizer=5,
            random_state=self._random_state,
        )

if __name__ == "__main__":
    experiment_main(MarkovGaussianProcess)
