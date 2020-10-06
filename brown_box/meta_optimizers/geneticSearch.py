from time import time
from typing import Optional

import numpy as np
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver
from scipy.optimize import Bounds

from ..utils import HyperTransformer, spec_to_bound


class GeneticSearch:
    def __init__(self, transformer: HyperTransformer) -> None:
        self._transformer = transformer
        self._api_config = transformer.api_config
        self._start_time = None
        self._timeout = None
        self._timeout_passed = None

    def suggest(
        self, fit_function: callable, seed: int = 42, timeout: Optional[int] = None
    ) -> dict:
        bounds = []
        for param_name, param_info in self._api_config.items():
            if param_info["type"] in ["real", "int"]:
                if "range" in param_info:
                    real_range = spec_to_bound(param_info)
                    ga_bound_value = (real_range[0][0][()], real_range[1][0][()])
                else:
                    ga_bound_value = (0, 1)
            elif param_info["type"] in ["cat", "bool"]:
                ga_bound_value = (0, 1)
            else:
                param_type = param_info["type"]
                raise ValueError(f"Unknown parameter type {param_type}")

            bounds.append(ga_bound_value)

        if timeout:
            self._start_time = time()
            self._timeout = timeout
            callback = self._timeout_callback
            self._timeout_passed = False
        else:
            callback = None

        func = self._GA_to_real_wrapper(fit_function)

        # TODO turn on polish? Find out our own way to polish result?
        solver = BrownEvolutionSolver(
            transformer=self._transformer,
            func=func,
            bounds=bounds,
            seed=seed,
            callback=callback,
            polish=False,
        )
        solver_return_value = solver.solve()
        if not solver_return_value["success"] and not self._timeout_passed:
            message = solver_return_value["message"]
            raise ValueError(
                f"DifferentialEvolutionSolver failed with message: {message}"
            )

        best_value = solver.x
        real_params = self._GA_to_real(best_value)
        if len(real_params.shape) == 1:
            real_params = np.array([real_params])
        result = self._transformer.to_hyper_space(real_params)
        result = {
            key: val[0][0] if type(val) is np.ndarray else val[0]
            for key, val in result.items()
        }

        return result

    def _timeout_callback(self, xk, convergence):
        if time() - self._start_time >= self._timeout:
            self._timeout_passed = True
            return True
        else:
            return False

    def _GA_to_real_wrapper(self, func):
        def transform(population_member):
            real_params = self._GA_to_real(population_member)
            return func(real_params)

        return transform

    def _GA_to_real(self, population_member):
        real_params = []
        for i, (param_name, param_info) in enumerate(self._api_config.items()):
            if param_info["type"] in ["real", "int"]:
                if "range" in param_info:
                    value = np.array([population_member[i]])
                else:
                    number_of_categories = len(param_info["values"])
                    cat_pos = min(
                        int(population_member[i] * number_of_categories),
                        number_of_categories - 1,
                    )
                    hypervalue = param_info["values"][cat_pos]
                    value = self._transformer._reals[param_name]([hypervalue])
            elif param_info["type"] == "cat":
                number_of_categories = len(param_info["values"])
                cat_pos = min(
                    int(population_member[i] * number_of_categories),
                    number_of_categories - 1,
                )
                hypervalue = param_info["values"][cat_pos]
                value = self._transformer._reals[param_name]([hypervalue])[0]
            elif param_info["type"] == "bool":
                hypervalue = int(population_member[i] >= 0.5)
                value = self._transformer._reals[param_name]([hypervalue])
            else:
                raise ValueError()
            real_params.append(value)

        real_params = np.concatenate(real_params)
        return real_params

class GeneticSearchNonRandom:
    def __init__(self, transformer: HyperTransformer, random, cost_function) -> None:
        self._transformer = transformer
        self._api_config = transformer.api_config
        self._start_time = None
        self._timeout = None
        self._timeout_passed = None
        self.top_points_real = []
        self.top_values = []
        self.random=random
        self.cost=cost_function

    def _timeout_callback(self, xk, convergence):
        if time() - self._start_time >= self._timeout:
            self._timeout_passed = True
            return True
        else:
            return False

    def suggest(
        self, timeout: Optional[int] = None
    ) -> dict:

        if timeout:
            self._start_time = time()
            self._timeout = timeout
            callback = self._timeout_callback
            self._timeout_passed = False
        else:
            callback = None
        top_points = np.vstack([self.top_points_real[:5, ...]]*3)
        dx = self._transformer.random_continuous(15, self.random)*0.01
        # TODO turn on polish? Find out our own way to polish result?
        solver = DifferentialEvolutionSolver(
            func=self.cost,
            bounds=Bounds(self._transformer._lb, self._transformer._ub),
            init=top_points + dx,
            seed=self.random,
            callback=callback,
            polish=False,
            strategy="best1bin",
            maxiter=1000,
            popsize=15,
            tol=0.01,
            mutation=(0.5, 1),
            recombination=0.7,
            maxfun=np.inf,
            disp=False,
            atol=0,
            updating="immediate",
            workers=1,
            constraints=(),
        )
        solver_return_value = solver.solve()
        if not solver_return_value["success"] and not self._timeout_passed:
            message = solver_return_value["message"]
            raise ValueError(
                f"DifferentialEvolutionSolver failed with message: {message}"
            )

        
        real_params = solver.x
        if len(real_params.shape) == 1:
            real_params = np.array([real_params])
        result = self._transformer.to_hyper_space(real_params)
        result = {key: val[0][0] if type(val) is np.ndarray else val[0] for key, val in result.items()}

        return result

    def observe(self, x, Y):
        indices = np.argsort(Y)
        self.top_values = [Y[idx] for idx in indices]
        _x = [x[idx] for idx in indices]
        _p = {k: [dic[k] for dic in _x] for k in _x[0]}
        X = self._transformer.to_real_space(**_p)
        self.top_points_real = X

class BrownEvolutionSolver(DifferentialEvolutionSolver):
    def __init__(
        self,
        transformer,
        func,
        bounds,
        args=(),
        strategy="best1bin",
        maxiter=1000,
        popsize=15,
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=None,
        maxfun=np.inf,
        callback=None,
        disp=False,
        polish=True,
        init="latinhypercube",
        atol=0,
        updating="immediate",
        workers=1,
        constraints=(),
    ):
        super().__init__(
            func,
            bounds,
            args,
            strategy,
            maxiter,
            popsize,
            tol,
            mutation,
            recombination,
            seed,
            maxfun,
            callback,
            disp,
            polish,
            init,
            atol,
            updating,
            workers,
            constraints,
        )
        self.transformer = transformer

    def _best1(self, samples):
        """best1bin, best1exp"""
        r0, r1 = samples[:2]
        new_pop_member = self.population[0] + self.scale * (
            self.population[r0] - self.population[r1]
        )
        # neni nahoda lepsi? + optimalizuj for cyklus
        # r = 2 * self.random_number_generator.random_sample(1)[0] - 1
        # new_pop_member = self.population[0] + self.scale * r

        for param_i, (param_name, param_info) in enumerate(
            self.transformer.api_config.items()
        ):
            if param_info["type"] in ["cat", "bool"] or (
                param_info["type"] in ["real", "int"] and "values" in param_info
            ):
                random_number = self.random_number_generator.random_sample(1)[0]
                a = random_number < self.scale
                if random_number < self.scale:
                    new_pop_member[
                        param_i
                    ] = self.random_number_generator.random_sample(1)[0]
                else:
                    new_pop_member[param_i] = self.population[0][param_i]

        new_pop_member = np.clip(new_pop_member, 0, 1)

        return new_pop_member

