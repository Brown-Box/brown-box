from time import time
from typing import Optional

import numpy as np
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver, _ConstraintWrapper, warnings, \
    MapWrapper, _FunctionWrapper, Bounds, check_random_state, string_types, new_bounds_to_old

from brown_box_package.brown_box.utils.hyper_transformer import HyperTransformer


class GeneticSearch:
    def __init__(self, transformer: HyperTransformer, fit_function: callable) -> None:
        self._transformer = transformer
        self._api_config = transformer.api_config
        self._fit_function = fit_function
        self._start_time = None
        self._timeout = None
        self._timeout_passed = None

    # TODO move fit function here
    def search(self, seed: int = 42, timeout: Optional[int] = None) -> dict:
        bounds = []
        for param_name, param_info in self._api_config.items():
            if param_info["type"] in ["real", "int"]:
                kwargs = {param_name: param_info["range"]}
                real_range = self._transformer.to_real_space(**kwargs)
                bounds.append([val[0] for val in real_range])
            elif param_info["type"] == "cat":
                bounds.append((0, 1))
            elif param_info["type"] == "bool":
                kwargs = {param_name: param_info["range"]}
                raise NotImplemented()
            else:
                param_type = param_info["type"]
                raise ValueError(f"Unknown parameter type {param_type}")

        if timeout:
            self._start_time = time()
            self._timeout = timeout
            callback = self._timeout_callback
            self._timeout_passed = False
        else:
            callback = None

        func = self._GA_to_real_wrapper(self._fit_function)

        # TODO turn on polish? Find out our own way to polish result?
        solver = BrownEvolutionSolver(
            transformer=self._transformer, func=func, bounds=bounds, seed=seed, callback=callback, polish=False
        )
        solver_return_value = solver.solve()
        if not solver_return_value["success"] and not self._timeout_passed:
            message = solver_return_value["message"]
            raise ValueError(
                f"DifferentialEvolutionSolver failed with message: {message}"
            )

        best_value = solver.x
        real_params = self._GA_to_real(best_value)
        # real_params = [np.array(val) if type(val) is not list else val for val in real_params]  # TODO je to potreba?
        real_params = np.stack(real_params)
        if len(real_params.shape) == 1:
            real_params = np.array([real_params])
        result = self._transformer.to_hyper_space(real_params)
        result = {key: val[0] for key, val in result.items()}

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
            if param_info["type"] == "real":
                value = population_member[i]
            # TODO ask Kuba how to process int. Is rounding right?
            elif param_info["type"] == "int":
                value = int(round(population_member[i]))
            elif param_info["type"] == "cat":
                number_of_categories = len(param_info["values"])
                cat_pos = min(int(population_member[i] * number_of_categories), number_of_categories)
                kwargs = {param_name: [param_info["values"][cat_pos]]}
                value = self._transformer.to_real_space(**kwargs)[0]
            elif param_info["type"] == "bool":
                raise NotImplemented()
            else:
                raise ValueError()
            real_params.append(value)
        return real_params


class BrownEvolutionSolver(DifferentialEvolutionSolver):
    # def __init__(self, transformer, *args, **kwargs):
    #     self.transformer = transformer
    #   super().__init__(*args, **kwargs)

    def __init__(self, transformer, func, bounds, args=(),
                 strategy='best1bin', maxiter=1000, popsize=15,
                 tol=0.01, mutation=(0.5, 1), recombination=0.7, seed=None,
                 maxfun=np.inf, callback=None, disp=False, polish=True,
                 init='latinhypercube', atol=0, updating='immediate',
                 workers=1, constraints=()):
        super().__init__(func, bounds, args,
                 strategy, maxiter, popsize,
                 tol, mutation, recombination, seed,
                 maxfun, callback, disp, polish,
                 init, atol, updating,
                 workers, constraints)
        self.transformer = transformer

    def _best1(self, samples):
        """best1bin, best1exp"""
        r0, r1 = samples[:2]
        new_pop_member = (self.population[0] + self.scale *
                         (self.population[r0] - self.population[r1]))
        for i, (param_name, param_info) in enumerate(self.transformer.api_config.items()):
            if param_info["type"] in ["cat"]:
                number_of_categories = len(param_info["values"])
                pop0_cat, r0_cat, r1_cat = [min(int(self.population[i] * number_of_categories), number_of_categories) for i in [0, r0, r1]]
                if r0_cat != r1_cat:
                    random_number = self.random_number_generator.random_sample(1)[0]
                    if random_number < self.scale:
                        if pop0_cat != r0_cat:
                            new_pop_member[i] = self.population[r0]
                        else:
                            new_pop_member[i] = self.population[r1]
                    else:
                        new_pop_member[i] = self.population[0]
            elif param_info["type"] == "bool":
                raise NotImplementedError()

        return new_pop_member