from time import time
from typing import Optional

import numpy as np
from scipy.optimize._differentialevolution import (
    DifferentialEvolutionSolver,
    _ConstraintWrapper,
    warnings,
    MapWrapper,
    _FunctionWrapper,
    Bounds,
    check_random_state,
    string_types,
    new_bounds_to_old,
)

from brown_box_package.brown_box.utils.hyper_transformer import HyperTransformer


class GeneticSearch:
    def __init__(self, transformer: HyperTransformer) -> None:
        self._transformer = transformer
        self._api_config = transformer.api_config
        self._start_time = None
        self._timeout = None
        self._timeout_passed = None

    # TODO move fit function here
    def search(
        self, fit_function: callable, seed: int = 42, timeout: Optional[int] = None
    ) -> dict:
        bounds = []
        for param_name, param_info in self._api_config.items():
            if param_info["type"] in ["real", "int"]:
                kwargs = {param_name: param_info["range"]}
                # real_range = self._transformer.to_real_space(**kwargs)
                # tr._reals[key]([value])
                real_range = self._transformer._reals[param_name](param_info["range"])
                bounds.append([val for val in real_range])
            elif param_info["type"] == "cat":
                bounds.append((0, 1))
            elif param_info["type"] == "bool":
                bounds.append((0, 1))
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
                value = np.array([population_member[i]])
            # TODO ask Kuba how to process int. Is rounding right?
            elif param_info["type"] == "int":
                value = np.array([int(round(population_member[i]))])
            elif param_info["type"] == "cat":
                number_of_categories = len(param_info["values"])
                cat_pos = min(
                    int(population_member[i] * number_of_categories),
                    number_of_categories,
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


class BrownEvolutionSolver(DifferentialEvolutionSolver):
    # def __init__(self, transformer, *args, **kwargs):
    #     self.transformer = transformer
    #   super().__init__(*args, **kwargs)

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
        for param_i, (param_name, param_info) in enumerate(
            self.transformer.api_config.items()
        ):
            if param_info["type"] in ["cat"]:
                number_of_categories = len(param_info["values"])
                pop0_cat, r0_cat, r1_cat = [
                    min(
                        int(self.population[i][param_i] * number_of_categories),
                        number_of_categories,
                    )
                    for i in [0, r0, r1]
                ]
                new_pop_member = self._maybe_switch_different(
                    new_pop_member, param_i, r0, r1, pop0_cat, r0_cat, r1_cat
                )
            elif param_info["type"] == "bool":
                pop0_bool, r0_bool, r1_bool = [
                    int(self.population[i][param_i] >= 0.5) for i in [0, r0, r1]
                ]
                new_pop_member = self._maybe_switch_different(
                    new_pop_member, param_i, r0, r1, pop0_bool, r0_bool, r1_bool
                )

        return new_pop_member

    def _maybe_switch_different(
        self, pop_member, param_i, r0, r1, pop0_val, r0_val, r1_val
    ):
        if r0_val != r1_val:
            random_number = self.random_number_generator.random_sample(1)[0]
            if random_number < self.scale:
                if pop0_val != r0_val:
                    pop_member[param_i] = self.population[r0][param_i]
                else:
                    pop_member[param_i] = self.population[r1][param_i]
            else:
                pop_member[param_i] = self.population[0][param_i]
        return pop_member
