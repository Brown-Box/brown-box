from time import time
from typing import Optional

import numpy as np
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver

from brown_box_package.brown_box.utils.hyper_transformer import HyperTransformer


class GeneticSearch:
    def __init__(self, transformer: HyperTransformer, fit_function: callable) -> None:
        self._transformer = transformer
        self._api_config = transformer.api_config
        self._fit_function = fit_function
        self._start_time = None
        self._timeout = None
        self._timeout_passed = None

    def search(self, seed: int = 42, timeout: Optional[int] = None) -> dict:
        bounds = []
        for param_name, param_info in self._api_config.items():
            if param_info["type"] not in ["real", "int"]:
                param_type = param_info["type"]
                raise ValueError(f"Unsupported type of parameter {param_type}")
            kwargs = {param_name: param_info["range"]}
            real_range = self._transformer.to_real_space(**kwargs)
            bounds.append((real_range[0][0], real_range[1][0]))

        if timeout:
            self._start_time = time()
            self._timeout = timeout
            callback = self._timeout_callback
            self._timeout_passed = False
        else:
            callback = None

        solver = DifferentialEvolutionSolver(
            self._fit_function, bounds=bounds, seed=seed, callback=callback
        )
        solver_return_value = solver.solve()
        if not solver_return_value["success"] and not self._timeout_passed:
            message = solver_return_value["message"]
            raise ValueError(
                f"DifferentialEvolutionSolver failed with message: {message}"
            )

        best_value = solver.x
        real_result = [
            [best_value[i]] for i in range(len(self._api_config.keys()))
        ]
        result = self._transformer.to_hyper_space(np.array(real_result))
        result = {key: val[0][0] for key, val in result.items()}

        return result

    def _timeout_callback(self, xk, convergence):
        if time() - self._start_time >= self._timeout:
            self._timeout_passed = True
            return True
        else:
            return False
