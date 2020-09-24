from time import time

from scipy.optimize._differentialevolution import DifferentialEvolutionSolver


class GeneticSearch:
    def __init__(self, api_config, fit_function):
        self._api_config = api_config
        self._fit_function = fit_function
        self._start_time = None
        self._timeout = None
        self._timeout_passed = None

    def search(self, seed=42, timeout=None):
        bounds = [
            (param_info["range"][0], param_info["range"][1])
            for param_info in self._api_config.values()
        ]

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
        result = {
            param_name: best_value[i]
            for i, param_name in enumerate(self._api_config.keys())
        }

        return result

    def _timeout_callback(self, xk, convergence):
        if time() - self._start_time >= self._timeout:
            self._timeout_passed = True
            return True
        else:
            return False
