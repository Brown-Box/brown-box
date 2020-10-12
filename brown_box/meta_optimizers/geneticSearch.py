from time import time
from typing import Optional

import numpy as np
from scipy.optimize._differentialevolution import (
    DifferentialEvolutionSolver,
    OptimizeResult,
    minimize,
    warnings,
)
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
    def __init__(
        self, transformer: HyperTransformer, random, cost_function, step=0
    ) -> None:
        self._transformer = transformer
        self._api_config = transformer.api_config
        self._start_time = None
        self._timeout = None
        self._timeout_passed = None
        self.top_points_real = []
        self.top_values = []
        self.random = random
        self.cost = cost_function
        self._iter = step

    def _timeout_callback(self, xk, convergence):
        if time() - self._start_time >= self._timeout:
            self._timeout_passed = True
            return True
        else:
            return False

    def suggest(self, timeout: Optional[int] = None) -> dict:

        if timeout:
            self._start_time = time()
            self._timeout = timeout
            callback = self._timeout_callback
            self._timeout_passed = False
        else:
            callback = None
        top_n = 5 + self._iter // 2
        n_rep = 3 + self._iter // 2
        top_points = np.vstack([self.top_points_real[:top_n, ...]] * n_rep)

        dx = self._transformer.random_continuous(top_n * n_rep, self.random)
        dx *= 0.05
        # TODO turn on polish? Find out our own way to polish result?
        solver = BrownEvolutionSolver(
            timeout=self._timeout,
            func=self.cost,
            bounds=Bounds(self._transformer._lb, self._transformer._ub),
            init=top_points + dx,
            seed=self.random,
            callback=callback,
            polish=False,
            strategy="best1bin",
            maxiter=1000,
            tol=0.01,
            mutation=(0.25, 0.75),
            recombination=0.7,
            maxfun=np.inf,
            disp=False,
            atol=0,
            updating="immediate",
            workers=1,
            constraints=(),
        )
        solver_return_value = solver.solve()
        message = solver_return_value["message"]
        if not solver_return_value["success"] and message != _status_message["timeout"]:
            raise ValueError(
                f"DifferentialEvolutionSolver failed with message: {message}"
            )

        real_params = solver.x
        if len(real_params.shape) == 1:
            real_params = np.array([real_params])
        result = self._transformer.to_hyper_space(real_params)
        result = {
            key: val[0][0] if type(val) is np.ndarray else val[0]
            for key, val in result.items()
        }

        return result

    def observe(self, x, Y):
        indices = np.argsort(Y)
        self.top_values = [Y[idx] for idx in indices]
        _x = [x[idx] for idx in indices]
        _p = {k: [dic[k] for dic in _x] for k in _x[0]}
        X = self._transformer.to_real_space(**_p)
        self.top_points_real = X


_status_message = {'success': 'Optimization terminated successfully.',
                   'maxfev': 'Maximum number of function evaluations has '
                              'been exceeded.',
                   'maxiter': 'Maximum number of iterations has been '
                              'exceeded.',
                   'timeout': 'Timeout reached.',
                   'pr_loss': 'Desired error not necessarily achieved due '
                              'to precision loss.',
                   'nan': 'NaN result encountered.'}


class BrownEvolutionSolver(DifferentialEvolutionSolver):
    def __init__(
        self,
        func,
        bounds,
        timeout=None,
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
        self._timeout = timeout

    def solve(self):
        """
        Runs the DifferentialEvolutionSolver.

        Returns
        -------
        res : OptimizeResult
            The optimization result represented as a ``OptimizeResult`` object.
            Important attributes are: ``x`` the solution array, ``success`` a
            Boolean flag indicating if the optimizer exited successfully and
            ``message`` which describes the cause of the termination. See
            `OptimizeResult` for a description of other attributes.  If `polish`
            was employed, and a lower minimum was obtained by the polishing,
            then OptimizeResult also contains the ``jac`` attribute.
        """
        nit, warning_flag = 0, False
        status_message = _status_message["success"]

        # The population may have just been initialized (all entries are
        # np.inf). If it has you have to calculate the initial energies.
        # Although this is also done in the evolve generator it's possible
        # that someone can set maxiter=0, at which point we still want the
        # initial energies to be calculated (the following loop isn't run).
        if np.all(np.isinf(self.population_energies)):
            (
                self.feasible,
                self.constraint_violation,
            ) = self._calculate_population_feasibilities(self.population)

            # only work out population energies for feasible solutions
            self.population_energies[
                self.feasible
            ] = self._calculate_population_energies(self.population[self.feasible])

            self._promote_lowest_energy()

        # do the optimisation.
        start_time = time()
        while self._timeout is None or time() < start_time + self._timeout:
            # evolve the population by a generation
            try:
                next(self)
            except StopIteration:
                warning_flag = True
                if self._nfev > self.maxfun:
                    status_message = _status_message["maxfev"]
                elif self._nfev == self.maxfun:
                    status_message = (
                        "Maximum number of function evaluations" " has been reached."
                    )
                break

            if self.disp:
                print(
                    "differential_evolution step %d: f(x)= %g"
                    % (time() - start_time, self.population_energies[0])
                )

            # should the solver terminate?
            convergence = self.convergence

            if (
                self.callback
                and self.callback(
                    self._scale_parameters(self.population[0]),
                    convergence=self.tol / convergence,
                )
                is True
            ):

                warning_flag = True
                status_message = (
                    "callback function requested stop early " "by returning True"
                )
                break

            if np.any(np.isinf(self.population_energies)):
                intol = False
            else:
                intol = np.std(
                    self.population_energies
                ) <= self.atol + self.tol * np.abs(np.mean(self.population_energies))
            if warning_flag or intol:
                break

        else:
            status_message = _status_message["timeout"]
            warning_flag = True

        DE_result = OptimizeResult(
            x=self.x,
            fun=self.population_energies[0],
            nfev=self._nfev,
            nit=time() - start_time,
            message=status_message,
            success=(warning_flag is not True),
        )

        if self.polish:
            polish_method = "L-BFGS-B"

            if self._wrapped_constraints:
                polish_method = "trust-constr"

                constr_violation = self._constraint_violation_fn(DE_result.x)
                if np.any(constr_violation > 0.0):
                    warnings.warn(
                        "differential evolution didn't find a"
                        " solution satisfying the constraints,"
                        " attempting to polish from the least"
                        " infeasible solution",
                        UserWarning,
                    )

            result = minimize(
                self.func,
                np.copy(DE_result.x),
                method=polish_method,
                bounds=self.limits.T,
                constraints=self.constraints,
            )

            self._nfev += result.nfev
            DE_result.nfev = self._nfev

            # polishing solution is only accepted if there is an improvement in
            # cost function, the polishing was successful and the solution lies
            # within the bounds.
            if (
                result.fun < DE_result.fun
                and result.success
                and np.all(result.x <= self.limits[1])
                and np.all(self.limits[0] <= result.x)
            ):
                DE_result.fun = result.fun
                DE_result.x = result.x
                DE_result.jac = result.jac
                # to keep internal state consistent
                self.population_energies[0] = result.fun
                self.population[0] = self._unscale_parameters(result.x)

        if self._wrapped_constraints:
            DE_result.constr = [
                c.violation(DE_result.x) for c in self._wrapped_constraints
            ]
            DE_result.constr_violation = np.max(np.concatenate(DE_result.constr))
            DE_result.maxcv = DE_result.constr_violation
            if DE_result.maxcv > 0:
                # if the result is infeasible then success must be False
                DE_result.success = False
                DE_result.message = (
                    "The solution does not satisfy the"
                    " constraints, MAXCV = " % DE_result.maxcv
                )

        return DE_result
