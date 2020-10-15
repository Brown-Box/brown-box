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

from ..utils import HyperTransformer


class GeneticSearchNonRandom:
    def __init__(
        self, transformer: HyperTransformer, random, cost_function, step=0
    ) -> None:
        self._transformer = transformer
        self._api_config = transformer.api_config
        self._start_time = None
        self._timeout = None
        self.top_points_real = []
        self.top_values = []
        self.random = random
        self.cost = cost_function
        self._iter = step

    def suggest(self, timeout: Optional[int] = None) -> dict:
        if timeout:
            self._start_time = time()
            self._timeout = timeout

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
            callback=None,
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
        tol=0.001,
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
        self._scale_mult = 1.0

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
        start_time = time()
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
        while self._timeout is None or time() < start_time + self._timeout:
            scale_mult = 1 - (time() - start_time) / self._timeout
            self._scale_mult = min(1, max(0.01, scale_mult))

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
            convergence = self.convergence + 1e-6

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

    def __next__(self):
        """
        Evolve the population by a single generation

        Returns
        -------
        x : ndarray
            The best solution from the solver.
        fun : float
            Value of objective function obtained from the best solution.
        """
        # the population may have just been initialized (all entries are
        # np.inf). If it has you have to calculate the initial energies
        if np.all(np.isinf(self.population_energies)):
            self.feasible, self.constraint_violation = (
                self._calculate_population_feasibilities(self.population))

            # only need to work out population energies for those that are
            # feasible
            self.population_energies[self.feasible] = (
                self._calculate_population_energies(
                    self.population[self.feasible]))

            self._promote_lowest_energy()

        if self.dither is not None:
            self.scale = self._scale_mult * (self.random_number_generator.rand()
                          * (self.dither[1] - self.dither[0]) + self.dither[0])

        if self._updating == 'immediate':
            # update best solution immediately
            for candidate in range(self.num_population_members):
                if self._nfev > self.maxfun:
                    raise StopIteration

                # create a trial solution
                trial = self._mutate(candidate)

                # ensuring that it's in the range [0, 1)
                self._ensure_constraint(trial)

                # scale from [0, 1) to the actual parameter value
                parameters = self._scale_parameters(trial)

                # determine the energy of the objective function
                if self._wrapped_constraints:
                    cv = self._constraint_violation_fn(parameters)
                    feasible = False
                    energy = np.inf
                    if not np.sum(cv) > 0:
                        # solution is feasible
                        feasible = True
                        energy = self.func(parameters)
                        self._nfev += 1
                else:
                    feasible = True
                    cv = np.atleast_2d([0.])
                    energy = self.func(parameters)
                    self._nfev += 1

                # compare trial and population member
                if self._accept_trial(energy, feasible, cv,
                                      self.population_energies[candidate],
                                      self.feasible[candidate],
                                      self.constraint_violation[candidate]):
                    self.population[candidate] = trial
                    self.population_energies[candidate] = energy
                    self.feasible[candidate] = feasible
                    self.constraint_violation[candidate] = cv

                    # if the trial candidate is also better than the best
                    # solution then promote it.
                    if self._accept_trial(energy, feasible, cv,
                                          self.population_energies[0],
                                          self.feasible[0],
                                          self.constraint_violation[0]):
                        self._promote_lowest_energy()

        elif self._updating == 'deferred':
            # update best solution once per generation
            if self._nfev >= self.maxfun:
                raise StopIteration

            # 'deferred' approach, vectorised form.
            # create trial solutions
            trial_pop = np.array(
                [self._mutate(i) for i in range(self.num_population_members)])

            # enforce bounds
            self._ensure_constraint(trial_pop)

            # determine the energies of the objective function, but only for
            # feasible trials
            feasible, cv = self._calculate_population_feasibilities(trial_pop)
            trial_energies = np.full(self.num_population_members, np.inf)

            # only calculate for feasible entries
            trial_energies[feasible] = self._calculate_population_energies(
                trial_pop[feasible])

            # which solutions are 'improved'?
            loc = [self._accept_trial(*val) for val in
                   zip(trial_energies, feasible, cv, self.population_energies,
                       self.feasible, self.constraint_violation)]
            loc = np.array(loc)
            self.population = np.where(loc[:, np.newaxis],
                                       trial_pop,
                                       self.population)
            self.population_energies = np.where(loc,
                                                trial_energies,
                                                self.population_energies)
            self.feasible = np.where(loc,
                                     feasible,
                                     self.feasible)
            self.constraint_violation = np.where(loc[:, np.newaxis],
                                                 cv,
                                                 self.constraint_violation)

            # make sure the best solution is updated if updating='deferred'.
            # put the lowest energy into the best solution position.
            self._promote_lowest_energy()

        return self.x, self.population_energies[0]

    next = __next__