import time
from operator import itemgetter

import bayesmark.random_search as rs
from bayesmark import np_util
from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main

N_SUGGESTIONS=1000

class RandomOptimizer(AbstractOptimizer):
    # Unclear what is best package to list for primary_import here.
    primary_import = "bayesmark"

    def __init__(self, api_config, random, cost_function):
        """Build wrapper class to use random search function in benchmark.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        cost_function : callback taking X.
        """
        AbstractOptimizer.__init__(self, api_config)
        self.random = random
        self.cost_function = cost_function

    def suggest(self, n=1, timeout=10):
        """Get suggestion.

        Parameters
        ----------
        n : int
            Desired number of parallel suggestions in the output

        Returns
        -------
        next_guess : list of dict
            List of `n_suggestions` suggestions to evaluate the objective
            function. Each suggestion is a dictionary where each key
            corresponds to a parameter being optimized.
        timeout: seconds
        """
        end_time = time.time() + timeout
        best_values = []  # tuples (X, Y)

        def get_guess():
            x = rs.suggest_dict(
                [], [], self.api_config, n_suggestions=N_SUGGESTIONS, random=self.random
            )
            _p = {k: [dic[k] for dic in x] for k in x[0]}
            y = self.cost_function(**_p)

            return [(_x, _y) for _x, _y in zip(x, y)] 

        iter_time = 0
        _iter = 0
        while time.time() + iter_time < end_time:
            start_time = time.time()
            guess = get_guess()
            best_values += guess
            iter_time = time.time() - start_time
            _iter += 1
        best_values.sort(key=itemgetter(1))

        # return just X so it is compatible with AbstractOptimizer
        return list(map(itemgetter(0), best_values[:n]))

    def observe(self, X, y):
        """Feed an observation back.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        """
        # Random search so don't do anything
        pass


if __name__ == "__main__":
    experiment_main(RandomOptimizer)
