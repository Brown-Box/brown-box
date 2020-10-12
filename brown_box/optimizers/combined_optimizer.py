from ..optimizers import BrownBoxAbstractOptimizer


class CombinedOptimizer(BrownBoxAbstractOptimizer):
    primary_import = "bayesmark"

    def __init__(self, api_config, optimizers=[BrownBoxAbstractOptimizer]):
        """This combines multiple optimizers to make more robust optimizer

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        """
        super().__init__(api_config)

        self.optimizers = optimizers

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
        optimizer = self.optimizers[self.current_iteration % len(self.optimizers)]
        return optimizer.suggest(n_suggestions)

    def observe(self, X, y):
        """Feed the observations back to hyperopt.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated.
        """
        super().observe(X, y)
        for optimizer in self.optimizers:
            optimizer.observe(X, y)
