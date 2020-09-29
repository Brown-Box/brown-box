import time
from scipy.optimize import minimize, Bounds
from bayesmark.abstract_optimizer import AbstractOptimizer


class SciPyOptimizer(AbstractOptimizer):
    def __init__(self, tr, random, cost_function):
        self.random=random
        self.tr=tr
        self.cost=cost_function

    def suggest(self, timeout=10):
        x1 = self.tr.random_continuous(1, self.random)
        x1 = self.tr.continuous_transform(x1)
        c0 = self.cost(x1)
        x = self.tr.to_hyper_space(x1)
        end_time = time.time() + timeout
        iter_time = 0
        _iter=0
        while (time.time() + 1.25*iter_time) < end_time:
            start_time = time.time()
            x0 = self.tr.random_continuous(1, self.random)
            ret = minimize(self.cost, x0, method="l-bfgs-b", bounds=Bounds(self.tr._lb, self.tr._ub))
            iter_time = time.time() - start_time
            if ret.success:
                x1 = self.tr.continuous_transform(ret.x.reshape(1, -1))
                c = self.cost(x1)
                if c < c0:
                    c0 = c
                    x = self.tr.to_hyper_space(x1)
                    # print(_iter, c, x,)
            _iter += 1
        return {k:v[0][0] for k,v in x.items()}

    def observe(self, x, Y):
        pass