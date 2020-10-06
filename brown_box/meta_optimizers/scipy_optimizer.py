from itertools import cycle
import time
import numpy as np

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

class SciPyOptimizerNonRandom(AbstractOptimizer):
    def __init__(self, tr, random, cost_function):
        self.random=random
        self.tr=tr
        self.cost=cost_function
        self.top_points_real = []
        self.top_values = []

    def suggest(self, timeout=10):
        x1 = np.expand_dims(self.top_points_real[0, ...], 0)
        x1 = self.tr.continuous_transform(x1)
        c0 = self.cost(x1)
        x = self.tr.to_hyper_space(x1)
        iter_time = 0
        _iter_c=0
        _iter=cycle(self.top_points_real.tolist())
        end_time = time.time() + timeout
        _x0=self.top_points_real[0, ...]
        while (time.time() + 1.25*iter_time) < end_time:
            start_time = time.time()
            dx = self.tr.random_continuous(1, self.random)*0.01
            x0 = _x0 + dx
            ret = minimize(self.cost, x0, method="l-bfgs-b", bounds=Bounds(self.tr._lb, self.tr._ub))
            iter_time = time.time() - start_time
            if ret.success:
                _x = np.expand_dims(ret.x, 0)
                x1 = self.tr.continuous_transform(_x)
                c = self.cost(x1)
                if c < c0:
                    c0 = c
                    x = self.tr.to_hyper_space(x1)
                    # print(_iter_c, c, x, dx)
            _iter_c += 1
        return {k:v[0][0] for k,v in x.items()}

    def observe(self, x, Y):
        indices = np.argsort(Y)
        self.top_values = [Y[idx] for idx in indices]
        _x = [x[idx] for idx in indices]
        _p = {k: [dic[k] for dic in _x] for k in _x[0]}
        X = self.tr.to_real_space(**_p)
        self.top_points_real = X
