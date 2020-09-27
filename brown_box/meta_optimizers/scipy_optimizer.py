import time
from scipy.optimize import minimize, Bounds

N_INIT = 10


def scipy_minimize(tr, gp, cost, random_state):
    # c0 = 1e10
    # for _ in range(N_INIT):  
    x0 = tr.randon_continuous(1, random_state)
    ret = minimize(cost, x0, method="l-bfgs-b", bounds=Bounds(tr._lb, tr._ub))
    x = tr.to_hyper_space(ret.x.reshape(1, -1))
    if ret.success:
        c = cost(ret.x)
        print(c, x)
        #     if c < c0:
        #         c0 = c
    return {k:v[0][0] for k,v in x.items()}