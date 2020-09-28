import time
from scipy.optimize import minimize, Bounds

N_INIT = 5


def scipy_minimize(tr, gp, cost, random_state):
    x1 = tr.randon_continuous(1, random_state)
    x1 = tr.continuous_transform(x1)
    c0 = cost(x1)
    x = tr.to_hyper_space(x1)
    for _ in range(N_INIT):  
        x0 = tr.randon_continuous(1, random_state)
        ret = minimize(cost, x0, method="l-bfgs-b", bounds=Bounds(tr._lb, tr._ub))
        if ret.success:
            x1 = tr.continuous_transform(ret.x.reshape(1, -1))
            c = cost(x1)
            # print(c, x)
            if c < c0:
                c0 = c
                x = tr.to_hyper_space(x1)
    return {k:v[0][0] for k,v in x.items()}