def ucb(gp, tr, kappa=2.6, **_):
    def cost(**kwargs):
        X = tr.to_real_space(**kwargs)
        mean, std = gp.predict(X, return_std=True)
        return mean - kappa * std

    return cost


def ucb_real(gp, tr, kappa=2.6, **_):
    def cost(X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        mean, std = gp.predict(X, return_std=True)
        return mean - kappa * std

    return cost
