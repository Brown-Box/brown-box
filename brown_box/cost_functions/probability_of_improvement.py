from scipy.stats import norm


def poi(gp, tr, min_y=0, xi=0.01, **_):
    def cost(**kwargs):
        X = tr.to_real_space(**kwargs)
        mean, std = gp.predict(X, return_std=True)
        z = (mean - min_y - xi) / std
        return norm.cdf(z)

    return cost

def poi_real(gp, tr, min_y=0, xi=0.01, **_):
    def cost(X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        mean, std = gp.predict(X, return_std=True)
        z = (mean - min_y - xi) / std
        return norm.cdf(z)

    return cost