from scipy.stats import norm


def poi(gp, tr, min_y=0, xi=0.1, **_):
    def cost(**kwargs):
        X = tr.to_real_space(**kwargs)
        mean, std = gp.predict(X, return_std=True)
        z = (mean + min_y + xi) / std
        return norm.cdf(z)

    return cost

def poi_real(gp, tr, min_y=0, xi=0.1, **_):
    def cost(X):
        mean, std = gp.predict(X.reshape(1, -1), return_std=True)
        z = (mean + min_y + xi) / std
        return norm.cdf(z)

    return cost