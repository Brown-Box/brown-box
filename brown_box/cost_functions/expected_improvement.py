from scipy.stats import norm


def ei(gp, tr, min_y=0, xi=0.1, **_):
    def cost(**kwargs):
        X = tr.to_real_space(**kwargs)
        mean, std = gp.predict(X, return_std=True)
        a = mean + min_y + xi
        z = a / std
        return a * norm.cdf(z) + std * norm.pdf(z)

    return cost


def ei_real(gp, tr, min_y=0, xi=0.01, **_):
    def cost(X):
        mean, std = gp.predict(X.reshape(1, -1), return_std=True)
        a = mean + min_y + xi
        z = a / std
        return a * norm.cdf(z) + std * norm.pdf(z)

    return cost
