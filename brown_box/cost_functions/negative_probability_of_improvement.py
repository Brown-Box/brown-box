from scipy.stats import norm


def neg_poi(gp, tr, max_y=0, xi=0.1, **_):
    def cost(**kwargs):
        X = tr.to_real_space(**kwargs)
        mean, std = gp.predict(X, return_std=True)
        z = (mean - max_y - xi) / std
        return -norm.cdf(z)

    return cost
