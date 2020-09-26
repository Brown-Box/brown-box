from scipy.stats import norm

def neg_ei(gp, tr, max_y=0, xi=0.1, **_):
    def cost(**kwargs):
        X = tr.to_real_space(**kwargs)
        mean, std = gp.predict(X, return_std=True)
        a = (mean - max_y - xi)
        z = a / std
        return -(a * norm.cdf(z) + std * norm.pdf(z))
    return cost
