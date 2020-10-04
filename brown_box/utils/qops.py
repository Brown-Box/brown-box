from bayesmark.space import bilog, biexp
from scipy.special import logit, expit
import numpy as np

def qlog10(x):
    return np.log10(np.asarray(x, dtype=int))

def qexp10(x):
    return np.rint(np.power(10.0, x)).astype(int)

def qbilog(x):
    return bilog(np.asarray(x, dtype=int))

def qbiexp(x):
    return biexp(x).astype(int)

def qlogit(x):
    return logit(np.asarray(x, dtype=int))

def qexpit(x):
    return expit(x).astype(int)
