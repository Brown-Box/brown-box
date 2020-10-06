from bayesmark.space import bilog, biexp
import numpy as np

def qlog10(x):
    return np.log10(np.rint(x).astype(int))

def qexp10(x):
    return np.rint(np.power(10.0, x)).astype(int)

def qbilog(x):
    return bilog(np.rint(x).astype(int))

def qbiexp(x):
    return np.rint(biexp(x)).astype(int)
