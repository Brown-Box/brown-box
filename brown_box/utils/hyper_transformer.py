import numpy as np
from typing import Dict
from collections import OrderedDict

from .qops import qbiexp, qbilog, qexp10, qlog10, qexpit, qlogit
from bayesmark.space import bilog, biexp
from scipy.special import logit, expit

def exp10(x):
    return np.power(10, x).astype(int)

CONT_REAL = {
    "int":{
        "log": qlog10,
        "bilog": qbilog,
        "logit": qlogit,
        "linear": lambda x: np.asarray(x, np.int),
        },
    "real":{
        "log": np.log10,
        "bilog": bilog,
        "logit": logit,
        "linear": lambda x: np.asarray(x, np.float),
    },
}

CONT_HYPER = {
    "int":{
        "log": qexp10,
        "bilog": qbiexp,
        "logit": qexpit,
        "linear": lambda x: np.asarray(x, np.int),
        },
    "real":{
        "log": exp10,
        "bilog": biexp,
        "logit": expit,
        "linear": lambda x: np.asarray(x, np.float),
    },
}


def cont_coerc(spec):
    _type = spec["type"]
    _space = spec["space"]
    if "values" in spec:
        vals = np.asarray([CONT_REAL[_type][_space](val) for val in spec["values"]])
        def _coerc(x):
            cross = abs(np.repeat(x, vals.shape[0]).reshape(-1, vals.shape[0])-vals)
            return vals[np.argmin(cross, axis=1)][:, None]
        return _coerc
    if "range" in spec:
        rng = [CONT_REAL[_type][_space](bound) for bound in spec["range"]]
        if _type == "real" or _space == "linear":
            def _coerc(x):
                return np.clip(x, rng[0], rng[1])
            return _coerc

        if _type == "int" and _space == "log":
            def _coerc(x):
                y = np.log10(np.power(10, x).astype(int))
                return np.clip(y, rng[0], rng[1])
            return _coerc

        if _type == "int" and _space == "bilog":
            def _coerc(x):
                y = bilog(biexp(x).astype(int))
                return np.clip(y, rng[0], rng[1])
            return _coerc

        if _type == "int" and _space == "logit":
            def _coerc(x):
                y = logit(expit(x).astype(int))
                return np.clip(y, rng[0], rng[1])
            return _coerc


def cat_real(values):
    def _real(cats):
        _n = len(cats)
        onehot = np.zeros((_n, len(values)))
        idx = [values.index(cat) for cat in cats ]
        onehot[range(_n), idx] = 1
        return onehot
    return _real

def cat_hyper(values):
    def _real(points):
        idx = np.argmax(points, axis=1)
        return [values[i] for i in idx]
    return _real

def hardmax(points):
    hard = np.zeros(points.shape)
    idx = np.argmax(points, axis=1)
    hard[range(idx.size), idx] = 1
    return hard

class HyperTransformer():
    def __init__(self, api_config: Dict):
        self.api_config = OrderedDict(api_config)

        self._reals = OrderedDict()
        self._coercs = []
        self._hypers = OrderedDict()
        self._slices = OrderedDict()

        _col = 0
        for key, spec in self.api_config.items():
            _type = spec["type"]
            if _type in {"int", "real"}:
                _space = spec["space"]
                self._reals[key] = CONT_REAL[_type][_space]
                self._hypers[key] = CONT_HYPER[_type][_space]
                self._coercs.append(cont_coerc(spec))
                self._slices[key] = slice(_col, _col+1)
                _col += 1
            if _type == "bool":
                self._reals[key] = lambda x: np.asarray(x, np.float)
                self._hypers[key] = lambda x: x > 0.5
                self._coercs.append(lambda x: np.clip(np.round(x, 0), 0, 1))
                self._slices[key] = slice(_col, _col+1)
                _col += 1
            if _type == "cat":
                _vals = spec["values"]
                _n = len(_vals)
                self._reals[key] = cat_real(_vals)
                self._hypers[key] = cat_hyper(_vals)
                self._coercs.append(hardmax)
                self._slices[key] = slice(_col, _col + _n)
                _col += _n



    def to_real_space(self, **kwargs) -> np.array:
        """Convert values from hyper space to real linear space.

        Categoricals values are one-hot encoded, boleans are retyped,
        integers are coerced. Moreover, non-linear spaces are resampled.
        """
        _real_vec = []
        for key, _real in self._reals.items():
            _real_vec.append(_real(kwargs[key]))
        return np.column_stack(_real_vec)

    def to_hyper_space(self, points: np.array) -> Dict:
        """Convert values from real linear space to hyper space.

        One-hot encoded categoricals are decoded, boleans are retyped,
        integers are coerced. Moreover, non-linear spaces are resampled.
        """
        hyper_dict = {}
        for key, _hyper in self._hypers.items():
            hyper_dict[key] = _hyper(points[:, self._slices[key]])
        return hyper_dict

    def continuous_transform(self, points: np.array) -> np.array:
        """ Apply constraints on points in real continuous space.

        This function is necessary for TransformerKernel definition. It
        takes input points as a continuous array and applies constraints
        defined from api_config spec.
        Constraints are following:
            Parts of vector representing individual one-hot encoded
            categorical variables are softmaxed.
            Parts of vector representing boolean or integer values are
            coerced.

        Note: This function does not transform from log to linear space
        hence it is a matter of `to_real_space` or `to_hyper_space`,
        """
        new_points = points.copy()
        for sl, _coerc in zip(self._slices.values(), self._coercs):
            new_points[:, sl] = _coerc(points[:, sl])
        return new_points

