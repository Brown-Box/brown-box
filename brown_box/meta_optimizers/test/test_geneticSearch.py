import pytest
from time import time

import numpy as np

from brown_box_package.brown_box.meta_optimizers.geneticSearch import GeneticSearch
from brown_box_package.brown_box.utils.hyper_transformer import HyperTransformer


def basic_fit_function(values):
    x = values[0]
    return 2 * x ** 2 + 4 * x + 2


def basic_fit_function_10(values):
    x = values[0]
    return abs(x - 1)


def fit_function_cat(values):
    cat_value = values[0]
    if np.array_equal([1., 0., 0.], cat_value):
        return 0
    elif np.array_equal([0., 1., 0.], cat_value):
        return 1
    elif np.array_equal([0., 0., 1.], cat_value):
        return 2
    else:
        raise ValueError(f"Bad value of x_real: {values}")


def fit_function_bool(values):
    x = values[0]
    return float(not x)


def test_happy_path():
    api_config = {
        "x": {"type": "real", "space": "linear", "range": (-10, 10)},
    }
    transformer = HyperTransformer(api_config)
    gs = GeneticSearch(transformer, basic_fit_function)
    proposal = gs.search()

    assert pytest.approx(-1, 0.1) == proposal["x"]


def test_timeout_callback():
    api_config = {
        "x": {"type": "real", "space": "linear", "range": (-10, 10)},
    }
    transformer = HyperTransformer(api_config)
    gs = GeneticSearch(transformer, basic_fit_function)
    start_time = time()
    proposal = gs.search(timeout=0.001)
    end_time = time()

    assert pytest.approx(-1, 1) == proposal["x"]
    assert end_time - start_time < 0.1


def test_log_space():
    api_config = {
        "x": {"type": "real", "space": "log", "range": (0.1, 100)},
    }
    transformer = HyperTransformer(api_config)
    gs = GeneticSearch(transformer, basic_fit_function_10)
    proposal = gs.search()

    assert pytest.approx(10, 0.1) == proposal["x"]


def test_categorical():
    api_config = {
        "f": {"type": "cat", "values": ["exp", "log", "abs"]},
    }
    transformer = HyperTransformer(api_config)
    gs = GeneticSearch(transformer, fit_function_cat)
    proposal = gs.search()

    assert "exp" == proposal["f"]


def test_bool():
    api_config = {
        "v": {"type": "bool"},
    }
    transformer = HyperTransformer(api_config)
    gs = GeneticSearch(transformer, fit_function_bool)
    proposal = gs.search()

    assert 1 == proposal["v"]


# test cat + real