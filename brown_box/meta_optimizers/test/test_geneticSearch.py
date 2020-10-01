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


def more_better_fit_function(values):
    x = values[0]
    return - x


def fit_function_cat(values):
    cat_value = values
    if np.array_equal([1.0, 0.0, 0.0], cat_value):
        return 0
    elif np.array_equal([0.0, 1.0, 0.0], cat_value):
        return 1
    elif np.array_equal([0.0, 0.0, 1.0], cat_value):
        return 2
    else:
        raise ValueError(f"Bad value: {values}")


def fit_function_bool(values):
    x = values[0]
    return float(not x)


def fit_function_mixed_type(values):
    x = values[0]
    b = values[1]
    c = values[2:]
    y_x = 2 * x ** 2 + 4 * x + 2
    y_b = float(not b)
    if np.array_equal([1.0, 0.0, 0.0], c):
        y_c = 0
    elif np.array_equal([0.0, 1.0, 0.0], c):
        y_c = 1
    elif np.array_equal([0.0, 0.0, 1.0], c):
        y_c = 2
    else:
        raise ValueError(f"Bad value: {c}")
    return y_x + y_b + y_c  # == 0 for (-1, 0, first_value)


def test_happy_path():
    api_config = {
        "x": {"type": "real", "space": "linear", "range": (-10, 10)},
    }
    transformer = HyperTransformer(api_config)
    gs = GeneticSearch(transformer)
    proposal = gs.suggest(basic_fit_function)

    assert pytest.approx(-1, 0.1) == proposal["x"]


def test_timeout_callback():
    api_config = {
        "x": {"type": "real", "space": "linear", "range": (-10, 10)},
    }
    transformer = HyperTransformer(api_config)
    gs = GeneticSearch(transformer)
    start_time = time()
    proposal = gs.suggest(basic_fit_function, timeout=0.001)
    end_time = time()

    assert pytest.approx(-1, 1) == proposal["x"]
    assert end_time - start_time < 0.1


def test_log_space():
    api_config = {
        "x": {"type": "real", "space": "log", "range": (0.1, 100)},
    }
    transformer = HyperTransformer(api_config)
    gs = GeneticSearch(transformer)
    proposal = gs.suggest(basic_fit_function_10)

    assert pytest.approx(10, 0.1) == proposal["x"]


def test_categorical():
    api_config = {
        "x": {"type": "cat", "values": ["exp", "log", "abs"]},
    }
    transformer = HyperTransformer(api_config)
    gs = GeneticSearch(transformer)
    proposal = gs.suggest(fit_function_cat)

    assert "exp" == proposal["x"]


def test_bool():
    api_config = {
        "x": {"type": "bool"},
    }
    transformer = HyperTransformer(api_config)
    gs = GeneticSearch(transformer)
    proposal = gs.suggest(fit_function_bool)

    assert 1 == proposal["x"]


def test_mixed_types():
    api_config = {
        "x": {"type": "real", "space": "linear", "range": (-10, 10)},
        "b": {"type": "bool"},
        "c": {"type": "cat", "values": ["exp", "log", "abs"]},
    }
    transformer = HyperTransformer(api_config)
    gs = GeneticSearch(transformer)
    proposal = gs.suggest(fit_function_mixed_type)
    assert (
        pytest.approx(-1, 0.1) == proposal["x"]
        and 1 == proposal["b"]
        and "exp" == proposal["c"]
    )


def test_bounds_log():
    min_max_iter = 10
    max_max_iter = 5000
    api_config = {
        "max_iter": {"type": "int", "space": "log", "range": (min_max_iter, max_max_iter)},
    }
    transformer = HyperTransformer(api_config)
    gs = GeneticSearch(transformer)
    for i in range(10):
        proposal = gs.suggest(more_better_fit_function, seed=i)
        print(proposal)
        assert min_max_iter <= proposal["max_iter"] <= max_max_iter
        # TODO you should get 5000 here ideally (there is some rounding/transformation error) (similar for minimum)