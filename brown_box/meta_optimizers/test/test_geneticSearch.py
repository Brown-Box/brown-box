import pytest
from time import time


from brown_box_package.brown_box.meta_optimizers.geneticSearch import GeneticSearch
from brown_box_package.brown_box.utils.hyper_transformer import HyperTransformer


def basic_fit_function(x):
    return 2 * x ** 2 + 4 * x + 2


def cat_fit_function(x):
    return ["exp", "log", "abs"].index(x)


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

    assert pytest.approx(-1, 0.1) == proposal["x"]
    assert end_time - start_time < 0.1
