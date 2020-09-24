import pytest
from time import time


from meta_optimizer.geneticSearch import GeneticSearch


def basic_fit_function(x):
    return 2 * x ** 2 + 4 * x + 2


def test_happy_path():
    api_config = {
        "x": {"type": "real", "space": "linear", "range": (-10, 10)},
    }
    gs = GeneticSearch(api_config, basic_fit_function)
    proposal = gs.search(1)

    assert pytest.approx(-1, 0.1) == proposal["x"]


def test_timeout_callback():
    api_config = {
        "x": {"type": "real", "space": "linear", "range": (-10, 10)},
    }
    gs = GeneticSearch(api_config, basic_fit_function)
    start_time = time()
    proposal = gs.search(1, timeout=0.001)
    end_time = time()

    assert pytest.approx(-1, 0.1) == proposal["x"]
    assert end_time - start_time < 0.1
