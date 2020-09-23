import pytest
import numpy as np

from hyper_transformer import HyperTransformer


def single_real_range_linear():
    api_config = {
        "x": {"type": "real", "space": "linear", "range": (-10, 10)},
    }
    x_in = [5.0, 3.0, 4.0]
    tr = HyperTransformer(api_config)
    y = tr.to_real_space(x=x_in)
    np.testing.assert_equal(np.stack([x_in], axis=1), y)

def single_real_values_linear():
    api_config = {
        "x": {"type": "real", "space": "linear", "values": np.arange(4)},
    }
    x_in = [5.0, 3.0, 4.0, 1.0]
    tr = HyperTransformer(api_config)
    y = tr.to_real_space(x=x_in)
    np.testing.assert_equal(np.stack([x_in],1), y)
    z = tr.to_hyper_space(y)
    np.testing.assert_equal(np.stack([x_in],1), z["x"])
    r = tr.continuous_transform(y)
    np.testing.assert_equal(np.stack([[3,3,3,1]],1), r)

def single_int_range_linear():
    api_config = {
        "x": {"type": "int", "space": "linear", "range": (-10, 10)},
    }
    x_in = [5, 3, 4]
    tr = HyperTransformer(api_config)
    y = tr.to_real_space(x=x_in)
    np.testing.assert_equal(np.stack([x_in],1), y)
    z = tr.to_hyper_space(y+0.1)
    np.testing.assert_equal(np.stack([x_in],1), z["x"])
    r = tr.continuous_transform(y)
    np.testing.assert_equal(np.stack([x_in],1), r)
    
    # coercion
    r = tr.continuous_transform(y+10)
    np.testing.assert_equal(np.stack([[10, 10, 10]],1), r)

def single_int_range_log():
    api_config = {
        "x": {"type": "int", "space": "log", "range": (1, 100)},
    }
    x_in = [1, 10, 100]
    tr = HyperTransformer(api_config)
    y = tr.to_real_space(x=x_in)
    np.testing.assert_equal(np.stack([[0,1,2]],1), y)
    z = tr.to_hyper_space(y)
    np.testing.assert_equal(np.stack([x_in],1), z["x"])
    r = tr.continuous_transform(y)
    np.testing.assert_equal(np.stack(np.log10([x_in]),1), r)
    
    # coercion
    r = tr.continuous_transform(y+10)
    np.testing.assert_equal(np.stack([[2, 2, 2]],1), r)

def single_bool_values_log():
    api_config = {
        "x": {"type": "bool"},
    }
    x_in = [True, True, False]
    tr = HyperTransformer(api_config)
    y = tr.to_real_space(x=x_in)
    np.testing.assert_equal(np.stack([[1,1,0]],1), y)
    z = tr.to_hyper_space(y)
    np.testing.assert_equal(np.stack([x_in],1), z["x"])
    r = tr.continuous_transform(y)
    np.testing.assert_equal(np.stack([[1, 1,0]],1), r)
    
    # coercion
    r = tr.continuous_transform(y+10)
    np.testing.assert_equal(np.stack([[1, 1, 1]],1), r)

    # coercion
    r = tr.continuous_transform(y-10)
    np.testing.assert_equal(np.stack([[0, 0, 0]],1), r)

def single_category()):
    api_config = {
        "x": {"type": "cat", "values": ["exp", "log", "abs"]},
    }
    x_in = ["exp", "exp", "log", "log"]
    tr = HyperTransformer(api_config)
    y = tr.to_real_space(x=x_in)
    np.testing.assert_equal(np.stack([[1,0,0],[1,0,0],[0,1,0],[0,1,0]]), y)
    z = tr.to_hyper_space(y)
    for _x, _z in zip(x_in, z["x"]):
        assert _x == _z
    r = tr.continuous_transform(y)
    np.testing.assert_equal(y, r)
    
    # coercion
    r = tr.continuous_transform(np.stack([[10,2,3],[2,33,-40],[3,40,30]]))
    np.testing.assert_equal(np.stack([[1,0,0],[0,1,0],[0,1,0]]), r)


def combined_int_cat_real()):
    api_config = {
        "x": {"type": "int", "space": "log", "range": (1, 100)},
        "f": {"type": "cat", "values": ["exp", "log", "abs"]},
        "y": {"type": "real", "space": "linear", "range": (1, 100)},
    }
    x_in = [1, 10, 100, 100]
    f_in = ["exp", "exp", "log", "abs"]
    y_in = [5, 10, 20, 50]
    tr = HyperTransformer(api_config)
    y = tr.to_real_space(x=x_in, f=f_in, y=y_in)
    np.testing.assert_equal(np.stack([[0,1,2,np.log10(100)],[1,1,0,0],[0,0,1,0],[0,0,0,1], [5,10,20, 50]],1), y)
    z = tr.to_hyper_space(y)
    np.testing.assert_equal(np.stack([x_in],1), z["x"])
    np.testing.assert_equal(np.stack([y_in],1), z["y"])
    for _f, _z in zip(f_in, z["f"]):
        assert _f == _z
    r = tr.continuous_transform(y)
    np.testing.assert_equal(y, r)
    
    # coercion
    r = tr.continuous_transform(y+np.stack([[0,0,0.1,0],[10,2,0,0],[1,-20,0.5,0],[0,1,0,0], [0,0,0,0]], 1))
    np.testing.assert_equal(y, r)
