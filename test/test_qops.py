import pytest
import numpy as np

from brown_box.utils import qbiexp, qbilog, qexp10, qlog10


def test_qlog10():
    in_log=(1,10,100)
    out_log=np.asarray((0.0,1.0,2.0))
    np.testing.assert_equal(qlog10(in_log), out_log)
    in_log=(1.1, 10.3, 100.7)
    out_log=np.asarray((0.0, 1.0, np.log10(101)))
    np.testing.assert_equal(qlog10(in_log), out_log)


def test_qexp10():
    in_exp=(-1, 1, 3)
    out_exp=np.asarray((0, 10, 1000))
    np.testing.assert_equal(qexp10(in_exp), out_exp)
    in_exp=(-1.0000001, 1.0002, 3.01)
    out_exp=np.asarray((0, 10, 1023))
    np.testing.assert_equal(qexp10(in_exp), out_exp)


def test_qbilog():
    in_log=(-100, -10, 10,100)
    _log11 = np.log(11)
    _log101 = np.log(101)
    _log12 = np.log(12)
    _log102 = np.log(102)
    # due to bias term in bilog
    out_log=np.asarray((-_log101, -_log11, _log11, _log101))
    np.testing.assert_equal(qbilog(in_log), out_log)
    in_log=(-100.7, -10.4, 10.9, 100.2)
    out_log=np.asarray((-_log102, -_log11, _log12, _log101))
    np.testing.assert_equal(qbilog(in_log), out_log)


def test_qbiexp():
    _log11 = np.log(11)
    _log101 = np.log(101)
    _log12 = np.log(12)
    _log102 = np.log(102)
    in_exp=(-_log11, -_log101, _log102)
    out_exp=np.asarray((-10, -100, 101))
    np.testing.assert_equal(qbiexp(in_exp), out_exp)
    in_exp=(-_log11-0.0000001, -_log101 + 0.02, _log102-0.003)
    out_exp=np.asarray((-10, -98, 101))
    np.testing.assert_equal(qbiexp(in_exp), out_exp)
