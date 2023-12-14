import numpy as np
import pytest

from tolvera import Tolvera


def infun(*args, **kwargs):
    data = np.random.rand(2).tolist()
    print(f"infun: {data}, args:{args}, kwargs:{kwargs}")
    return data


def outfun(data):
    print(f"outfun: {data}")
    return data


@pytest.fixture
def setup():
    return Tolvera(iml=True)


def test_iml_init(setup):
    tv = setup
    assert tv.iml is not None, "IML instance not created."


def test_iml_vec2vec_call(setup):
    tv = setup
    tv.iml.test2test = {
        "type": "vec2vec",
        "size": (2, 2),
        "randomise": True,
    }
    a, aa = tv.iml("test2test", [0.5, 0.5]), tv.iml.test2test([0.5, 0.5])
    b, bb = tv.iml("test2test", [0.2, 0.2]), tv.iml.test2test([0.2, 0.2])
    c, cc = tv.iml("test2test", [0.1, 0.1]), tv.iml.test2test([0.1, 0.1])
    d, dd = tv.iml("test2test", [0.0, 0.0]), tv.iml.test2test([0.0, 0.0])
    assert a == aa, f"{a} != {aa}"
    assert b == bb, f"{b} != {bb}"
    assert c == cc, f"{c} != {cc}"
    assert d == dd, f"{d} != {dd}"


def test_iml_vec2fun_call(setup):
    tv = setup
    tv.iml.test2test = {
        "type": "vec2fun",
        "size": (2, 2),
        "io": (None, outfun),
        "randomise": True,
    }
    a, aa = tv.iml("test2test", [0.5, 0.5]), tv.iml.test2test([0.5, 0.5])
    b, bb = tv.iml("test2test", [0.2, 0.2]), tv.iml.test2test([0.2, 0.2])
    c, cc = tv.iml("test2test", [0.1, 0.1]), tv.iml.test2test([0.1, 0.1])
    d, dd = tv.iml("test2test", [0.0, 0.0]), tv.iml.test2test([0.0, 0.0])
    assert a == aa, f"{a} != {aa}"
    assert b == bb, f"{b} != {bb}"
    assert c == cc, f"{c} != {cc}"
    assert d == dd, f"{d} != {dd}"


def test_iml_fun2vec_call(setup):
    tv = setup
    tv.iml.test2test = {
        "type": "fun2vec",
        "size": (2, 2),
        "io": (infun, None),
        "randomise": True,
    }
    a, aa = tv.iml("test2test", [0.5, 0.5]), tv.iml.test2test([0.5, 0.5])
    b, bb = tv.iml("test2test", [0.2, 0.2]), tv.iml.test2test([0.2, 0.2])
    c, cc = tv.iml("test2test", [0.1, 0.1]), tv.iml.test2test([0.1, 0.1])
    d, dd = tv.iml("test2test", [0.0, 0.0]), tv.iml.test2test([0.0, 0.0])
    assert a != aa, f"{a} == {aa}"
    assert b != bb, f"{b} == {bb}"
    assert c != cc, f"{c} == {cc}"
    assert d != dd, f"{d} == {dd}"


def test_iml_fun2fun_call(setup):
    tv = setup
    tv.iml.test2test = {
        "type": "fun2fun",
        "size": (2, 2),
        "io": (infun, outfun),
        "randomise": True,
    }
    a, aa = tv.iml("test2test", [0.5, 0.5]), tv.iml.test2test([0.5, 0.5])
    b, bb = tv.iml("test2test", [0.2, 0.2]), tv.iml.test2test([0.2, 0.2])
    c, cc = tv.iml("test2test", [0.1, 0.1]), tv.iml.test2test([0.1, 0.1])
    d, dd = tv.iml("test2test", [0.0, 0.0]), tv.iml.test2test([0.0, 0.0])
    assert a != aa, f"{a} == {aa}"
    assert b != bb, f"{b} == {bb}"
    assert c != cc, f"{c} == {cc}"
    assert d != dd, f"{d} == {dd}"
