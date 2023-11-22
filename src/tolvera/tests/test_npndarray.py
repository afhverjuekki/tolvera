import pytest
import numpy as np
import taichi as ti
from tolvera.npndarray_dict import NpNdarrayDict, np_vec2, np_vec3, np_vec4

shape = (2,2)

@pytest.fixture
def setup():
    return NpNdarrayDict({
        'i':  (np.int32, 2, 10),
        'f':  (np.float32, 0., 1.),
        'v2': (np_vec2, 0., 1.),
        'v3': (np_vec3, 0., 1.),
        'v4': (np_vec4, 0., 1.),
    }, shape)

@pytest.fixture
def setup_two():
    return NpNdarrayDict({
        'i':  (np.int32, 2, 10),
        'f':  (np.float32, 0., 1.),
        'v2': (np_vec2, 0., 1.),
        'v3': (np_vec3, 0., 1.),
        'v4': (np_vec4, 0., 1.),
    }, shape)

@pytest.fixture
def taichi_setup():
    ti.init()
    @ti.dataclass
    class TaichiData:
        i: ti.i32
        f: ti.f32
        v2: ti.math.vec2
        v3: ti.math.vec3
        v4: ti.math.vec4
    field = TaichiData.field(shape=shape)
    return field

@pytest.fixture
def taichi_setup_two():
    ti.init()
    @ti.dataclass
    class TaichiData:
        i: ti.i32
        f: ti.f32
        v2: ti.math.vec2
        v3: ti.math.vec3
        v4: ti.math.vec4
    field = TaichiData.field(shape=shape)
    index = len(shape) * (0,)
    field[index] = TaichiData(1, 1., (1., 1.), (1., 1., 1.), (1., 1., 1., 1.))
    return field

def test_init(setup):
    assert setup.data is not None

def test_set_value(setup):
    index = len(shape) * (0,)
    setup.set_value('i', index, 5)
    assert setup.data['i'][index] == 5

def test_get_value(setup):
    index = len(shape) * (0,)
    setup.set_value('i', index, 5)
    assert setup.get_value('i', index) == 5

def test_set_row(setup):
    index = (shape[1],)
    row = np.ones(index)
    setup.set_row('i', 0, row)
    assert np.array_equal(setup.data['i'][0], row)

def test_get_row(setup):
    index = (shape[1],)
    row = np.ones(index)
    setup.set_row('i', 0, row)
    assert np.array_equal(setup.get_row('i', 0), row)

def test_set_col(setup):
    index = (shape[0],)
    col = np.ones(index)
    setup.set_col('i', 0, col)
    assert np.array_equal(setup.data['i'][:,0], col)

def test_get_col(setup):
    index = (shape[0],)
    col = np.ones(index)
    setup.set_col('i', 0, col)
    assert np.array_equal(setup.get_col('i', 0), col)

def test_set_data(setup, setup_two):
    setup.set_data(setup_two.data)
    assert setup.data == setup_two.data

def test_set_row_for_attribute(setup):
    pass
    # setup.set_row_for_attribute()
    # assert 

def test_get_row_for_attribute(setup):
    pass
    # setup.get_row_for_attribute()
    # assert 

def test_set_col_for_attribute(setup):
    pass
    # setup.set_col_for_attribute()
    # assert 

def test_get_col_for_attribute(setup):
    pass
    # setup.get_col_for_attribute()
    # assert 

def test_randomise(setup):
    setup.randomise()
    index = len(shape) * (0,)
    assert setup.data['i'][index] != 0

def test_randomise_index(setup):
    pass
    # setup.randomise_index()
    # assert 

def test_randomise_row(setup):
    pass
    # setup.randomise_row()
    # assert 

def test_randomise_col(setup):
    pass
    # setup.randomise_col()
    # assert 

def test_randomise_attribute_index(setup):
    pass
    # setup.randomise_attribute_index()
    # assert 

def test_randomise_attribute_row(setup):
    pass
    # setup.randomise_attribute_row()
    # assert 

def test_randomise_attribute_col(setup):
    pass
    # setup.randomise_attribute_col()
    # assert 

def test_apply_function(setup):
    pass
    # setup.apply_function()
    # assert 

def test_elementwise_operation(setup):
    pass
    # setup.elementwise_operation()
    # assert 

def test_broadcast_operation(setup):
    pass
    # setup.broadcast_operation()
    # assert 

def test_taichi_to_numpy(setup, taichi_setup_two):
    np_data = taichi_setup_two.to_numpy()
    setup.set_data(np_data)
    assert np.array_equal(setup.data['i'], np_data['i'])

def test_taichi_from_numpy(setup, taichi_setup):
    setup.randomise()
    taichi_setup.from_numpy(setup.data)
    np_data = taichi_setup.to_numpy()
    assert np.array_equal(setup.data['i'], np_data['i'])
