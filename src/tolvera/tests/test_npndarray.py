import pytest
import numpy as np
import taichi as ti
from taichi.lang.struct import StructField
from tolvera.npndarray_dict import NpNdarrayDict, np_vec2, np_vec3, np_vec4, dict_to_vector_args, shapes_from_ndarray_dict, ndarraydict_from_vector_args

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

def test_init(setup: NpNdarrayDict):
    assert setup.data is not None

def test_set_value(setup: NpNdarrayDict):
    index = len(shape) * (0,)
    setup.set_value('i', index, 5)
    assert setup.data['i'][index] == 5

def test_get_value(setup: NpNdarrayDict):
    index = len(shape) * (0,)
    setup.set_value('i', index, 5)
    assert setup.get_value('i', index) == 5

# def test_set_row(setup):
#     index = (shape[1],)
#     row = np.ones(index)
#     setup.set_row('i', 0, row)
#     assert np.array_equal(setup.data['i'][0], row)

# def test_get_row(setup):
#     index = (shape[1],)
#     row = np.ones(index)
#     setup.set_row('i', 0, row)
#     assert np.array_equal(setup.get_row('i', 0), row)

# def test_set_col(setup):
#     index = (shape[0],)
#     col = np.ones(index)
#     setup.set_col('i', 0, col)
#     assert np.array_equal(setup.data['i'][:,0], col)

# def test_get_col(setup):
#     index = (shape[0],)
#     col = np.ones(index)
#     setup.set_col('i', 0, col)
#     assert np.array_equal(setup.get_col('i', 0), col)

def test_set_data(setup: NpNdarrayDict, setup_two: NpNdarrayDict):
    setup.set_data(setup_two.data)
    assert setup.data == setup_two.data

def test_set_row_for_attribute(setup: NpNdarrayDict):
    pass
    # setup.set_row_for_attribute()
    # assert 

def test_get_row_for_attribute(setup: NpNdarrayDict):
    pass
    # setup.get_row_for_attribute()
    # assert 

def test_set_col_for_attribute(setup: NpNdarrayDict):
    pass
    # setup.set_col_for_attribute()
    # assert 

def test_get_col_for_attribute(setup: NpNdarrayDict):
    pass
    # setup.get_col_for_attribute()
    # assert 

def test_randomise(setup: NpNdarrayDict):
    setup.randomise()
    index = len(shape) * (0,)
    assert setup.data['i'][index] != 0

def test_randomise_index(setup: NpNdarrayDict):
    pass
    # setup.randomise_index()
    # assert 

def test_randomise_row(setup: NpNdarrayDict):
    pass
    # setup.randomise_row()
    # assert 

def test_randomise_col(setup: NpNdarrayDict):
    pass
    # setup.randomise_col()
    # assert 

def test_randomise_attribute_index(setup: NpNdarrayDict):
    pass
    # setup.randomise_attribute_index()
    # assert 

def test_randomise_attribute_row(setup: NpNdarrayDict):
    pass
    # setup.randomise_attribute_row()
    # assert 

def test_randomise_attribute_col(setup: NpNdarrayDict):
    pass
    # setup.randomise_attribute_col()
    # assert 

def test_apply_function(setup: NpNdarrayDict):
    pass
    # setup.apply_function()
    # assert 

def test_elementwise_operation(setup: NpNdarrayDict):
    pass
    # setup.elementwise_operation()
    # assert 

def test_broadcast_operation(setup: NpNdarrayDict):
    pass
    # setup.broadcast_operation()
    # assert 

def test_taichi_to_numpy(setup: NpNdarrayDict, taichi_setup_two: StructField):
    np_data = taichi_setup_two.to_numpy()
    setup.set_data(np_data)
    assert np.array_equal(setup.data['i'], np_data['i'])

def test_taichi_from_numpy(setup: NpNdarrayDict, taichi_setup: StructField):
    setup.randomise()
    taichi_setup.from_numpy(setup.data)
    np_data = taichi_setup.to_numpy()
    assert np.array_equal(setup.data['i'], np_data['i'])

@pytest.fixture
def setup_np_float32():
    data_dict = {
        'array1': (np.float32, 0, 10),
        'array2': (np.float32, 0, 10)
    }
    shape = (3, 3)
    np_ndarray_dict = NpNdarrayDict(data_dict, shape)
    return np_ndarray_dict

@pytest.fixture
def setup_np_vec3():
    data_dict = {
        'array1': (np.float32, 0, 10),
        'np_vec3': (np_vec3, 0, 10)
    }
    shape = (3, 3)
    np_ndarray_dict = NpNdarrayDict(data_dict, shape)
    return np_ndarray_dict

def test_initial_state_np_float32(setup_np_float32: NpNdarrayDict):
    np_ndarray_dict = setup_np_float32
    assert np_ndarray_dict.data['array1'].shape == (3, 3)
    assert np_ndarray_dict.data['array2'].shape == (3, 3)

def test_set_row_from_dict_np_float32(setup_np_float32: NpNdarrayDict):
    np_ndarray_dict = setup_np_float32
    row_values_dict = {
        'array1': [1, 2, 3],
        'array2': [4, 5, 6]
    }
    np_ndarray_dict.set_row_from_dict(1, row_values_dict)
    assert np_ndarray_dict.data['array1'][1, 0] == 1
    assert np_ndarray_dict.data['array1'][1, 1] == 2
    assert np_ndarray_dict.data['array1'][1, 2] == 3
    assert np_ndarray_dict.data['array2'][1, 0] == 4
    assert np_ndarray_dict.data['array2'][1, 1] == 5
    assert np_ndarray_dict.data['array2'][1, 2] == 6

def test_set_row_from_list_np_float32(setup_np_float32: NpNdarrayDict):
    np_ndarray_dict = setup_np_float32
    row_list = [7, 8, 9, 10, 11, 12]
    np_ndarray_dict.set_row_from_list(2, row_list)
    assert np_ndarray_dict.data['array1'][2, 0] == 7
    assert np_ndarray_dict.data['array1'][2, 1] == 8
    assert np_ndarray_dict.data['array1'][2, 2] == 9
    assert np_ndarray_dict.data['array2'][2, 0] == 10
    assert np_ndarray_dict.data['array2'][2, 1] == 11
    assert np_ndarray_dict.data['array2'][2, 2] == 12

def test_initial_state_np_vec3(setup_np_vec3: NpNdarrayDict):
    np_ndarray_dict = setup_np_vec3
    assert np_ndarray_dict.data['array1'].shape == (3, 3)
    assert np_ndarray_dict.data['np_vec3'].shape == (3, 3, 3)

def test_set_row_from_dict_np_vec3(setup_np_vec3: NpNdarrayDict):
    np_ndarray_dict = setup_np_vec3
    row_values_dict = {
        'array1': [1, 2, 3],
        'np_vec3': [[4, 4, 4], [3, 3, 3], [2, 2, 2]]
    }
    np_ndarray_dict.set_row_from_dict(1, row_values_dict)
    assert np_ndarray_dict.data['array1'][1, 0] == 1
    assert np_ndarray_dict.data['array1'][1, 1] == 2
    assert np_ndarray_dict.data['array1'][1, 2] == 3
    assert np_ndarray_dict.data['np_vec3'][1, 0, 0] == 4
    assert np_ndarray_dict.data['np_vec3'][1, 1, 0] == 3
    assert np_ndarray_dict.data['np_vec3'][1, 2, 0] == 2

def test_set_row_from_list_np_vec3(setup_np_vec3: NpNdarrayDict):
    np_ndarray_dict = setup_np_vec3
    row_list = [5, 6, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10]
    np_ndarray_dict.set_row_from_list(1, row_list)
    assert np_ndarray_dict.data['array1'][1, 0] == 5
    assert np_ndarray_dict.data['array1'][1, 1] == 6
    assert np_ndarray_dict.data['array1'][1, 2] == 7
    assert np_ndarray_dict.data['np_vec3'][1, 0, 0] == 8
    assert np_ndarray_dict.data['np_vec3'][1, 1, 0] == 9
    assert np_ndarray_dict.data['np_vec3'][1, 2, 0] == 10

def test_set_slice_from_dict_np_vec3(setup_np_vec3: NpNdarrayDict):
    np_ndarray_dict = setup_np_vec3
    slice_values_dict = {
        'array1': [[4, 5, 6], [7, 8, 9]],  # Corrected shape for 'array1'
        'np_vec3': [[[13, 13, 13], [14, 14, 14], [15, 15, 15]], 
                    [[16, 16, 16], [17, 17, 17], [18, 18, 18]]]
    }

    np_ndarray_dict.set_slice_from_dict(slice(1, 3), slice_values_dict)

    assert np.all(np_ndarray_dict.data['array1'][1, :] == np.array([4, 5, 6]))
    assert np.all(np_ndarray_dict.data['array1'][2, :] == np.array([7, 8, 9]))

    assert np.all(np_ndarray_dict.data['np_vec3'][1, :, :] == np.array([[13, 13, 13], [14, 14, 14], [15, 15, 15]]))
    assert np.all(np_ndarray_dict.data['np_vec3'][2, :, :] == np.array([[16, 16, 16], [17, 17, 17], [18, 18, 18]]))


def test_set_slice_from_list_np_vec3(setup_np_vec3: NpNdarrayDict):
    np_ndarray_dict = setup_np_vec3
    slice_values_list = [4, 5, 6, 7, 8, 9, 13, 13, 13, 14, 14, 14, 15, 15, 15, 16, 16, 16, 17, 17, 17, 18, 18, 18]

    np_ndarray_dict.set_slice_from_list(slice(1, 3), slice_values_list)

    # Checking values in 'array1'
    assert np.all(np_ndarray_dict.data['array1'][1, :] == np.array([4, 5, 6]))
    assert np.all(np_ndarray_dict.data['array1'][2, :] == np.array([7, 8, 9]))

    # Checking values in 'np_vec3'
    assert np.all(np_ndarray_dict.data['np_vec3'][1, :, :] == np.array([[13, 13, 13], [14, 14, 14], [15, 15, 15]]))
    assert np.all(np_ndarray_dict.data['np_vec3'][2, :, :] == np.array([[16, 16, 16], [17, 17, 17], [18, 18, 18]]))

@pytest.fixture
def setup_vector_args():
    setup = NpNdarrayDict({
        'i':  (np.int32, 2, 10),
        'f':  (np.float32, 0., 1.),
        'v2': (np_vec2, 0., 1.),
    }, (2,2))
    setup.randomise()
    return setup

def test_dict_to_vector_args(setup_vector_args):
    compare = setup_vector_args.data['i'].flatten()
    vector_args = dict_to_vector_args(setup_vector_args.data)
    assert np.all(compare == vector_args[1:5])

def test_shapes_from_ndarray_dict(setup_vector_args):
    shapes = shapes_from_ndarray_dict(setup_vector_args.data)
    assert shapes['i'] == (2, 2)
    assert shapes['f'] == (2, 2)
    assert shapes['v2'] == (2, 2, 2)

def test_ndarraydict_from_vector_args(setup_vector_args):
    vector_args = dict_to_vector_args(setup_vector_args.data)
    shapes = shapes_from_ndarray_dict(setup_vector_args.data)
    npdict = ndarraydict_from_vector_args(vector_args, shapes)
    assert np.all(setup_vector_args.data['i'] == npdict['i'])
    assert np.all(setup_vector_args.data['f'] == npdict['f'])
    assert np.all(setup_vector_args.data['v2'] == npdict['v2'])
