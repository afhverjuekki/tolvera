'''
TODO: Add assertions
'''

import taichi as ti
from tolvera.utils import ti_serialize, ti_deserialize

import pytest

@pytest.fixture
def setup():
    ti.init()

def test_structfield():
    @ti.dataclass
    class Sphere:
        center: ti.math.vec3
        radius: ti.f32
    f = Sphere.field(shape=(1,))
    print('StructField', type(f))
    serialized = ti_serialize(f)
    print('StructField', serialized)
    ti_deserialize(f, serialized)

def test_scalarfield():
    f = ti.field(ti.f32, shape=())
    print('ScalarField', type(f))
    serialized = ti_serialize(f)
    print('ScalarField', serialized)
    ti_deserialize(f, serialized)

def test_matrixfield():
    f = ti.Vector.field(n=2, dtype=float, shape=(3, 3))
    print('MatrixField', type(f))
    serialized = ti_serialize(f)
    print('MatrixField', serialized)
    ti_deserialize(f, serialized)

def test_vectorndarray():
    f = ti.ndarray(dtype=ti.math.vec3, shape=(4, 4))
    print('VectorNdarray', type(f))
    serialized = ti_serialize(f)
    print('VectorNdarray', serialized)
    ti_deserialize(f, serialized)

def test_scalarndarray():
    f = ti.ndarray(dtype=ti.f32, shape=(4, 4))
    print('ScalarNdarray', type(f))
    serialized = ti_serialize(f)
    print('ScalarNdarray', serialized)
    ti_deserialize(f, serialized)
