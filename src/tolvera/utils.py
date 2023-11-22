'''
TODO: Save/load
'''

import os
from pathlib import Path
from typing import Any
import unicodedata
import numpy as np
import jsons
import base64
import time

import taichi as ti
from taichi.lang.field import ScalarField
from taichi._lib.core.taichi_python import DataType
from .npndarray_dict import np_vec2, np_vec3, np_vec4

TiNpTypeMap = {
    ti.i32: np.int32,
    ti.f32: np.float32,
    ti.math.vec2: np_vec2,
    ti.math.vec3: np_vec3,
    ti.math.vec4: np_vec4,
}

def remove_accents(input: str):
    nfkd_form = unicodedata.normalize('NFKD', input)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def clean_name(name: str):
    return remove_accents(name).strip().lower()

# @ti.data_oriented
class CONSTS:
    '''
    Dict of CONSTS that can be used in Taichi scope
    '''
    def __init__(self, dict: dict[str, (DataType, Any)]):
        self.struct = ti.types.struct(**{k: v[0] for k, v in dict.items()})
        self.consts = self.struct(**{k: v[1] for k, v in dict.items()})
    def __getattr__(self, name):
        try:
            return self.consts[name]
        except:
            raise AttributeError(f"CONSTS has no attribute {name}")
    def __getitem__(self, name):
        try:
            return self.consts[name]
        except:
            raise AttributeError(f"CONSTS has no attribute {name}")

def ndarray_b64_serialize(ndarray):
    return {
        "@type": "ndarray",
        "dtype": str(ndarray.dtype),
        "shape": ndarray.shape,
        "b64": base64.b64encode(ndarray.tobytes()).decode('utf-8')
    }

def ndarray_b64_deserialize(serialized):
    return np.frombuffer(base64.b64decode(serialized["b64"]), dtype=np.dtype(serialized["dtype"])).reshape(serialized["shape"])

def np_serialize(ndarray):
    return jsons.dumps(ndarray_b64_serialize(ndarray))

def np_deserialize(json_str):
    return ndarray_b64_deserialize(jsons.loads(json_str))

def ti_serialize(field):
    if isinstance(field, (ScalarField, ti.lang.struct.StructField, ti.lang.matrix.MatrixField, ti.lang.matrix.VectorNdarray, ti.lang._ndarray.ScalarNdarray)):
        ndarray = field.to_numpy()
        if isinstance(ndarray, dict): # For StructField where to_numpy() returns a dict
            serialized = jsons.dumps({k: ndarray_b64_serialize(v) for k, v in ndarray.items()})
        else: # For other fields
            serialized = jsons.dumps(ndarray_b64_serialize(ndarray))
        field.serialized = serialized
        return serialized
    else:
        raise TypeError(f"Unsupported field type for serialization: {type(field)}")

def ti_deserialize(field, json_str):
    if isinstance(field, (ScalarField, ti.lang.struct.StructField, ti.lang.matrix.MatrixField, ti.lang.matrix.VectorNdarray, ti.lang._ndarray.ScalarNdarray)):
        data = jsons.loads(json_str)
        if isinstance(field, ti.lang.struct.StructField): # For StructField
            field.from_numpy({k: ndarray_b64_deserialize(v) for k, v in data.items()})
        else: # For other fields
            field.from_numpy(ndarray_b64_deserialize(data))
        field.serialized = None
    else:
        raise TypeError(f"Unsupported field type for deserialization: {type(field)}")

def time_function(func, *args, **kwargs):
    """Time how long it takes to run a function and print the result
    """
    start = time.time()
    func(*args, **kwargs)
    end = time.time()
    print(f"[Tolvera.utils] {func.__name__}() ran in {end-start:.4f}s")
    return end-start

def validate_path(path: str) -> bool:
    """
    Validate a path using os.path and pathlib.

    Args:
        path (str): The path to be validated.

    Returns:
        bool: True if the path is valid, raises an exception otherwise.

    Raises:
        TypeError: If the input is not a string.
        FileNotFoundError: If the path does not exist.
        PermissionError: If the path is not accessible.
    """
    if not isinstance(path, str):
        raise TypeError(f"Expected a string for path, but received {type(path)}")

    path_obj = Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"The path {path} does not exist or is not a file")

    if not os.access(path, os.R_OK):
        raise PermissionError(f"The path {path} is not accessible")

    return True

def validate_json_path(path: str) -> bool:
    """
    Validate a JSON file path. It uses validate_path for initial validation.

    Args:
        path (str): The JSON file path to be validated.

    Returns:
        bool: True if the path is a valid JSON file path, raises an exception otherwise.

    Raises:
        ValueError: If the path does not end with '.json'.
    """
    # Using validate_path for basic path validation
    validate_path(path)

    if not path.endswith('.json'):
        raise ValueError("Path should end with '.json'")

    return True

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
