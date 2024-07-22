"""Module for working with dictionary of NumPy ndarrays.

Primaril used by State.
"""

from collections import defaultdict
from typing import Any, Callable, Union

import numpy as np
from iipyper import ndarray_from_json, ndarray_from_repr, ndarray_to_json
from taichi import f32, i32
from taichi.math import vec2, vec3, vec4

from .utils import create_safe_slice, flatten

np_vec2 = np.array([0.0, 0.0], dtype=np.float32)
np_vec3 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
np_vec4 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

TiNpTypeMap = {
    i32: np.int32,
    f32: np.float32,
    vec2: np_vec2,
    vec3: np_vec3,
    vec4: np_vec4,
}


def dict_from_vector_args(a: list, scalars=None):
    """Convert a list of arguments to a dictionary.

    Args:
    - a: A list of arguments.
    - scalars: A list of keys that should be unwrapped from lists.

    Returns:
    - A dictionary of keyword arguments.
    """
    a = list(a)
    kw = defaultdict(list)
    k = None
    while len(a):
        item = a.pop(0)
        if isinstance(item, str):
            k = item
        else:
            if k is None:
                print(f"ERROR: bad syntax in {a}")
            kw[k].append(item)
    # unwrap scalars
    for item in scalars or []:
        if item in kw:
            kw[item] = kw[item][0]
    return kw


def dict_to_vector_args(kw):
    """Convert a dictionary to a list of arguments.

    This function takes a dictionary and returns a list of arguments.

    Args:
    - kw: A dictionary of keyword arguments.

    Returns:
    - A list of arguments.
    """
    args = []
    for key, value in kw.items():
        args.append(key)
        if isinstance(value, (list, np.ndarray)):
            # If it's a numpy array (regardless of its shape), flatten it and extend the list
            if isinstance(value, np.ndarray):
                value = value.flatten()
            args.extend(value)
        else:
            # Append the scalar value associated with the key
            args.append(value)
    return args


def ndarraydict_from_vector_args(lst, shapes):
    """Convert a list to a dictionary where each list is turned into a numpy array.

    This function takes a list in the format output by `dict_from_vector_args` and converts it
    into a dictionary. Each key's list of values is converted into a numpy array with a
    specified shape.

    Args:
    - lst: The list to be converted.
    - shapes: A dictionary where keys correspond to the keys in the original list and
              values are tuples representing the desired shape of the numpy array.

    Returns:
    - A dictionary with keys mapped to numpy arrays.
    """

    def flatten(lst):
        """Flatten a nested list or return a non-nested list as is."""
        if all(isinstance(el, list) for el in lst):
            # Flatten only if all elements are lists
            return [item for sublist in lst for item in sublist]
        return lst

    kw = defaultdict(list)
    k = None
    for item in lst:
        if isinstance(item, str):
            k = item
        else:
            kw[k].append(item)

    for key, shape in shapes.items():
        if key in kw:
            values = flatten(kw[key])
            array_size = np.prod(shape)
            if len(values) != array_size:
                raise ValueError(
                    f"Shape mismatch for key '{key}': expected {array_size} elements, got {len(values)}."
                )
            kw[key] = np.array(values).reshape(shape)

    return dict(kw)


def shapes_from_ndarray_dict(ndarray_dict):
    """Return a dictionary of shapes given a dictionary of numpy ndarrays.

    This function takes a dictionary where values are numpy ndarrays and returns
    a new dictionary with the same keys, where each value is the shape of the ndarray.

    Args:
    - ndarray_dict: A dictionary where values are numpy ndarrays.

    Returns:
    - A dictionary where each key maps to the shape of the corresponding ndarray.
    """
    shapes = {}
    for key, array in ndarray_dict.items():
        shapes[key] = array.shape
    return shapes


class NpNdarrayDict:
    """
    A class that encapsulates a dictionary of NumPy ndarrays, each associated with a specific data type and a defined min-max range.
    It provides a structured and efficient way to manage and manipulate multidimensional arrays with constraints on their values.

    Attributes:
        data (Dict[str, Dict[str, Union[np.ndarray, Any]]]): A dictionary where each key represents an attribute,
        and the value is another dictionary with keys 'array', 'min', and 'max', representing the ndarray,
        its minimum value, and its maximum value, respectively.
        shape (Tuple[int, int]): The shape of the ndarrays, which is consistent across all attributes.

    Example:
        state = NpNdarrayDict({
            'i':  (np.int32, 2, 10),
            'f':  (np.float32, 0., 1.),
            'v2': (np_vec2, 0., 1.),
            'v3': (np_vec3, 0., 1.),
            'v4': (np_vec4, 0., 1.),
        }, (2,2))
        state.set_value('i', (0, 0), 5)
        print(state.get_value('i', (0, 0)))
        5
    """

    def __init__(self, data_dict: dict[str, tuple[Any, Any, Any]], shape: tuple[int]):
        """
        Initialize the State class.

        Args:
            data_dict: A dictionary where keys are attribute names and values are tuples
                       of (dtype, min_value, max_value).
            shape: The shape of the numpy arrays for each attribute.

        """
        self.shape = shape
        self.init(data_dict, shape)

    def init(
        self, data_dict: dict[str, tuple[Any, Any, Any]], shape: tuple[int]
    ) -> None:
        self.dict = {}
        self.data = {}
        self.size = 0
        for key, (dtype, min_val, max_val) in data_dict.items():
            dshape = self.shape
            length = 1
            # handle np_vec2, np_vec3, np_vec4
            if isinstance(dtype, np.ndarray):
                dshape = dshape + dtype.shape
                length = dtype.shape[0]
                dtype = np.float32
            self.dict[key] = {
                "dtype": dtype,
                "min": min_val,
                "max": max_val,
                "length": length,
                "shape": dshape,
                "ndims": len(dshape),
            }
            self.data[key] = np.zeros(dshape, dtype=dtype)
            size = self.data[key].size
            self.dict[key]["size"] = size
            self.size += size

    """
    to|from vec | list (iml)
    """

    def from_vec(self, vec: list):
        vec_start = 0
        for key in self.data.keys():
            attr_vec_size = self.dict[key]["size"]
            attr_vec = vec[vec_start : vec_start + attr_vec_size]
            self.attr_from_vec(key, attr_vec)
            vec_start += attr_vec_size

    def to_vec(self) -> list:
        vec = []
        for key in self.data.keys():
            vec += self.attr_to_vec(key).tolist()
        return vec

    def attr_from_vec(self, attr: str, vec: list):
        if attr not in self.data:
            raise KeyError(f"Key {attr} not in {self.data.keys()}")
        attr_shape, attr_dtype = self.dict[attr]["shape"], self.dict[attr]["dtype"]
        if len(vec) != np.prod(attr_shape):
            raise ValueError(
                f"Length of vec {len(vec)} does not match the shape of {attr} {attr_shape}"
            )
        nparr = np.array(vec, dtype=attr_dtype)
        if len(attr_shape) > 1:
            nparr = np.reshape(nparr, attr_shape)
        try:
            self.data[attr] = nparr
        except ValueError as e:
            print(f"ValueError occurred while setting {attr}: {e}")
            raise

    def attr_to_vec(self, attr: str) -> list:
        if attr not in self.data:
            raise KeyError(f"Key {attr} not in {self.data.keys()}")
        vec = self.data[attr].flatten()
        return vec

    def slice_from_vec(
        self, slice_args: Union[int, tuple[int, ...], slice], slice_vec: list
    ):
        # TODO: unique slice obj needed per key...
        # slice_obj = create_safe_slice(slice_args)
        raise NotImplementedError(f"slice_from_vec()")

    def slice_to_vec(self, slice_args: Union[int, tuple[int, ...], slice]) -> list:
        # TODO: unique slice obj needed per key...
        # vec = []
        # for key in self.data.keys():
        #     slice_obj = create_safe_slice(slice_args)
        #     vec += self.attr_slice_to_vec(key, slice_obj)
        # return vec
        raise NotImplementedError(f"slice_from_vec()")

    def attr_slice_from_vec(
        self, attr: str, slice_args: Union[int, tuple[int, ...], slice], slice_vec: list
    ):
        if attr not in self.data:
            raise KeyError(f"Key {attr} not in {self.data.keys()}")
        slice_obj = create_safe_slice(slice_args)
        attr_shape, attr_dtype = self.dict[attr]["shape"], self.dict[attr]["dtype"]
        nparr = np.array(slice_vec, dtype=attr_dtype)
        if len(attr_shape) > 1:
            nparr = np.reshape(nparr, attr_shape)
        try:
            self.data[attr][slice_obj] = nparr
        except ValueError as e:
            print(f"ValueError occurred while setting slice: {e}")
            raise

    def attr_slice_to_vec(
        self, attr: str, slice_args: Union[int, tuple[int, ...], slice]
    ) -> list:
        if attr not in self.data:
            raise KeyError(f"Key {attr} not in {self.data.keys()}")
        slice_obj = create_safe_slice(slice_args)
        vec = self.data[attr][slice_obj].flatten()
        return vec

    """
    vec slice helpers
    """

    def get_slice_size(self, slice_args: Union[int, tuple[int, ...], slice]) -> int:
        slice_obj = create_safe_slice(slice_args)
        return np.sum([self.data[key][slice_obj].size for key in self.data.keys()])

    def get_attr_slice_size(
        self, attr: str, slice_args: Union[int, tuple[int, ...], slice]
    ) -> int:
        if attr not in self.data:
            raise KeyError(f"Key {attr} not in {self.data.keys()}")
        slice_obj = create_safe_slice(slice_args)
        return self.data[attr][slice_obj].size

    """
    to|from vec_args (simple osc)
    """

    """
    to|from ndarray | ndarraydict (serialised formats, complex osc)
    """

    """
    ...
    """

    def set_slice_from_dict(self, slice_indices: tuple, slice_values: dict):
        for key, values in slice_values.items():
            if key not in self.data:
                raise KeyError(f"Key {key} not found in data")

            array_slice = self.data[key][slice_indices]
            if array_slice.shape != np.array(values).shape:
                raise ValueError(
                    f"Shape {array_slice.shape} of values for key {key} does not match the shape of the slice {np.array(values).shape}"
                )

            self.data[key][slice_indices] = np.array(
                values, dtype=self.dict[key]["dtype"]
            )

    # def list_to_dict(self, _list: list) -> dict:
    #     """
    #     Convert a flat list to a dictionary.

    #     :param _list: The flat list to convert.
    #     :return: A dictionary that matches self.dict.
    #     """
    #     pass

    # def list_len_to_dict_shape(self, _list: list) -> dict:
    #     """
    #     Convert a flat list to a dictionary of shapes.

    #     :param _list: The flat list to convert.
    #     :return: shape of the dictionary of _list based on self.shape.
    #     """
    #     list_len = len(_list)
    #     dict_shape = ()
    #     for key in self.data.keys():
    #         dict_shape += self.dict[key]['shape'][1:]
    #     dict_len = np.prod(dict_shape)
    #     if list_len != dict_len:
    #         raise ValueError(f"Length of list {_list} does not match the length of the dictionary {dict_len}")
    #     return dict_shape

    def set_slice_from_list(self, slice_indices: tuple, slice_values_list: list):
        list_index = 0

        for key in self.data.keys():
            # Determine the total number of elements required for the current key
            num_elements = np.prod(self.dict[key]["shape"][1:])
            print(f"[{key}] num_elements: {num_elements}")

            # Extract the slice from slice_values_list and reshape if necessary
            slice_shape = self.dict[key]["shape"][1:]
            slice = slice_values_list[list_index : list_index + num_elements]
            print(f"[{key}] slice_shape: {slice_shape}, slice: {slice}")

            # Check if the slice has the correct length
            if len(slice) != num_elements:
                raise ValueError(
                    f"Slice length {len(slice)} for key {key} does not match the number of elements {num_elements}"
                )

            # Reshape the slice for ndarrays with more than 2 dimensions
            if len(slice_shape) > 1:
                slice = np.reshape(slice, slice_shape)
                print(f"[{key}] (reshaping) slice_shape: {slice_shape}, slice: {slice}")

            # Assign the slice to the corresponding key
            self.data[key][slice_indices] = slice

            list_index += num_elements
            print(f"[{key}] list_index: {list_index}, num_elements: {num_elements}")

        print(f"data: {self.data}")

        # Check if there are extra values in slice_values_list
        if list_index != len(slice_values_list):
            raise ValueError(
                f"Extra values {slice_values_list[list_index:]} in slice_values_list {slice_values_list} that do not correspond to any array"
            )

    def set_data(self, new_data: dict[str, np.ndarray]) -> None:
        """
        Set the data with a new data dictionary.

        Args:
            new_data: A dictionary representing the new data, where each key is an
                    attribute and the value is a numpy array.

        Raises:
            ValueError: If the new data is invalid (e.g., wrong shape, type, or value range).
        """
        try:
            self.data = new_data
        except ValueError as e:
            print(f"ValueError occurred while setting data: {e}")
            raise

    def get_data(self) -> dict[str, np.ndarray]:
        """
        Get the entire current data as a dictionary.

        Returns:
            A dictionary where each key is an attribute and the value is a numpy array.
        """
        return self.data

    def validate(self, new_state: dict[str, np.ndarray]) -> bool:
        raise NotImplementedError("validate() not implemented")

    def randomise(self) -> None:
        """
        Randomize the entire state dictionary based on the datatype, minimum,
        and maximum values for each attribute.
        """
        for key in self.data:
            data_type = self.dict[key]["dtype"]
            min_val = self.dict[key]["min"]
            max_val = self.dict[key]["max"]
            shape = self.dict[key]["shape"]

            if np.issubdtype(data_type, np.integer):
                self.data[key] = np.random.randint(
                    min_val, max_val + 1, size=shape, dtype=data_type
                )
            elif np.issubdtype(data_type, np.floating):
                self.data[key] = np.random.uniform(min_val, max_val, size=shape).astype(
                    data_type
                )
            # Add more conditions here if you have other data types

    def randomise_attr(self, key: str) -> None:
        """
        Randomize a specific attribute in the state dictionary based on its datatype,
        minimum, and maximum values.

        Args:
            key: The attribute key to randomize.
        """
        data_type = self.dict[key]["dtype"]
        min_val = self.dict[key]["min"]
        max_val = self.dict[key]["max"]
        shape = self.dict[key]["shape"]

        if np.issubdtype(data_type, np.integer):
            self.data[key] = np.random.randint(
                min_val, max_val + 1, size=shape, dtype=data_type
            )
        elif np.issubdtype(data_type, np.floating):
            self.data[key] = np.random.uniform(min_val, max_val, size=shape).astype(
                data_type
            )
        # Add more conditions here if you have other data types

    def attr_apply(self, key: str, func: Callable[[np.ndarray], np.ndarray]) -> None:
        """
        Apply a user-defined function to the array of a specified key.

        Args:
            key: The attribute key.
            func: A function that takes a numpy array and returns a numpy array.

        Raises:
            KeyError: If the key is not found.
        """
        if key not in self.data:
            raise KeyError(f"Key {key} not found")

        self.data[key] = func(self.data[key])

    def attr_broadcast(
        self,
        key: str,
        other: Union[np.ndarray, "NpNdarrayDict"],
        op: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> None:
        """
        Perform a broadcasting operation between the array of the specified key and another array or NpNdarrayDict.

        Args:
            key: The key of the array in the dictionary to operate on.
            other: The other array or NpNdarrayDict to use in the operation.
            op: A function to perform the operation. This should be a NumPy ufunc (like np.add, np.multiply).

        Raises:
            KeyError: If the key is not found in the dictionary.
            ValueError: If the operation cannot be broadcasted or if it violates the min-max constraints.
        """
        if key not in self.data:
            raise KeyError(f"Key {key} not found")

        if isinstance(other, NpNdarrayDict):
            if other.shape != self.shape:
                raise ValueError("Shapes of NpNdarrayDict objects do not match")
            other_array = other.data[key]  # Assuming we want to operate on the same key
        elif isinstance(other, np.ndarray):
            other_array = other
        else:
            raise ValueError(
                "The 'other' parameter must be either a NumPy ndarray or NpNdarrayDict"
            )

        result = op(self.data[key], other_array)

        # Check if the result is within the allowed min-max range
        if np.any(result < self.dict[key]["min"]) or np.any(
            result > self.dict[key]["max"]
        ):
            raise ValueError("Operation result violates min-max constraints")

        self.data[key] = result
