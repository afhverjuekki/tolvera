from taichi import i32, f32
from taichi.math import vec2, vec3, vec4
import numpy as np
from collections import defaultdict
from typing import Any, Union, Callable
from iipyper import ndarray_from_json, ndarray_from_json

from .utils import flatten

np_vec2 = np.array([0.,0.], dtype=np.float32)
np_vec3 = np.array([0.,0.,0.], dtype=np.float32)
np_vec4 = np.array([0.,0.,0.,0.], dtype=np.float32)

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
        # print(type(item), item)
        if isinstance(item, str):
            k = item
        else:
            if k is None:
                print(f'ERROR: anguilla: bad OSC syntax in {a}')
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
                raise ValueError(f"Shape mismatch for key '{key}': expected {array_size} elements, got {len(values)}.")
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
        >>> state = NpNdarrayDict({
                'i':  (np.int32, 2, 10),
                'f':  (np.float32, 0., 1.),
                'v2': (np_vec2, 0., 1.),
                'v3': (np_vec3, 0., 1.),
                'v4': (np_vec4, 0., 1.),
            }, (2,2))
        >>> state.set_value('i', (0, 0), 5)
        >>> print(state.get_value('i', (0, 0)))
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
    
    def init(self, data_dict: dict[str, tuple[Any, Any, Any]], shape: tuple[int]) -> None:
        self.dict = {}
        self.data = {}
        for key, (dtype, min_val, max_val) in data_dict.items():
            dshape = self.shape
            length = 1
            if isinstance(dtype, np.ndarray):
                dshape = dshape + dtype.shape
                length = dtype.shape[0]
                dtype = np.float32
            self.dict[key] = {
                'dtype': dtype, 
                'min': min_val, 
                'max': max_val, 
                'length': length,
                'shape': dshape
            }
            self.data[key] = np.zeros(dshape, dtype=dtype)

    def set_slice_from_dict(self, slice_indices: tuple, slice_values: dict):
        for key, values in slice_values.items():
            if key not in self.data:
                raise KeyError(f"Key {key} not found in data")
            
            array_slice = self.data[key][slice_indices]
            if array_slice.shape != np.array(values).shape:
                raise ValueError(f"Shape {array_slice.shape} of values for key {key} does not match the shape of the slice {np.array(values).shape}")

            self.data[key][slice_indices] = np.array(values, dtype=self.dict[key]['dtype'])

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
            num_elements = np.prod(self.dict[key]['shape'][1:])
            print(f"[{key}] num_elements: {num_elements}")
            
            # Extract the slice from slice_values_list and reshape if necessary
            slice_shape = self.dict[key]['shape'][1:]
            slice = slice_values_list[list_index:list_index + num_elements]
            print(f"[{key}] slice_shape: {slice_shape}, slice: {slice}")
            
            # Check if the slice has the correct length
            if len(slice) != num_elements:
                raise ValueError(f"Slice length {len(slice)} for key {key} does not match the number of elements {num_elements}")
            
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
            raise ValueError(f"Extra values {slice_values_list[list_index:]} in slice_values_list {slice_values_list} that do not correspond to any array")

    def set_row_from_dict(self, row_index: int, row_values: dict):
        """
        Sets the values of a specified row for each array in self.data.

        :param row_index: Index of the row to modify.
        :param row_values: Dictionary of new values for the row, keyed by the dictionary keys.
        """
        for key, values in row_values.items():
            if key not in self.data:
                raise KeyError(f"Key {key} not found in data")
            
            if row_index >= self.dict[key]['shape'][0]:
                raise IndexError("Row index out of bounds for key {key}")

            if len(values) != self.dict[key]['shape'][1]:
                raise ValueError(f"Length of values for key {key} does not match the number of columns")

            self.data[key][row_index, :] = np.array(values, dtype=self.dict[key]['dtype'])

    def set_row_from_list(self, row_index: int, row_list: list):
        """
        Sets the values of a specified row for each array in self.data from a flat list.

        :param row_index: Index of the row to modify.
        :param row_list: Flat list of new values for the rows, segmented by the arrays.
        """
        row_values = {}
        list_index = 0

        for key in self.data.keys():
            # Determine the total number of elements required for the current key
            num_elements = np.prod(self.dict[key]['shape'][1:])
            if list_index + num_elements > len(row_list):
                raise ValueError(f"Insufficient values provided for key {key}")

            # Extract the segment from row_list and reshape if necessary
            segment = row_list[list_index:list_index + num_elements]
            if self.dict[key]['dtype'] == np.float32 and len(self.dict[key]['shape']) > 2:
                # Reshape the segment for ndarrays with more than 2 dimensions
                segment = np.reshape(segment, self.dict[key]['shape'][1:])
            row_values[key] = segment
            list_index += num_elements

        if list_index != len(row_list):
            raise ValueError("Extra values in row_list that do not correspond to any array")

        self.set_row_from_dict(row_index, row_values)
        
    def set_attribute(self, key: str, index: tuple[int], value: Any) -> None:
        """
        Set an entire attribute to a given value.

        Args:
            key: The attribute key.
            index: The index at which to set the value.
            value: The value to set.

        Raises:
            KeyError: If the key is not found in the state.
            ValueError: If the value is out of the specified range for the key.

        """
        if key in self.data:
            if self.dict[key]['min'] <= value <= self.dict[key]['max']:
                self.data[key][index] = value
            else:
                raise ValueError(f"Value {value} is out of range for key {key}")
        else:
            raise KeyError(f"Key {key} not found in state")
    
    def get_attribute(self, key: str, index: tuple[int], attr: str) -> np.ndarray:
        """
        Get an entire attribute.

        Args:
            key: The attribute key.
            index: The index from which to get the value.
            attr: The attribute to get.

        Returns:
            The value at the specified index for the given key.

        Raises:
            KeyError: If the key is not found in the state.

        """
        if key in self.data:
            return self.data[key][index][attr]
        else:
            raise KeyError(f"Key {key} not found in state")

    def set_data(self, new_data: dict[str, np.ndarray]) -> None:
        """
        Set the data with a new data dictionary.

        Args:
            new_data: A dictionary representing the new data, where each key is an
                    attribute and the value is a numpy array.

        Raises:
            ValueError: If the new data is invalid (e.g., wrong shape, type, or value range).
        """
        if self.validate_new_state(new_data):
            self.data = new_data
        else:
            raise ValueError("Invalid data")

    def get_data(self) -> dict[str, np.ndarray]:
        """
        Get the entire current data as a dictionary.

        Returns:
            A dictionary where each key is an attribute and the value is a numpy array.
        """
        return self.data

    def validate_new_state(self, new_state: dict[str, np.ndarray]) -> bool:
        """
        Validate a new state dictionary to ensure it conforms to the expected structure and constraints.

        Args:
            new_state: A dictionary representing the new state to be validated.

        Returns:
            True if the new state is valid, False otherwise.
        """
        for key, array in new_state.items():
            # Check if the key exists in the current state
            if key not in self.data:
                print(f"Key {key} not found in current state")
                return False

            # Check if the shape of the array matches
            if array.shape != self.data[key].shape:
                print(f"Shape {array.shape} does not match expected shape {self.data[key].shape}")
                return False

            # # Check if the data type of the array matches
            # if array.dtype != self.data[key].dtype:
            #     return False

            # # Check if all values are within the specified range
            # if not (self.data[key]['min'] <= array).all() or not (array <= self.data[key]['max']).all():
            #     return False

        return True

    def randomise(self) -> None:
        """
        Randomize the entire state dictionary based on the datatype, minimum,
        and maximum values for each attribute.
        """
        for key in self.data:
            data_type = self.dict[key]['dtype']
            min_val   = self.dict[key]['min']
            max_val   = self.dict[key]['max']
            shape     = self.dict[key]['shape']

            if np.issubdtype(data_type, np.integer):
                self.data[key] = np.random.randint(min_val, max_val + 1, size=shape, dtype=data_type)
            elif np.issubdtype(data_type, np.floating):
                self.data[key] = np.random.uniform(min_val, max_val, size=shape).astype(data_type)
            # Add more conditions here if you have other data types

    def randomise_index(self, index: tuple[int]) -> None:
        """
        Randomize the values at a specific index for all attributes.

        Args:
            index: The index to randomize.
        """
        for key in self.data:
            data_type = self.dict[key]['dtype']
            min_val = self.dict[key]['min']
            max_val = self.dict[key]['max']

            if np.issubdtype(data_type, np.integer):
                self.data[key][index] = np.random.randint(min_val, max_val + 1, dtype=data_type)
            elif np.issubdtype(data_type, np.floating):
                self.data[key][index] = np.random.uniform(min_val, max_val).astype(data_type)

    def randomise_attribute_index(self, key: str, index: tuple[int]) -> None:
        """
        Randomize the value at a specific index for a given attribute.

        Args:
            key: The attribute key.
            index: The index to randomize.

        Raises:
            KeyError: If the key is not found in the state.
        """
        if key not in self.data:
            raise KeyError(f"Key {key} not found in state")

        data_type = self.dict[key]['dtype']
        min_val = self.dict[key]['min']
        max_val = self.dict[key]['max']

        if np.issubdtype(data_type, np.integer):
            self.data[key][index] = np.random.randint(min_val, max_val + 1, dtype=data_type)
        elif np.issubdtype(data_type, np.floating):
            self.data[key][index] = np.random.uniform(min_val, max_val).astype(data_type)

    def apply_function(self, key: str, func: Callable[[np.ndarray], np.ndarray]) -> None:
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

    def elementwise_operation(self, key1: str, key2: str, op: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> None:
        """
        Perform an element-wise operation between two arrays in the dictionary.

        Args:
            key1: The first attribute key.
            key2: The second attribute key.
            op: A function to perform element-wise operation.

        Raises:
            KeyError: If either key is not found.
        """
        if key1 not in self.data or key2 not in self.data:
            raise KeyError(f"One of the keys {key1}, {key2} not found")

        array1 = self.data[key1]
        array2 = self.data[key2]
        self.data[key1] = op(array1, array2)

    def broadcast_operation(self, key: str, other: Union[np.ndarray, 'NpNdarrayDict'], op: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> None:
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
            raise ValueError("The 'other' parameter must be either a NumPy ndarray or NpNdarrayDict")

        result = op(self.data[key], other_array)

        # Check if the result is within the allowed min-max range
        if np.any(result < self.dict[key]['min']) or np.any(result > self.dict[key]['max']):
            raise ValueError("Operation result violates min-max constraints")

        self.data[key] = result
