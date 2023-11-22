import numpy as np
from typing import Any, Union, Callable

np_vec2 = np.array([0.0, 0.0], dtype=np.float32)
np_vec3 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
np_vec4 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

class NpNdarrayDict:
    """
    A class that encapsulates a dictionary of NumPy ndarrays, each associated with a specific data type and a defined min-max range. 
    It provides a structured and efficient way to manage and manipulate multidimensional arrays with constraints on their values. 

    The class offers various methods to set and get values, rows, and columns for individual arrays or across the entire dictionary. 
    It also includes functionalities for randomizing array elements within their defined ranges and validating the integrity of the data structure. 

    This class is particularly useful in scenarios where multiple related datasets need to be maintained and manipulated in a synchronized manner, 
    such as in scientific computing, data analysis, and machine learning applications where data consistency and integrity are crucial.

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

    def set_value(self, key: str, index: tuple[int], value: Any) -> None:
        """
        Set a value at a specific index for a given attribute.

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

    def get_value(self, key: str, index: tuple[int]) -> Any:
        """
        Get a value at a specific index for a given attribute.

        Args:
            key: The attribute key.
            index: The index from which to get the value.

        Returns:
            The value at the specified index for the given key.

        Raises:
            KeyError: If the key is not found in the state.

        """
        if key in self.data:
            return self.data[key][index]
        else:
            raise KeyError(f"Key {key} not found in state")

    def set_row(self, key: str, row_index: int, row_values: np.ndarray) -> None:
        """
        Set an entire row for the specified attribute.

        Args:
            key: The attribute key in the state dictionary.
            row_index: The index of the row to be set.
            row_values: A numpy array of values to set in the row.

        Raises:
            ValueError: If the row length is invalid or the key is not found.
        """
        if key in self.data and len(row_values) == self.shape[1]:
            self.data[key][row_index, :] = row_values
        else:
            raise ValueError("Invalid row length or key")

    def get_row(self, key: str, row_index: int) -> np.ndarray:
        """
        Get an entire row for the specified attribute.

        Args:
            key: The attribute key in the state dictionary.
            row_index: The index of the row to be retrieved.

        Returns:
            A numpy array representing the row for the specified attribute.

        Raises:
            KeyError: If the key is not found in the state.
        """
        if key in self.data:
            return self.data[key][row_index, :]
        else:
            raise KeyError("Key not found")

    def set_col(self, key: str, col_index: int, col_values: np.ndarray) -> None:
        """
        Set an entire column for the specified attribute.

        Args:
            key: The attribute key in the state dictionary.
            col_index: The index of the column to be set.
            col_values: A numpy array of values to set in the column.

        Raises:
            ValueError: If the column length is invalid or the key is not found.
        """
        if key in self.data and len(col_values) == self.shape[0]:
            self.data[key][:, col_index] = col_values
        else:
            raise ValueError("Invalid column length or key")

    def get_col(self, key: str, col_index: int) -> np.ndarray:
        """
        Get an entire column for the specified attribute.

        Args:
            key: The attribute key in the state dictionary.
            col_index: The index of the column to be retrieved.

        Returns:
            A numpy array representing the column for the specified attribute.

        Raises:
            KeyError: If the key is not found in the state.
        """
        if key in self.data:
            return self.data[key][:, col_index]
        else:
            raise KeyError("Key not found")

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

    def set_row_for_attribute(self, key: str, row_index: int, row_values: np.ndarray) -> None:
        """
        Set an entire row for a specific attribute.

        Args:
            key: The attribute key.
            row_index: The index of the row to set.
            row_values: An array of values to set in the row.

        Raises:
            ValueError: If the row length is invalid or the key is not found.

        """
        if key in self.data and len(row_values) == self.shape[1]:
            self.data[key][row_index, :] = row_values
        else:
            raise ValueError("Invalid row length or key")

    def get_row_for_attribute(self, key: str, row_index: int) -> np.ndarray:
        """
        Get an entire row for a specific attribute.

        Args:
            key: The attribute key.
            row_index: The index of the row to get.

        Returns:
            An array representing the row for the given key.

        Raises:
            KeyError: If the key is not found in the state.

        """
        if key in self.data:
            return self.data[key][row_index, :]
        else:
            raise KeyError("Key not found")

    def set_col_for_attribute(self, key: str, col_index: int, col_values: np.ndarray) -> None:
        """
        Set an entire column for a specific attribute.

        Args:
            key: The attribute key.
            col_index: The index of the column to set.
            col_values: An array of values to set in the column.

        Raises:
            ValueError: If the column length is invalid or the key is not found.

        """
        if key in self.data and len(col_values) == self.shape[0]:
            self.data[key][:, col_index] = col_values
        else:
            raise ValueError("Invalid column length or key")

    def get_col_for_attribute(self, key: str, col_index: int) -> np.ndarray:
        """
        Get an entire column for a specific attribute.

        Args:
            key: The attribute key.
            col_index: The index of the column to get.

        Returns:
            An array representing the column for the given key.

        Raises:
            KeyError: If the key is not found in the state.

        """
        if key in self.data:
            return self.data[key][:, col_index]
        else:
            raise KeyError("Key not found")

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

    def randomise_index(self, index: tuple[int, int]) -> None:
        """
        Randomize the values at a specific index for all attributes.

        Args:
            index: The index to randomize (row_index, column_index).
        """
        for key in self.data:
            data_type = self.dict[key]['dtype']
            min_val = self.dict[key]['min']
            max_val = self.dict[key]['max']

            if np.issubdtype(data_type, np.integer):
                self.data[key][index] = np.random.randint(min_val, max_val + 1, dtype=data_type)
            elif np.issubdtype(data_type, np.floating):
                self.data[key][index] = np.random.uniform(min_val, max_val).astype(data_type)

    def randomise_row(self, row_index: int) -> None:
        """
        Randomize the values in a specific row for all attributes.

        Args:
            row_index: The index of the row to randomize.
        """
        for key in self.data:
            data_type = self.dict[key]['dtype']
            min_val = self.dict[key]['min']
            max_val = self.dict[key]['max']

            if np.issubdtype(data_type, np.integer):
                self.data[key][row_index, :] = np.random.randint(min_val, max_val + 1, size=self.shape[1], dtype=data_type)
            elif np.issubdtype(data_type, np.floating):
                self.data[key][row_index, :] = np.random.uniform(min_val, max_val, size=self.shape[1]).astype(data_type)

    def randomise_col(self, col_index: int) -> None:
        """
        Randomize the values in a specific column for all attributes.

        Args:
            col_index: The index of the column to randomize.
        """
        for key in self.data:
            data_type = self.dict[key]['dtype']
            min_val = self.dict[key]['min']
            max_val = self.dict[key]['max']

            if np.issubdtype(data_type, np.integer):
                self.data[key][:, col_index] = np.random.randint(min_val, max_val + 1, size=self.shape[0], dtype=data_type)
            elif np.issubdtype(data_type, np.floating):
                self.data[key][:, col_index] = np.random.uniform(min_val, max_val, size=self.shape[0]).astype(data_type)

    def randomise_attribute_index(self, key: str, index: tuple[int, int]) -> None:
        """
        Randomize the value at a specific index for a given attribute.

        Args:
            key: The attribute key.
            index: The index to randomize (row_index, column_index).

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

    def randomise_attribute_row(self, key: str, row_index: int) -> None:
        """
        Randomize the values in a specific row for a given attribute.

        Args:
            key: The attribute key.
            row_index: The index of the row to randomize.

        Raises:
            KeyError: If the key is not found in the state.
        """
        if key not in self.data:
            raise KeyError(f"Key {key} not found in state")

        data_type = self.dict[key]['dtype']
        min_val = self.dict[key]['min']
        max_val = self.dict[key]['max']

        if np.issubdtype(data_type, np.integer):
            self.data[key][row_index, :] = np.random.randint(min_val, max_val + 1, size=self.shape[1], dtype=data_type)
        elif np.issubdtype(data_type, np.floating):
            self.data[key][row_index, :] = np.random.uniform(min_val, max_val, size=self.shape[1]).astype(data_type)

    def randomise_attribute_col(self, key: str, col_index: int) -> None:
        """
        Randomize the values in a specific column for a given attribute.

        Args:
            key: The attribute key.
            col_index: The index of the column to randomize.

        Raises:
            KeyError: If the key is not found in the state.
        """
        if key not in self.data:
            raise KeyError(f"Key {key} not found in state")

        data_type = self.dict[key]['dtype']
        min_val = self.dict[key]['min']
        max_val = self.dict[key]['max']

        if np.issubdtype(data_type, np.integer):
            self.data[key][:, col_index] = np.random.randint(min_val, max_val + 1, size=self.shape[0], dtype=data_type)
        elif np.issubdtype(data_type, np.floating):
            self.data[key][:, col_index] = np.random.uniform(min_val, max_val, size=self.shape[0]).astype(data_type)

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
