"""State and StateDict classes for Tölvera.

Every Tölvera instance has a StateDict, which is a dictionary of State instances.
The StateDict is accessible via the 's' attribute of a Tölvera instance, and can
be used to create and access states.

Each State instance has a Taichi struct field and a corresponding NpNdarrayDict,
which handles OSC accessors and endpoints.
"""

from typing import Any

import jsons
import numpy as np
import taichi as ti
from taichi._lib.core.taichi_python import DataType

from .npndarray_dict import NpNdarrayDict, TiNpTypeMap, np_vec2, np_vec3, np_vec4
from .utils import *


class StateDict(dotdict):
    """StateDict class for Tölvera.
    
    This class is a dictionary of State instances, and is accessible via the 's'
    attribute of a Tölvera instance.

    States can be created by assigning a dictionary or a tuple to a StateDict key.
    and can be used in Taichi scope and Python scope respectively.

    Example:
        tv = Tolvera(**kwargs)

        tv.s.mystate = {
            "state": {
                "id":  (ti.i32, 0, tv.pn - 1),
                "pos": (ti.math.vec2, -1.0, 1.0),
                "vel": (ti.math.vec2, -1.0, 1.0),
            }, 
            "shape": (tv.pn, 1), 
            "osc": "get", 
            "randomise": True
        }

        tv.s.mystate.field.pos[0] = 0.5
    """
    def __init__(self, tolvera) -> None:
        """Initialise a StateDict for Tölvera.
        
        Args:
            tolvera (Tolvera): Tolvera instance to which this StateDict belongs.
        """
        self.tv = tolvera
        self.size = 0

    def set(self, name, kwargs: Any) -> None:
        """Set a state in the StateDict.

        Args:
            name (str): Name of the state.
            kwargs (Any): State attributes.
        
        Raises:
            ValueError: If the state is already in the StateDict.
            Exception: If the state cannot be added.
        """
        if name in self and name != "size":
            raise ValueError(f"[tolvera.state.StateDict] '{name}' already in dict.")
        try:
            self.add(name, kwargs)
        except Exception as e:
            raise type(e)(f"[tolvera.state.StateDict] {e}") from e

    def add(self, name, kwargs: Any):
        """Add a state to the StateDict.

        Args:
            name (str): Name of the state.
            kwargs (Any): State attributes.

        Raises:
            TypeError: If kwargs is not a dict or tuple.
        """
        if name == "tv" and type(kwargs) is not dict and type(kwargs) is not tuple:
            self[name] = kwargs
        elif name == "size" and type(kwargs) is int:
            self[name] = kwargs
        elif type(kwargs) is dict:
            self[name] = State(self.tv, name=name, **kwargs)
            self.size += self[name].size
        elif type(kwargs) is tuple:
            self[name] = State(self.tv, name, *kwargs)
            self.size += self[name].size
        else:
            raise TypeError(
                f"[tolvera.state.StateDict] set() requires dict|tuple, not {type(kwargs)}"
            )

    def from_vec(self, states: list[str], vector: list[float]):
        """Copy data from a vector to states in the StateDict.

        Args:
            states (list[str]): List of state names.
            vector (list[float]): Vector of data to copy.

        Raises:
            Exception: If the vector is not the correct size.
        """
        sizes_sum = self.get_size(states)
        assert sizes_sum == len(
            vector
        ), f"sizes_sum={sizes_sum} != len(vector)={len(vector)}"
        vec_start = 0
        for state in states:
            s = self.tv.s[state]
            vec = vector[vec_start : vec_start + s.size]
            s.from_vec(vec)
            vec_start += s.size

    def get_size(self, states: str | list[str]) -> int:
        """Return the size of the states in the StateDict.

        Args:
            states (str | list[str]): State name or list of state names.

        Returns:
            int: Size of the states.
        """
        if isinstance(states, str):
            states = [states]
        return sum([self.tv.s[state].size for state in states])

    def __setattr__(self, __name: str, __value: Any) -> None:
        """Set a state in the StateDict.

        Args:
            __name (str): Name of the state.
            __value (Any): State attributes.
        """
        self.set(__name, __value)


@ti.data_oriented
class State:
    """State class for Tölvera.
    
    This class takes a name, dictionary of state attributes, and a shape, and
    creates a Taichi struct field and a corresponding dictionary of NumPy arrays 
    (NpNdarrayDict) for a state.

    The Taichi struct field can be used in Taichi scope, and the NpNdarrayDict
    can be used in Python scope, and the two are kept in sync by the from_nddict()
    and to_nddict() methods.

    The State class also handles OSC accessors for the state, which use the
    NpNdarrayDict to get and set data. A Tölvera instance is therefore required
    to initialise a State instance.

    State attributes are defined as a dictionary of attribute names and tuples of
    (Taichi type, min value, max value). The domain of the attribute is used when
    randomising the data in the state, and by OSCMap endpoints and client patches.

    The state is n-dimensional based on the shape argument, and the NpNdarrayDict
    provides methods for accessing the data in the state in n-dimensional slices.

    Example:
        ```py
        tv.s.flock_p = {
            "state": {
                "separate": (ti.math.vec2, 0.0, 1.0),
                "align": (ti.math.vec2, 0.0, 1.0),
                "cohere": (ti.math.vec2, 0.0, 1.0),
                "nearby": (ti.i32, 0, self.tv.p.n - 1),
            },
            "shape": self.tv.pn, # particle count
            "osc": ("get"),
            "randomise": False,
        }
        ```
    """
    def __init__(
        self,
        tolvera,
        name: str,
        state: dict[str, tuple[DataType, Any, Any]],
        shape: int | tuple[int] = None,
        osc: str | tuple = None,  # ('get', 'set', 'stream')
        randomise: bool = True,
        methods: dict[str, Any] = None,
    ):
        """Initialise a state for Tölvera.

        Args:
            tolvera (Tolvera): Tolvera instance to which this state belongs.
            name (str): Name of this state.
            state (dict[str, tuple[DataType, Any, Any]]): Dict of state attributes.
            shape (int | tuple[int], optional): Shape of the state. Defaults to 1.
            methods (dict[str, Any], optional): Flag for OSC via iipyper. Defaults to False.
        """
        self.tv = tolvera
        assert name is not None, "State must have a name."
        self.name = name
        shape = 1 if shape is None else shape
        self.setup_data(state, shape, randomise, methods)
        self.setup_osc(osc)

    def setup_data(
        self,
        dict: dict[str, tuple[DataType, Any, Any]],
        shape: int | tuple[int],
        randomise: bool = True,
        methods: dict[str, Any] = None,
    ):
        """Setup data structures and data for this state.

        Args:
            dict (dict[str, tuple[DataType, Any, Any]]): Dict of state attributes.
            shape (int | tuple[int]): Shape of the state.
            randomise (bool, optional): Flag to randomise the data on creation. Defaults to True.
            methods (dict[str, Any], optional): Dict of Taichi field struct methods. Defaults to None.
        """
        self.create_struct_field(dict, shape, methods)
        self.create_npndarray_dict()
        if randomise:
            self.randomise()

    def create_struct_field(
        self,
        dict: dict[str, tuple[DataType, Any, Any]],
        shape: int | tuple[int],
        methods: dict[str, Any] = None,
    ):
        """Create a Taichi struct field for this state.

        Args:
            dict (dict[str, tuple[DataType, Any, Any]]): Dict of state attributes.
            shape (int | tuple[int]): Shape of the state.
            methods (dict[str, Any], optional): Dict of Taichi field struct methods. Defaults to None.
        """
        self.dict = dict
        self.shape = (shape,) if isinstance(shape, int) else shape
        if methods is None:
            self.struct = ti.types.struct(**{k: v[0] for k, v in self.dict.items()})
        else:
            self.methods = methods if methods is not None else {}
            self.struct = ti.types.struct(
                **{k: v[0] for k, v in self.dict.items()}, methods=self.methods
            )
        self.field = self.struct.field(shape=self.shape)

    def create_npndarray_dict(self):
        """Create a NpNdarrayDict for this state.

        Raises:
            NotImplementedError: If no Numpy type is found for a Taichi type.
        """
        nddict = {}
        for k, v in self.dict.items():
            titype, min_val, max_val = v
            nptype = TiNpTypeMap.get(titype)
            if nptype is None:
                raise NotImplementedError(f"no nptype for {titype}")
            nddict[k] = (nptype, min_val, max_val)
        self.nddict = NpNdarrayDict(nddict, self.shape)
        self.size = self.nddict.size

    def randomise(self):
        """Randomise the data in this state."""
        self.nddict.randomise()
        self.from_nddict()

    def randomise_attr(self, attr: str):
        """Randomise an attribute in this state.

        Args:
            attr (str): Attribute name.
        """
        self.nddict.randomise_attr(attr)
        self.from_nddict()

    def setup_osc(self, osc: tuple|str = None):
        """Setup OSC for this state.

        Args:
            osc (tuple | str, optional): ("get", "set", "stream"). Defaults to None.
        """
        self.osc = osc is not None
        if not self.osc: return
        if isinstance(osc, str): osc = (osc,)
        self.osc_set = "set" in osc if self.osc else False
        self.osc_get = "get" in osc if self.osc else False
        self.osc_stream = "stream" in osc if self.osc else False
        self.setter_name = f"{self.tv.name_clean}_set_{self.name}"
        self.getter_name = f"{self.tv.name_clean}_get_{self.name}"
        self.stream_name = f"{self.tv.name_clean}_stream_{self.name}"
        if self.tv.osc is not False and self.osc:
            self.osc = self.tv.osc
            if self.osc_set: self.add_osc_setters()
            # if self.osc_get: self.add_osc_getters()
            # if self.osc_stream: self.add_osc_streams()

    def add_osc_setters(self):
        name = self.setter_name
        self.osc.map.receive_args_inline(name + "_randomise", self.randomise)

    def add_osc_getters(self):
        name = self.getter_name
        for k, v in self.dict.items():
            ranges = (int(v[0]), int(v[0]), int(v[1]))
            kwargs = {"i": ranges, "j": ranges, "attr": (k, k, k)}
            self.osc.map.receive_args_inline(f"{name}", self.osc_getter, **kwargs)

    # def osc_getter(self, i: int, j: int, attribute: str):
    #     ret = self.get((i, j), attribute)
    #     if ret is not None:
    #         route = self.osc.map.pascal_to_path(self.getter_name)  # +'/'+attribute
    #         self.osc.host.return_to_sender_by_name(
    #             (route, attribute, ret), self.osc.client_name
    #         )
    #     return ret

    # def add_osc_streams(self):
    #     # add send in broadcast mode
    #     raise NotImplementedError("add_osc_streams not implemented")

    def serialize(self) -> str:
        return ti_serialize(self.field)

    def deserialize(self, json_str: str):
        ti_deserialize(self.field, json_str)

    def save(self, path: str):
        # TODO: path validation, save to path, etc.
        json_str = self.serialize()
        raise NotImplementedError("save not implemented")

    def load(self, path: str):
        # TODO: path validation, file ext., etc.
        # TODO: data validation (pydantic?)
        json_str = jsons.load(path)
        self.deserialize(json_str)
        raise NotImplementedError("load not implemented")

    def from_nddict(self):
        """Copy data from NpNdarrayDict to Taichi field.
        
        Raises:
            Exception: If data cannot be copied.
        """
        try:
            data = self.nddict.get_data()
            self.field.from_numpy(data)
        except Exception as e:
            raise Exception(f"[tolvera.state.from_nddict] {e}") from e

    def to_nddict(self):
        """Copy data from Taichi field to NpNdarrayDict.

        Raises:
            Exception: If data cannot be copied.
        """
        try:
            data = self.field.to_numpy()
            self.nddict.set_data(data)
        except Exception as e:
            raise Exception(f"[tolvera.state.to_nddict] {e}") from e
    
    def set_from_nddict(self, data: dict):
        """Copy data from NumPy array dict to Taichi field.

        Args:
            data (dict): NumPy array dict to copy.

        Raises:
            Exception: If data cannot be copied.
        """
        try:
            self.field.from_numpy(data)
        except Exception as e:
            raise Exception(f"[tolvera.state.from_numpy] {e}") from e

    """
    npndarray_dict wrappers
    """

    def from_vec(self, vec: list):
        """Wrapper for NpNdarrayDict.from_vec()."""
        self.to_nddict()
        self.nddict.from_vec(vec)
        self.from_nddict()

    def to_vec(self) -> list:
        """Wrapper for NpNdarrayDict.to_vec()."""
        self.to_nddict()
        return self.nddict.to_vec()

    def attr_from_vec(self, attr: str, vec: list):
        """Wrapper for NpNdarrayDict.attr_from_vec()."""
        self.to_nddict()
        self.nddict.attr_from_vec(attr, vec)
        self.from_nddict()

    def attr_to_vec(self, attr: str) -> list:
        """Wrapper for NpNdarrayDict.attr_to_vec()."""
        self.to_nddict()
        return self.nddict.attr_to_vec(attr)

    def slice_from_vec(self, slice_args: list, slice_vec: list):
        """Wrapper for NpNdarrayDict.slice_from_vec()."""
        self.to_nddict()
        self.nddict.slice_from_vec(slice_args, slice_vec)
        self.from_nddict()

    def slice_to_vec(self, slice_args: list) -> list:
        """Wrapper for NpNdarrayDict.slice_to_vec()."""
        self.to_nddict()
        return self.nddict.slice_to_vec(slice_args)

    def attr_slice_from_vec(self, attr: str, slice_args: list, slice_vec: list):
        """Wrapper for NpNdarrayDict.attr_slice_from_vec()."""
        self.to_nddict()
        self.nddict.attr_slice_from_vec(attr, slice_args, slice_vec)
        self.from_nddict()

    def attr_slice_to_vec(self, attr: str, slice_args: list) -> list:
        """Wrapper for NpNdarrayDict.attr_slice_to_vec()."""
        self.to_nddict()
        return self.nddict.attr_slice_to_vec(attr, slice_args)

    def attr_size(self, attr: str) -> int:
        """Return the size of the attribute."""
        return self.nddict.data[attr].size
    
    """
    misc
    """

    def fill(self, value: ti.f32):
        """Fill the Taichi field with a value."""
        self.field.fill(value)

    @ti.func
    def __getitem__(self, index: ti.i32):
        """Return the Taichi field attribute.
        
        Args:
            index (ti.i32): Attribute index.
        """
        return self.field[index]

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """Return the Taichi field."""
        return self.field
