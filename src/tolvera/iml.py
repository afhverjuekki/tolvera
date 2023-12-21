"""IMLBase and IMLDict classes for Tölvera.

Every Tölvera instance has an IMLDict, which is a dictionary of IML instances.
The IMLDict is accessible via the 'iml' attribute of a Tölvera instance, and can
be used to create and access IML instances.

Each IML instance has a type, which is one of the following:
- `vec2vec`: Vector to vector mapping.
- `vec2fun`: Vector to function mapping.
- `vec2osc`: Vector to OSC mapping.
- `fun2vec`: Function to vector mapping.
- `fun2fun`: Function to function mapping.
- `fun2osc`: Function to OSC mapping.
- `osc2vec`: OSC to vector mapping.
- `osc2fun`: OSC to function mapping.
- `osc2osc`: OSC to OSC mapping.
"""

import inspect
from typing import Any

import numpy as np
import torch
from anguilla import IML as iiIML

from .osc.update import Updater
from .utils import *

__all__ = [
    "IMLBase",
    "IMLVec2Vec",
    "IMLVec2Fun",
    "IMLVec2OSC",
    "IMLFun2Vec",
    "IMLFun2Fun",
    "IMLFun2OSC",
    "IMLOSC2Vec",
    "IMLOSC2OSC",
    "IMLOSC2Fun",
    "IMLDict",
    "IML_TYPES",
]

IML_TYPES = [
    "vec2vec",
    "vec2fun",
    "vec2osc",
    "fun2vec",
    "fun2fun",
    "fun2osc",
    "osc2vec",
    "osc2osc",
    "osc2fun",
]

RAND_METHODS = [
    "rand",
    "uniform",
    "normal",
    "exponential",
    "cauchy",
    "lognormal",
    "sigmoid",
    "beta",
]


def rand_select(method="rand"):
    """Select randomisation method.
    
    Args:
        method (str, optional): Randomisation method. Defaults to "rand".
    
    Raises:
        ValueError: Invalid method.
        
    Returns:
        callable: Randomisation method.
    """
    match method:
        case "rand":
            return rand_n
        case "uniform":
            return rand_uniform
        case "normal":
            return rand_normal
        case "exponential":
            return rand_exponential
        case "cauchy":
            return rand_cauchy
        case "lognormal":
            return rand_lognormal
        case "sigmoid":
            return rand_sigmoid
        case "beta":
            return rand_beta
        case _:
            raise ValueError(
                f"[tolvera._iml.rand_select] Invalid method '{method}'. Valid methods: {RAND_METHODS}."
            )


class IMLDict(dotdict):
    """IML mapping dict

    Similarly to StateDict, this class inherits from dotdict to enable instantiation
    via assignment.

    Example:
        tv = Tolvera(**kwargs)

        tv.iml.flock_p2flock_s = {
            'type': 'vec2vec', 
            'size': (tv.s.flock_p.size, tv.s.flock_s.size), 
            'randomise': True,
            'config': {'interp': 'Ripple'},
            'map_kw': {'k': 10, 'ripple_depth': 5, 'ripple': 5}
        }
    """

    def __init__(self, tolvera) -> None:
        """Initialise IMLDict

        Args:
            tolvera (Tolvera): Tölvera instance.
        """
        self.tv = tolvera
        self.i = {}  # input vectors dict
        self.o = {}  # output vectors dict

    def set(self, name, kwargs: dict) -> Any:
        """Set IML instance.

        Args:
            name (str): Name of IML instance.
            kwargs (dict): IML instance kwargs.

        Raises:
            ValueError: Cannot replace 'tv' IML instance.
            ValueError: Cannot replace 'i' IML instance.
            ValueError: Cannot replace 'o' IML instance.
            NotImplementedError: set() with tuple not implemented yet.
            TypeError: set() requires dict|tuple, not _type_.
            Exception: Other exceptions.

        Returns:
            Any: IML instance.
        """
        try:
            if name == "tv" and type(kwargs) is not dict and type(kwargs) is not tuple:
                if name in self:
                    raise ValueError(
                        f"[tolvera._iml.IMLDict] '{name}' cannot be replaced."
                    )
                self[name] = kwargs
            elif name == "i" or name == "o":
                if type(kwargs) is not dict:
                    raise ValueError(
                        f"[tolvera._iml.IMLDict] '{name}' is a reserved dict."
                    )
                self[name] = kwargs
            elif type(kwargs) is dict:
                if "type" not in kwargs:
                    raise ValueError(
                        f"[tolvera._iml.IMLDict] IMLDict requires 'type' key."
                    )
                return self.add(name, kwargs["type"], **kwargs)
            elif type(kwargs) is tuple:
                # iml_type = kwargs[0] # TODO: which index is 'iml_type'?
                # return self.add(name, iml_type, *kwargs)
                raise NotImplementedError(
                    f"[tolvera._iml.IMLDict] set() with tuple not implemented yet."
                )
            else:
                raise TypeError(
                    f"[tolvera._iml.IMLDict] set() requires dict|tuple, not {type(kwargs)}"
                )
        except Exception as e:
            raise type(e)(f"[tolvera._iml.IMLDict] {e}") from e

    def __setattr__(self, __name: str, __value: Any) -> None:
        """Set IML instance.

        Args:
            __name (str): Name of IML instance.
            __value (Any): IML instance kwargs.
        """
        self.set(__name, __value)

    def add(self, name: str, iml_type: str, **kwargs) -> Any:
        """Add IML instance.

        Args:
            name (str): Name of IML instance.
            iml_type (str): IML type.

        Raises:
            ValueError: Invalid IML_TYPE.

        Returns:
            Any: IML instance.
        """
        # TODO: should ^ be kwargs and not **kwargs?
        match iml_type:
            case "vec2vec":
                ins = IMLVec2Vec(**kwargs)
            case "vec2fun":
                ins = IMLVec2Fun(**kwargs)
            case "vec2osc":
                ins = IMLVec2OSC(self.tv.osc.map, **kwargs)
            case "fun2vec":
                ins = IMLFun2Vec(**kwargs)
            case "fun2fun":
                ins = IMLFun2Fun(**kwargs)
            case "fun2osc":
                ins = IMLFun2OSC(self.tv.osc.map, **kwargs)
            case "osc2vec":
                ins = IMLOSC2Vec(self.tv.osc.map, **kwargs)
            case "osc2fun":
                ins = IMLOSC2Fun(self.tv.osc.map, **kwargs)
            case "osc2osc":
                ins = IMLOSC2OSC(self.tv.osc.map, **kwargs)
            case _:
                raise ValueError(
                    f"[tolvera._iml.IMLDict] Invalid IML_TYPE '{iml_type}'. Valid IML_TYPES: {IML_TYPES}."
                )
        self[name] = ins
        self.o[name] = None
        return ins

    def __call__(self, name=None, *args: Any, **kwargs: Any) -> Any:
        """Call IML instance or all IML instances.

        Args:
            name (str, optional): Name of IML instance to call. Defaults to None.

        Raises:
            ValueError: 'name' not in dict.

        Returns:
            Any: IML output or dict of IML outputs.
        """
        if name is not None:
            if name in self:
                # OSC updaters are handled by tv.osc.map (OSCMap)
                # TODO: Rethink this?
                if "OSC" not in type(self[name]).__name__:
                    return self[name](*args, **kwargs)
            else:
                raise ValueError(f"[tolvera._iml.IMLDict] '{name}' not in dict.")
        else:
            outvecs = {}
            for iml in self:
                if iml == "tv" or iml == "i" or iml == "o":
                    continue
                cls_name = type(self[iml]).__name__
                if "OSC" in cls_name:
                    continue
                if "Vec2" in cls_name:
                    if iml in self.i:
                        invec = self.i[iml]
                        outvecs[iml] = self[iml](invec, *args, **kwargs)
                else:
                    outvecs[iml] = self[iml](*args, **kwargs)
            self.i.clear()
            self.o.update(outvecs)
            return self.o


class IMLBase(iiIML):
    """IML mapping base class.
    
    This class inherits from anguilla.IML and adds some functionality.
    It is not intended to be used directly, but rather to be inherited from.
    See IMLVec2Vec, IMLVec2Fun, IMLVec2OSC, IMLFun2Vec, IMLFun2Fun, IMLFun2OSC,
    IMLOSC2Vec, IMLOSC2OSC, IMLOSC2Fun.

    The base class is initialised with a size tuple (input, output) and a config dict
    which is passed to anguilla.IML.

    It provides a randomise() method which adds random pairs to the mapping.
    It also provides methods to remove pairs (remove_oldest, remove_newest, remove_random).
    It also provides a lag() method which lags the mapped data.
    Finally, it provides an update() method which is called by the updater (see .osc.update).
    """
    def __init__(self, **kwargs) -> None:
        """Initialise IMLBase
    
        kwargs:
            size (tuple, required): (input, output) sizes.
            io (tuple, optional): (input, output) functions.
            config (dict, optional): {embed_input, embed_output, interpolate, index, verbose}.
            updater (cls, optional): See iipyper.osc.update (Updater, OSCUpdater, ...).
            update_rate (int, optional): Updater's update rate (defaults to 1).
            randomise (bool, optional): Randomise mapping on init (defaults to False).
            rand_pairs (int, optional): Number of random pairs to add (defaults to 32).
            rand_input_weight (Any, optional): Random input weight (defaults to None).
            rand_output_weight (Any, optional): Random output weight (defaults to None).
            rand_method (str, optional): rand_method type (see utils).
            rand_kw (dict, optional): Random kwargs to pass to rand_method (see utils).
            map_kw (dict, optional): kwargs to use in IML.map().
            infun_kw (dict, optional): kwargs to use in infun() (type 'Fun2*' only).
            outfun_kw (dict, optional): kwargs to use in outfun() (type '*2Fun' only).
            lag (bool, optional): Lag mapped data (defaults to False). Incompatible with '*2Fun' types.
            lag_coef (float, optional): Lag coefficient (defaults to 0.5 if `lag` is True).
        """
        assert "size" in kwargs, f"IMLBase requires 'size' kwarg."
        self.size = kwargs["size"]
        self.updater = kwargs.get(
            "updater", Updater(self.update, kwargs.get("update_rate", 1))
        )
        self.config = kwargs.get("config", {})
        if isinstance(self.size[0], tuple):
            self.config["embed_input"] = "ProjectAndSort"
        print(f"[tolvera._iml.IMLBase] Initialising IML with config: {self.config}")
        super().__init__(**self.config)
        self.data = dotdict()
        self.map_kw = kwargs.get("map_kw", {})
        self.infun_kw = kwargs.get("infun_kw", {})
        self.outfun_kw = kwargs.get("outfun_kw", {})
        if kwargs.get("randomise", False):
            self.init_randomise(**kwargs)
        self.lag = kwargs.get("lag", False)
        if self.lag:
            self.init_lag(**kwargs)

    def init_randomise(self, **kwargs):
        """Initialise randomise() method with kwargs

        kwargs: see __init__ kwargs.
        """
        self.rand_pairs = kwargs.get("rand_pairs", 32)
        self.rand_input_weight = kwargs.get("rand_input_weight", None)
        self.rand_output_weight = kwargs.get("rand_output_weight", None)
        self.rand_method = kwargs.get("rand_method", "rand")
        self.rand_kw = kwargs.get("rand_kw", {})
        self.randomise(
            self.rand_pairs,
            self.rand_input_weight,
            self.rand_output_weight,
            self.rand_method,
            **self.rand_kw,
        )

    def init_lag(self, **kwargs):
        """Initialise lag() method with kwargs
        
        kwargs: see __init__ kwargs.
        """
        self.lag_coef = kwargs.get("lag_coef", 0.5)
        self.lag = Lag(coef=self.lag_coef)
        print(
            f"[tolvera._iml.IMLBase] Lagging mapped data with coef {self.lag_coef}."
        )

    def randomise(
        self,
        times: int,
        input_weight=None,
        output_weight=None,
        method: str = "rand",
        **kwargs,
    ):
        """Randomise mapping.

        Args:
            times (int): Number of random pairs to add.
            input_weight (Any, optional): Weighting for the input vector. Defaults to None.
            output_weight (Any, optional): Weighting for the output vector. Defaults to None.
            method (str, optional): Randomisation method. Defaults to "rand".
        """
        self.rand = rand_select(method)
        while len(self.pairs) < times:
            self.add_random_pair(input_weight, output_weight, **kwargs)

    def set_random_method(self, method: str = "rand"):
        """Set random method.

        Args:
            method (str, optional): Randomisation method. Defaults to "rand".
        """
        self.rand = rand_select(method)

    def add_random_pair(self, input_weight=None, output_weight=None, **kwargs):
        """Add random pair to mapping.

        Args:
            input_weight (Any, optional): Weighting for the input vector. Defaults to None.
            output_weight (Any, optional): Weighting for the output vector. Defaults to None.
            **kwargs: see create_random_pair kwargs.
        """
        indata, outdata = self.create_random_pair(input_weight, output_weight, **kwargs)
        self.add(indata, outdata)

    def create_random_pair(self, input_weight=None, output_weight=None, **kwargs):
        """Create random pair.

        Args:
            input_weight (Any, optional): Weighting for the input vector. Defaults to None.
            output_weight (Any, optional): Weighting for the output vector. Defaults to None.
            **kwargs:
                rand_method (str, optional): Randomisation method. Defaults to "rand".
                rand_kw (dict, optional): Random kwargs to pass to rand_method (see utils).
            
        Raises:
            ValueError: Invalid input_weight type.
            ValueError: Invalid output_weight type.

        Returns:
            tuple: (input, output) vectors.
        """
        if self.rand == None and "rand_method" not in kwargs:
            print(f"[tolvera._iml.IMLBase] No 'rand' method set. Using 'rand'.")
            self.set_random_method()
        elif "rand_method" in kwargs:
            self.set_random_method(kwargs["rand_method"])
        if input_weight is None:
            input_weight = self.rand_input_weight
        if output_weight is None:
            output_weight = self.rand_output_weight
        indata = self.rand(self.size[0], **kwargs)
        outdata = self.rand(self.size[1], **kwargs)
        if input_weight is not None:
            if isinstance(input_weight, np.ndarray):
                indata *= torch.from_numpy(input_weight)
            elif isinstance(input_weight, (torch.Tensor, float, int)):
                indata *= input_weight
            elif isinstance(input_weight, list):
                indata *= torch.Tensor(input_weight)
            else:
                raise ValueError(
                    f"[tolvera._iml.IMLBase] Invalid input_weight type '{type(input_weight)}'."
                )
        if output_weight is not None:
            if isinstance(output_weight, np.ndarray):
                outdata *= torch.from_numpy(output_weight)
            elif isinstance(output_weight, (torch.Tensor, float, int)):
                outdata *= output_weight
            elif isinstance(output_weight, list):
                outdata *= torch.Tensor(output_weight)
            else:
                raise ValueError(
                    f"[tolvera._iml.IMLBase] Invalid output_weight type '{type(output_weight)}'."
                )
        return indata, outdata

    def remove_oldest(self, n: int = 1):
        """Remove oldest pair(s) from mapping.

        Args:
            n (int, optional): Number of pairs to remove. Defaults to 1.
        """
        if len(self.pairs) > n - 1:
            [self.remove(min(self.pairs.keys())) for _ in range(n)]

    def remove_newest(self, n: int = 1):
        """Remove newest pair(s) from mapping.

        Args:
            n (int, optional): Number of pairs to remove. Defaults to 1.
        """
        if len(self.pairs) > n - 1:
            [self.remove(max(self.pairs.keys())) for _ in range(n)]

    def remove_random(self, n: int = 1):
        """Remove random pair(s) from mapping.

        Args:
            n (int, optional): Number of pairs to remove. Defaults to 1.
        """
        if len(self.pairs) > n - 1:
            [self.remove(np.random.choice(list(self.pairs.keys()))) for _ in range(n)]

    def lag_mapped_data(self, lag_coef: float = 0.5):
        """Lag mapped data.

        Args:
            lag_coef (float, optional): Lag coefficient. Defaults to 0.5.
        """
        self.data.mapped = self.lag(self.data.mapped, lag_coef)

    def update(self, invec: list|torch.Tensor|np.ndarray) -> list|torch.Tensor|np.ndarray:
        """Update mapped data.

        Args:
            invec (list|torch.Tensor|np.ndarray): Input vector.

        Returns:
            list|torch.Tensor|np.ndarray: Mapped data.
        """
        if len(self.pairs) == 0:
            return None
        self.data.mapped = self.map(invec, **self.map_kw)
        if hasattr(self, "lag") and type(self.lag) is Lag:
            self.lag_mapped_data()
        return self.data.mapped

    def update_rate(self, rate: int = None):
        """Update rate getter/setter.

        Args:
            rate (int, optional): Update rate. Defaults to None.

        Returns:
            int: Update rate.
        """
        if rate is not None:
            self.updater.count = rate
        return self.updater.count

    def __call__(self, *args) -> Any:
        """Call updater with args.

        Args:
            *args: Updater args.

        Returns:
            Any: Mapped data.
        """
        return self.updater(*args)


class IMLVec2Vec(IMLBase):
    """IML vector to vector mapping

    Args:
        kwargs:
            see IMLBase kwargs.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


class IMLVec2Fun(IMLBase):
    """IML vector to function mapping

    Args:
        kwargs:
            io (tuple, required): (None, callable) output function.
            see IMLBase kwargs.
    """

    def __init__(self, **kwargs) -> None:
        assert "io" in kwargs, f"IMLVec2Fun requires 'io' kwarg."
        assert (
            kwargs["io"][0] is None
        ), f"IMLVec2Fun requires 'io' kwarg to be (None, outfun)."
        assert callable(
            kwargs["io"][1]
        ), f"IMLVec2Fun requires 'io' kwarg to be (None, outfun)."
        self.outfun = kwargs["io"][1]
        super().__init__(**kwargs)

    def update(self, invec):
        mapped = self.map(invec, **self.map_kw)
        self.data.mapped = self.outfun(mapped, **self.outfun_kw)
        return self.data.mapped


class IMLVec2OSC(IMLBase):
    """IML vector to OSC mapping

    Args:
        osc_map (OSCMap, required): OSCMap instance.
        kwargs:
            io (tuple, required): (None, str) output OSC route.
            see IMLBase kwargs.
    """

    def __init__(self, osc_map, **kwargs) -> None:
        self.osc_map = osc_map
        """
        self.osc_map.add_somethingsomething(self.update)
        kwargs['updater'] = OSCUpdater from self.osc_map
        does OSCUpdater have args, kwargs?
        """
        assert "io" in kwargs, f"IMLVec2OSC requires 'io' kwarg."
        assert (
            kwargs["io"][0] is None
        ), f"IMLVec2OSC requires 'io' kwarg to be (None, osc_route)."
        assert (
            type(kwargs["io"][1]) is str
        ), f"IMLVec2OSC requires 'io' kwarg to be (None, osc_route)."
        self.route = kwargs["io"][1]
        super().__init__(**kwargs)

    def update(self, invec):
        raise NotImplementedError(
            f"[tolvera._iml.IMLVec2OSC] update() not implemented."
        )


class IMLFun2Vec(IMLBase):
    """IML function to vector mapping

    Args:
        kwargs:
            io (tuple, required): (callable, None) input function.
            see IMLBase kwargs.
    """

    def __init__(self, **kwargs) -> None:
        assert "io" in kwargs, f"IMLFun2Vec requires 'io' kwarg."
        assert callable(
            kwargs["io"][0]
        ), f"IMLFun2Vec requires 'io' kwarg to be (infun, None)."
        assert (
            kwargs["io"][1] is None
        ), f"IMLFun2Vec requires 'io' kwarg to be (infun, None)."
        self.infun = kwargs["io"][0]
        self.infun_params = inspect.signature(self.infun).parameters
        super().__init__(**kwargs)

    def update(self):
        if len(self.infun_params) > 0:
            invec = self.infun(**self.infun_kw)
        else:
            invec = self.infun()
        self.data.mapped = self.map(invec, **self.map_kw)
        return self.data.mapped


class IMLFun2Fun(IMLBase):
    """IML function to function mapping

    Args:
        kwargs:
            io (tuple, required): (callable, callable) input and output functions.
            see IMLBase kwargs.
    """

    def __init__(self, **kwargs) -> None:
        assert "io" in kwargs, f"IMLFun2Fun requires 'io' kwarg."
        assert callable(
            kwargs["io"][0]
        ), f"IMLFun2Fun requires 'io' kwarg to be (infun, outfun)."
        assert callable(
            kwargs["io"][1]
        ), f"IMLFun2Fun requires 'io' kwarg to be (infun, outfun)."
        self.infun = kwargs["io"][0]
        self.infun_params = inspect.signature(self.infun).parameters
        self.outfun = kwargs["io"][1]
        self.outfun_params = inspect.signature(self.outfun).parameters
        super().__init__(**kwargs)

    def update(self):
        if len(self.infun_params) > 0:
            invec = self.infun(**self.infun_kw)
        else:
            invec = self.infun()
        mapped = self.map(invec, **self.map_kw)
        self.data.mapped = self.outfun(mapped, **self.outfun_kw)
        return self.data.mapped


class IMLFun2OSC(IMLBase):
    """IML function to OSC mapping

    Args:
        osc_map (OSCMap, required): OSCMap instance.
        kwargs:
            see IMLBase kwargs.
    """

    def __init__(self, osc_map, **kwargs) -> None:
        self.osc_map = osc_map
        super().__init__(**kwargs)

    def update(self, invec):
        raise NotImplementedError(
            f"[tolvera._iml.IMLFun2OSC] update() not implemented."
        )


class IMLOSC2Vec(IMLBase):
    """IML OSC to vector mapping

    Args:
        osc_map (OSCMap, required): OSCMap instance.
        kwargs:
            see IMLBase kwargs.
    """

    def __init__(self, osc_map, **kwargs) -> None:
        self.osc_map = osc_map
        super().__init__(**kwargs)

    def update(self, invec):
        raise NotImplementedError(
            f"[tolvera._iml.IMLOSC2Vec] update() not implemented."
        )


class IMLOSC2Fun(IMLBase):
    """IML OSC to function mapping

    Args:
        osc_map (OSCMap, required): OSCMap instance.
        kwargs:
            see IMLBase kwargs.
    """

    def __init__(self, osc_map, **kwargs) -> None:
        self.osc_map = osc_map
        super().__init__(**kwargs)

    def update(self, invec):
        raise NotImplementedError(
            f"[tolvera._iml.IMLOSC2Fun] update() not implemented."
        )


class IMLOSC2OSC(IMLBase):
    """IML OSC to OSC mapping

    Args:
        osc_map (OSCMap, required): OSCMap instance.
        kwargs:
            see IMLBase kwargs.
    """

    def __init__(self, osc_map, **kwargs) -> None:
        self.osc_map = osc_map
        super().__init__(**kwargs)

    def update(self, invec):
        """
        see iml.app.server.map?
        """
        raise NotImplementedError(
            f"[tolvera._iml.IMLOSC2OSC] update() not implemented."
        )
