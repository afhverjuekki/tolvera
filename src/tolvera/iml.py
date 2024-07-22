"""
IML stands for Interactive Machine Learning. Tölvera wraps the 
[anguilla](https://intelligent-instruments-lab.github.io/anguilla/) package
to provide convenient ways for quickly creating mappings between vectors, functions
and OSC routes. 

Every Tölvera instance has an IMLDict, which is a dictionary of IML instances.
The IMLDict is accessible via the `iml` attribute of a Tölvera instance, and can
be used to create and access IML instances.

There are 9 IML types, which are listed below.

Example:
    Here we create a mapping based on states created by `tv.v.flock`,
    where the per-particle state `flock_p` is mapped to the species rule matrix `flock_s`.
    Since this is a `fun2fun` mapping (see IML Types below), we provide input and output 
    functions, and Tölvera updates the mapping automatically every `render()` call.
    ```py
    from tolvera import Tolvera, run

    def main(**kwargs):
        tv = Tolvera(**kwargs)

        tv.iml.flock_p2flock_s = {
            'size': (tv.s.flock_p.size, tv.s.flock_s.size), 
            'io': (tv.s.flock_p.to_vec, tv.s.flock_s.from_vec),
            'randomise': True,
        }
        
        @tv.render
        def _():
            tv.px.diffuse(0.99)
            tv.v.flock(tv.p)
            tv.px.particles(tv.p, tv.s.species, 'circle')
            return tv.px

    if __name__ == '__main__':
        run(main)
    ```

IML Types:
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

from iipyper.osc import OSC as iiOSC
from .osc.oscmap import OSCMap
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

# see anguilla.app.server.main
ANGUILLA_ROUTES = {
    'config': '/anguilla/config',
    'add': '/anguilla/add',
    'add_batch': '/anguilla/add_batch',
    'remove': '/anguilla/remove',
    'remove_near': '/anguilla/remove_near',
    'map': '/anguilla/map',
    'map_batch': '/anguilla/map_batch',
    'reset': '/anguilla/reset',
    'load': '/anguilla/load',
    'save': '/anguilla/save',
}

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
                f"[tolvera.iml.rand_select] Invalid method '{method}'. Valid methods: {RAND_METHODS}."
            )


class IMLDict(dotdict):
    """IML mapping dict

    Similarly to `StateDict`, this class inherits from `dotdict` to enable instantiation
    via assignment.
    """

    def __init__(self, context) -> None:
        """Initialise IMLDict

        Args:
            context (TolveraContext): TolveraContext instance.
        """
        self.ctx = context
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
            if name == "ctx" and type(kwargs) is not dict and type(kwargs) is not tuple:
                if name in self:
                    raise ValueError(
                        f"[tolvera.iml.IMLDict] '{name}' cannot be replaced."
                    )
                self[name] = kwargs
            elif name == "i" or name == "o":
                if type(kwargs) is not dict:
                    raise ValueError(
                        f"[tolvera.iml.IMLDict] '{name}' is a reserved dict."
                    )
                self[name] = kwargs
            elif type(kwargs) is dict:
                iml_type = self.infer_type(kwargs['io'])
                return self.add(name, iml_type, **kwargs)
            elif type(kwargs) is tuple:
                # iml_type = kwargs[0] # TODO: which index is 'iml_type'?
                # return self.add(name, iml_type, *kwargs)
                raise NotImplementedError(
                    f"[tolvera.iml.IMLDict] set() with tuple not implemented yet."
                )
            else:
                raise TypeError(
                    f"[tolvera.iml.IMLDict] set() requires dict|tuple, not {type(kwargs)}"
                )
        except Exception as e:
            raise type(e)(f"[tolvera.iml.IMLDict] {e}") from e

    def infer_type(self, io: tuple) -> str:
        """Infer IML type from kwargs.

        Args:
            io (tuple): IML input-output types.

        Raises:
            ValueError: Invalid IML types.

        Returns:
            str: IML type.
        """
        iml_type = None
        i, o = io[0], io[1]
        if type(i).__name__ == "method":
            on = type(o).__name__
            if   on == "method": iml_type = "fun2fun"
            elif o == list:      iml_type = "fun2vec"
            elif o == str:       iml_type = "fun2osc"
            else:
                raise ValueError(f"[tolvera.iml.IMLDict] Invalid types '{i}' & '{o}'.")
        elif i == list:
            on = type(o).__name__
            if   o == list:      iml_type = "vec2vec"
            elif on == "method": iml_type = "vec2fun"
            elif o == str:       iml_type = "vec2osc"
            else:
                raise ValueError(f"[tolvera.iml.IMLDict] Invalid types '{i}' & '{o}'.")
        elif i == str:
            on = type(o).__name__
            if   o == str:       iml_type = "osc2osc"
            elif on == "method": iml_type = "osc2fun"
            elif o == list:      iml_type = "osc2vec"
            else:
                raise ValueError(f"[tolvera.iml.IMLDict] Invalid types '{i}' & '{o}'.")
        return iml_type

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
                ins = IMLVec2OSC(self.ctx.osc.map, **kwargs)
            case "fun2vec":
                ins = IMLFun2Vec(**kwargs)
            case "fun2fun":
                ins = IMLFun2Fun(**kwargs)
            case "fun2osc":
                ins = IMLFun2OSC(self.ctx.osc.map, **kwargs)
            case "osc2vec":
                ins = IMLOSC2Vec(self.ctx.osc.map, self.o, name, **kwargs)
            case "osc2fun":
                ins = IMLOSC2Fun(self.ctx.osc.map, **kwargs)
            case "osc2osc":
                ins = IMLOSC2OSC(self.ctx.osc.map, self.ctx.osc, **kwargs)
            case _:
                raise ValueError(
                    f"[tolvera.iml.IMLDict] Invalid IML_TYPE '{iml_type}'. Valid IML_TYPES: {IML_TYPES}."
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
                raise ValueError(f"[tolvera.iml.IMLDict] '{name}' not in dict.")
        else:
            outvecs = {}
            for iml in self:
                if iml == "ctx" or iml == "i" or iml == "o":
                    continue
                cls_name = type(self[iml]).__name__
                if "Vec2OSC" in cls_name:
                    self[iml].invec = self.i[iml]
                elif "OSC" in cls_name:
                    # Fun2OSC, OSC2Fun, OSC2OSC and OSC2Vec 
                    # are handled by tv.osc.map (OSCMap)
                    continue
                elif "Vec2" in cls_name:
                    # Vec2Vec, Vec2Fun
                    if iml in self.i:
                        invec = self.i[iml]
                        outvecs[iml] = self[iml](invec, *args, **kwargs)
                else:
                    # Fun2Fun, Fun2Vec
                    outvecs[iml] = self[iml](*args, **kwargs)
            self.i.clear()
            self.o.update(outvecs)
            return self.o


class IMLBase(iiIML):
    """
    This class inherits from [anguilla](https://intelligent-instruments-lab.github.io/anguilla) 
    and adds some functionality. It is not intended to be used directly, but rather 
    to be inherited from.

    The base class is initialised with a size tuple (input, output) and a config dict
    which is passed to `anguilla.IML`.

    It provides a `randomise` method which adds random pairs to the mapping.
    It also provides methods to remove pairs (`remove_oldest`, `remove_newest`, `remove_random`).
    It also provides a `lag` method which lags the mapped data.
    Finally, it provides an `update` method which is called by the `updater` (see `.osc.update`).
    """
    def __init__(self, **kwargs) -> None:
        """Initialise IMLBase
    
        kwargs:
            size (tuple, required): (input, output) sizes.
            io (tuple, optional): (input, output) functions.
            config (dict, optional): {embed_input, embed_output, interpolate, index, verbose}.
            updater (cls, optional): See iipyper.osc.update (Updater, OSCSendUpdater, ...).
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
        print(f"[tolvera.iml.IMLBase] Initialising IML with config: {self.config}")
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
            f"[tolvera.iml.IMLBase] Lagging mapped data with coef {self.lag_coef}."
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
            **kwargs: see random_pair kwargs.
        """
        indata, outdata = self.random_pair(input_weight, output_weight, **kwargs)
        self.add(indata, outdata)

    def random_input(self, **kwargs) -> torch.Tensor:
        """Random input vector.

        Args:
            **kwargs: self.rand kwargs.

        Returns:
            torch.Tensor: Random input vector.
        """
        return self.rand(self.size[0], **kwargs)
    
    def random_output(self, **kwargs) -> torch.Tensor:
        """Random output vector.

        Args:
            **kwargs: self.rand kwargs

        Returns:
            torch.Tensor: Random output vector.
        """
        return self.rand(self.size[1], **kwargs)

    def random_pair(self, input_weight=None, output_weight=None, **kwargs):
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
            print(f"[tolvera.iml.IMLBase] No 'rand' method set. Using 'rand'.")
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
                    f"[tolvera.iml.IMLBase] Invalid input_weight type '{type(input_weight)}'."
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
                    f"[tolvera.iml.IMLBase] Invalid output_weight type '{type(output_weight)}'."
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
    """IML vector to vector mapping.

    Input vector is accessed via `tv.iml.i['name']`.
    Output vector is accessed via `tv.iml.o['name']`.

    Example:
        ```py
        tv.iml.flock_p2flock_s = {
            'io': (None, None),
            'size': (tv.s.flock_p.size, tv.s.flock_s.size)
        }

        def update():
            invec = tv.s.flock_p.to_vec()
            tv.iml.i = {'flock_p2flock_s': invec}
            flock_s_outvec = tv.iml.o['flock_p2flock_s']
            if flock_s_outvec is not None:
                tv.s.flock_s.from_vec(flock_s_outvec)
        ```

    Args:
        kwargs:
            see IMLBase kwargs.
    """

    def __init__(self, **kwargs) -> None:
        """Initialise IMLVec2Vec"""
        super().__init__(**kwargs)


class IMLVec2Fun(IMLBase):
    """IML vector to function mapping

    Example:
        ```py
        def update(outvec):
            print('outvec', outvec)

        tv.iml.flock_p2fun = {
            'size': (tv.s.flock_p.size, 8), 
            'io': (None, update),
        }
        ```
    """
    def __init__(self, **kwargs) -> None:
        """Initialise IMLVec2Fun
        
        Args:
            kwargs:
                io (tuple, required): (None, callable) output function.
                see IMLBase kwargs.
        """
        assert "io" in kwargs, f"IMLVec2Fun requires 'io=(None, callable)' kwarg."
        assert (
            kwargs["io"][0] is None
        ), f"IMLVec2Fun 'io[0]' not None, got {type(kwargs['io'][0])}."
        assert callable(
            kwargs["io"][1]
        ), f"IMLVec2Fun 'io[1]' not callable, got {type(kwargs['io'][1])}."
        self.outfun = kwargs["io"][1]
        super().__init__(**kwargs)

    def update(self, invec: list|torch.Tensor|np.ndarray) -> list|torch.Tensor|np.ndarray:
        """Update mapped data.

        Args:
            invec (list | torch.Tensor | np.ndarray): Input vector.

        Returns:
            list|torch.Tensor|np.ndarray: Mapped data.
        """
        mapped = self.map(invec, **self.map_kw)
        self.data.mapped = self.outfun(mapped, **self.outfun_kw)
        return self.data.mapped


class IMLVec2OSC(IMLBase):
    """IML vector to OSC mapping.

    Example:
        Sends the output vector to '/tolvera/flock'.

        ```py
        tv.iml.flock_p2osc = {
            'size': (tv.s.flock_p.size, 8), 
            'io': (None, 'tolvera_flock'),
        }
        ```
    """
    def __init__(self, osc_map: OSCMap, **kwargs) -> None:
        """Initialise IMLVec2OSC

        Args:
            osc_map (OSCMap, required): OSCMap instance.
            kwargs:
                io (tuple, required): (None, str) output OSC route.
                see IMLBase kwargs.
        """
        assert "io" in kwargs, f"IMLVec2OSC requires 'io=(None, str)' kwarg."
        assert (
            kwargs["io"][0] is None
        ), f"IMLVec2OSC 'io[0]' is not None, got {type(kwargs['io'][0])}."
        assert (
            type(kwargs["io"][1]) is str
        ), f"IMLVec2OSC 'io[1]' is not str, got {type(kwargs['io'][1])}."
        self.osc_map = osc_map
        self.out_osc_route = '/return'+ANGUILLA_ROUTES['map']
        self.osc_map.send_list_inline(self.out_osc_route, self.update, kwargs["size"][1], count=kwargs.get("update_rate", 10))
        kwargs["updater"] = self.osc_map.dict["send"][self.out_osc_route]['updater']
        super().__init__(**kwargs)

    def update(self) -> list|torch.Tensor|np.ndarray:
        """Update mapped data.

        Returns:
            list|torch.Tensor|np.ndarray: Mapped data.
        """
        if len(self.pairs) == 0:
            return None
        if self.invec is not None:
            self.data.mapped = self.map(self.invec, **self.map_kw)
            if hasattr(self, "lag") and type(self.lag) is Lag:
                self.lag_mapped_data()
            return self.data.mapped.tolist()
        else:
            return None


class IMLFun2Vec(IMLBase):
    """IML function to vector mapping.

    Output vector is accessed via `tv.iml.o['name']`.

    Example:
        ```py
        tv.iml.flock_p2vec = {
            'size': (tv.s.flock_p.size, 8), 
            'io': (tv.s.flock_p.to_vec, None),
        }
        # ...
        flock_s_outvec = tv.iml.o['flock_p2flock_s']
        ```
    """
    def __init__(self, **kwargs) -> None:
        """Initialise IMLFun2Vec

        Args:
            kwargs:
                io (tuple, required): (callable, None) input function.
                see IMLBase kwargs.
        """
        assert "io" in kwargs, f"IMLFun2Vec requires 'io=(callable, None)' kwarg."
        assert callable(
            kwargs["io"][0]
        ), f"IMLFun2Vec 'io[0]' not callable, got {type(kwargs['io'][0])}."
        assert (
            kwargs["io"][1] is None
        ), f"IMLFun2Vec 'io[1]' not None, got {type(kwargs['io'][1])}."
        self.infun = kwargs["io"][0]
        self.infun_params = inspect.signature(self.infun).parameters
        super().__init__(**kwargs)

    def update(self) -> list|torch.Tensor|np.ndarray:
        """Update mapped data.

        Returns:
            list|torch.Tensor|np.ndarray: Mapped data.
        """
        if len(self.infun_params) > 0:
            invec = self.infun(**self.infun_kw)
        else:
            invec = self.infun()
        self.data.mapped = self.map(invec, **self.map_kw)
        return self.data.mapped


class IMLFun2Fun(IMLBase):
    """IML function to function mapping.

    Example:
        ```py
        def infun():
            return [0,0,0,0]

        def outfun(vector):
            print('outvec', vector)

        tv.iml.test2test = {
            'size': (4, 8), 
            'io': (infun, outfun),
        }
        ```
    """
    def __init__(self, **kwargs) -> None:
        """Initialise IMLFun2Fun

        Args:
            kwargs:
                io (tuple, required): (callable, callable) input and output functions.
                see IMLBase kwargs.
        """
        assert "io" in kwargs, f"IMLFun2Fun requires 'io=(callable, callable)' kwarg."
        assert callable(
            kwargs["io"][0]
        ), f"IMLFun2Fun 'io[0]' not callable, got {type(kwargs['io'][0])}."
        assert callable(
            kwargs["io"][1]
        ), f"IMLFun2Fun 'io[1]' not callable, got {type(kwargs['io'][1])}."
        self.infun = kwargs["io"][0]
        self.infun_params = inspect.signature(self.infun).parameters
        self.outfun = kwargs["io"][1]
        self.outfun_params = inspect.signature(self.outfun).parameters
        super().__init__(**kwargs)

    def update(self) -> list|torch.Tensor|np.ndarray:
        """Update mapped data.

        Returns:
            list|torch.Tensor|np.ndarray: Mapped data.
        """
        if len(self.infun_params) > 0:
            invec = self.infun(**self.infun_kw)
        else:
            invec = self.infun()
        mapped = self.map(invec, **self.map_kw)
        self.data.mapped = self.outfun(mapped, **self.outfun_kw)
        return self.data.mapped


class IMLFun2OSC(IMLBase):
    """IML function to OSC mapping

    Example:
        This will send the output vector to '/out/vec'.

        ```py
        def infun():
            return [0,0,0,0]

        tv.iml.test2osc = {
            'size': (4, 8), 
            'io': (infun, 'out_vec'),
        }
        ```
    """
    def __init__(self, osc_map: OSCMap, **kwargs) -> None:
        """Initialise IMLFun2OSC

        Args:
            osc_map (OSCMap, required): OSCMap instance.
            kwargs:
                io (tuple, required): (callable, str) input function and output OSC route.
                see IMLBase kwargs.
        """
        assert "io" in kwargs, f"IMLFun2Vec requires 'io=(callable, str)' kwarg."
        assert callable(
            kwargs["io"][0]
        ), f"IMLFun2Vec 'io[0]' not callable, got {type(kwargs['io'][0])}."
        assert (
            isinstance(kwargs["io"][1], str)
        ), f"IMLFun2Vec 'io[1]' not str, got {type(kwargs['io'][1])}."
        self.infun = kwargs["io"][0]
        self.infun_params = inspect.signature(self.infun).parameters
        self.osc_map = osc_map
        self.out_osc_route = '/return'+ANGUILLA_ROUTES['map']
        self.osc_map.send_list_inline(self.out_osc_route, self.update, kwargs["size"][1], count=kwargs.get("update_rate", 10))
        kwargs["updater"] = self.osc_map.dict["send"][self.out_osc_route]['updater']
        super().__init__(**kwargs)

    def update(self) -> list[float]:
        """Update mapped data.

        Returns:
            list[float]: Mapped data.
        """
        if len(self.infun_params) > 0:
            invec = self.infun(**self.infun_kw)
        else:
            invec = self.infun()
        self.data.mapped = self.map(invec, **self.map_kw)
        return self.data.mapped.tolist()


class IMLOSC2Vec(IMLBase):
    """IML OSC to vector mapping

    Example:
        This will map the OSC input to the output vector and store it in `tv.iml.o['name']`.

        ```py
        tv.iml.test2vec = {
            'size': (4, 8), 
            'io': ('in_vec', None),
        }
        # ...
        flock_s_outvec = tv.iml.o['flock_p2flock_s']
        ```
    """
    def __init__(self, osc_map, outvecs: dict, name: str, **kwargs) -> None:
        """Initialise IMLOSC2Vec

        Args:
            osc_map (OSCMap, required): OSCMap instance.
            outvecs (dict): Output vectors dict.
            name (str): Name of output vector.
            kwargs:
                io (tuple, required): (str, None) input OSC route.
                see IMLBase kwargs.
        """
        assert "io" in kwargs, f"IMLOSC2Vec requires 'io=(str, None)' kwarg."
        assert (
            type(kwargs["io"][0]) is str
        ), f"IMLOSC2Vec 'io[0]' not str, got {type(kwargs['io'][0])}."
        assert (
            kwargs["io"][1] is None
        ), f"IMLOSC2Vec 'io[1]' is not None, got {type(kwargs['io'][1])}."
        self.name = kwargs.get("name", None)
        self.osc_map = osc_map
        self.osc_in_route = ANGUILLA_ROUTES['map']
        self.osc_map.receive_list_inline(self.osc_in_route, self.update, kwargs["size"][0], count=kwargs.get("update_rate", 10))
        kwargs["updater"] = self.osc_map.dict["receive"][self.osc_in_route]['updater']
        self.outvecs = outvecs
        self.name = name
        super().__init__(**kwargs)

    def update(self, vector: list[float]) -> list[float]:
        """Update mapped data.

        Args:
            vector (list[float]): Input vector.

        Returns:
            list[float]: Mapped data.
        """
        self.data.mapped = self.map(vector, **self.map_kw)
        if self.name is not None:
            self.outvecs[self.name] = self.data.mapped
        return self.data.mapped


class IMLOSC2Fun(IMLBase):
    """IML OSC to function mapping

    Example:
        ```py
        def outfun(vector):
            print('outvec', vector)

        tv.iml.test2fun = {
            'size': (4, 8), 
            'io': ('in_vec', outfun),
        }
        ```
    """
    def __init__(self, osc_map, **kwargs) -> None:
        """Initialise IMLOSC2Fun

        Args:
            osc_map (OSCMap, required): OSCMap instance.
            kwargs:
                io (tuple, required): (str, callable) input OSC route and output function.
                see IMLBase kwargs.
        """
        assert "io" in kwargs, f"IMLOSC2Fun requires 'io=(str, callable)' kwarg."
        assert (
            type(kwargs["io"][0]) is str
        ), f"IMLOSC2Fun 'io[0]' not str, got {type(kwargs['io'][0])}."
        assert callable(
            kwargs["io"][1]
        ), f"IMLOSC2Fun 'io[1]' is not callable, got {type(kwargs['io'][1])}."
        self.osc_map = osc_map
        self.osc_in_route = ANGUILLA_ROUTES['map']
        self.osc_map.receive_list_inline(self.osc_in_route, self.update, kwargs["size"][0], count=kwargs.get("update_rate", 10))
        kwargs["updater"] = self.osc_map.dict["receive"][self.osc_in_route]['updater']
        self.outfun = kwargs["io"][1]
        self.outfun_params = inspect.signature(self.outfun).parameters
        super().__init__(**kwargs)

    def update(self, vector: list[float]) -> list[float]:
        """Update mapped data.

        Args:
            vector (list[float]): Input vector.

        Returns:
            list[float]: Mapped data.
        """
        mapped = self.map(vector, **self.map_kw)
        self.data.mapped = self.outfun(mapped, **self.outfun_kw)
        return self.data.mapped


class IMLOSC2OSC(IMLBase):
    """IML OSC to OSC mapping

    Example:
        '/in/vec' is mapped and the output sent to '/out/vec'.

        ```py
        tv.iml.test2fun = {
            'size': (4, 8), 
            'io': ('in_vec', 'out_vec'),
        }
        ```
    """
    def __init__(self, osc_map: OSCMap, osc: iiOSC, **kwargs) -> None:
        """Initialise IMLOSC2OSC

        Args:
            osc_map (OSCMap, required): OSCMap instance.
            osc (OSC): iipyper OSC instance.
            kwargs:
                io (tuple, required): (str, str) input and output OSC routes.
                see IMLBase kwargs.
        """
        assert "io" in kwargs, f"IMLOSC2OSC requires 'io=(str, str)' kwarg."
        assert (
            type(kwargs["io"][0]) is str
        ), f"IMLOSC2OSC 'io[0]' not str, got {type(kwargs['io'][0])}."
        assert (
            type(kwargs["io"][1]) is str
        ), f"IMLOSC2OSC 'io[1]' is not str, got {type(kwargs['io'][1])}."
        self.osc = osc
        self.osc_map = osc_map
        self.osc_in_route = ANGUILLA_ROUTES['map']
        self.osc_map.receive_list_inline(self.osc_in_route, self.update, kwargs["size"][0], count=kwargs.get("update_rate", 10))
        kwargs["updater"] = self.osc_map.dict["receive"][self.osc_in_route]['updater']
        self.out_osc_route = '/return'+ANGUILLA_ROUTES['map']
        super().__init__(**kwargs)

    def update(self, vector: list[float]) -> list[float]:
        """Update mapped data.

        Args:
            vector (list[float]): Input vector.

        Returns:
            list[float]: Mapped data.
        """
        self.data.mapped = self.map(vector, **self.map_kw)
        self.osc.host.send(self.out_osc_route, *self.data.mapped.tolist())
        return self.data.mapped
