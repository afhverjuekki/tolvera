from typing import Any
from anguilla import IML as iiIML
from torch import from_numpy
from iipyper import Updater

from .utils import *

class IMLFun2Fun:
    """IML function to function mapping
    
    Args:
        tolvera: Tolvera instance
        kwargs:
            name: name of the mapping
            iml_config: config dict for IML
            size: size of the input and output vectors
            input_func: function to map
            output_func: function to map to
            update_rate: update rate of the mapping
    """
    def __init__(self, tolvera, **kwargs) -> None:
        self.tv = tolvera
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.iml_config['feature_size'] = self.size[0]
        self.iml = iiIML(**self.iml_config)
        self.updater = Updater(self.map, self.update_rate)
        self.data = dotdict()
    def map(self, *args, **kwargs):
        self.data.input = self.fun[0](*args, **kwargs)
        self.data.mapped[:] = from_numpy(self.iml.map(self.data.input))
        # if self._lag_coef != 0.0:
        #     self.target = self.lag(self.target, self._lag_coef)
        return self.data.mapped
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.updater(*args, **kwds)

class IMLOSC2OSC(iiIML):
    """IML OSC to OSC mapping"""
    def __init__(self, tolvera, **kwargs) -> None:
        super().__init__(**kwargs)
        self.tv = tolvera

class IMLOSC2Fun(iiIML):
    """IML OSC to function mapping"""
    def __init__(self, tolvera, **kwargs) -> None:
        super().__init__(**kwargs)
        self.tv = tolvera

class IMLFun2OSC(iiIML):
    """IML function to OSC mapping"""
    def __init__(self, tolvera, **kwargs) -> None:
        super().__init__(**kwargs)
        self.tv = tolvera

class IMLDict(dotdict):
    def __init__(self, tolvera) -> None:
        self.tv = tolvera
    def set(self, name, kwargs: dict) -> None:
        if name in self:
            raise ValueError(f"[tolvera._iml.IMLDict] '{name}' already in dict.")
        try:
            if name == 'tv' and type(kwargs) is not dict and type(kwargs) is not tuple:
                self[name] = kwargs
            elif type(kwargs) is dict:
                if 'type' not in kwargs:
                    raise ValueError(f"[tolvera._iml.IMLDict] IMLDict requires 'type' key.")
                self.add(name, kwargs['type'], **kwargs)
            elif type(kwargs) is tuple:
                iml_type = kwargs[0] # TODO: which index is 'iml_type'?
                self.add(name, iml_type, *kwargs)
            else:
                raise TypeError(f"[tolvera._iml.IMLDict] set() requires dict|tuple, not {type(kwargs)}")
        except TypeError as e:
            print(f"[tolvera._iml.IMLDict] TypeError setting {name}: {e}")
        except ValueError as e:
            print(f"[tolvera._iml.IMLDict] ValueError setting {name}: {e}")
        except Exception as e:
            print(f"[tolvera._iml.IMLDict] UnexpectedError setting {name}: {e}")
            raise
    def __setattr__(self, __name: str, __value: Any) -> None:
        self.set(__name, __value)
    def add(self, name, iml_type, **kwargs):
        match iml_type:
            case 'fun2fun': ins = IMLFun2Fun(self.tv, **kwargs)
            case 'osc2osc': ins = IMLOSC2OSC(self.tv, **kwargs)
            case 'osc2fun': ins = IMLOSC2Fun(self.tv, **kwargs)
            case 'fun2osc': ins = IMLFun2OSC(self.tv, **kwargs)
            case _:
                raise ValueError(f"[tolvera._iml.IMLDict] Invalid iml_type '{iml_type}'. Valid iml_types are 'fun2fun', 'osc2osc', 'osc2fun', 'fun2osc'.")
        self[name] = ins
    def __call__(self, name, *args: Any, **kwds: Any) -> Any:
        if name in self:
            self[name](*args, **kwds)
        else:
            raise ValueError(f"[tolvera._iml.IMLDict] '{name}' not in dict.")

