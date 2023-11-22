from typing import Any
from iml import IML as iiIML, embed, interpolate, nnsearch
from iml.types import Optional, Union

from .utils import *

class IMLFuncToFunc(iiIML):
    """IML function to function mapping"""
    def __init__(self, context, **kwargs):
        super().__init__(**kwargs)
        self.ctx = context

class IMLOSCToOSC(iiIML):
    """IML OSC to OSC mapping"""
    def __init__(self, context, **kwargs):
        super().__init__(**kwargs)
        self.ctx = context

class IMLOSCToFunc(iiIML):
    """IML OSC to function mapping"""
    def __init__(self, context, **kwargs):
        super().__init__(**kwargs)
        self.ctx = context

class IMLFuncToOSC(iiIML):
    """IML function to OSC mapping"""
    def __init__(self, context, **kwargs):
        super().__init__(**kwargs)
        self.ctx = context

class IMLDict(dotdict):
    def __init__(self, tolvera) -> None:
        self.tv = tolvera
    def set(self, name, kwargs: dict) -> None:
        if name in self:
            raise ValueError(f"[tolvera.state.StateDict] '{name}' already in dict.")
        try:
            if name == 'tv' and type(kwargs) is not dict and type(kwargs) is not tuple:
                self[name] = kwargs
            elif type(kwargs) is dict:
                self[name] = IML(self.tv, name=name, **kwargs)
            elif type(kwargs) is tuple:
                self[name] = IML(self.tv, name, *kwargs)
            else:
                raise TypeError(f"[tolvera.state.StateDict] set() requires dict|tuple, not {type(kwargs)}")
        except TypeError as e:
            print(f"[tolvera.state.StateDict] TypeError setting {name}: {e}")
        except ValueError as e:
            print(f"[tolvera.state.StateDict] ValueError setting {name}: {e}")
        except Exception as e:
            print(f"[tolvera.state.StateDict] UnexpectedError setting {name}: {e}")
            raise

class IML:
    def __init__(self, context, **kwargs) -> None:
        self.ctx = context
        self.kwargs = kwargs
        self.iml = iiIML(**kwargs)
        # TODO: Add default config(s)
        self.config = {}
        self.dict = {}
    def add_instance(self, key: str, *kwargs: Any):
        kwargs['iml'] = iiIML(self.config)
        self.dict[key] = kwargs
        if self.ctx.osc is not False:
            self.add_to_osc_map(key)
    def add_to_osc_map(self, key: str):
        pass
