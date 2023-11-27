from typing import Any
from anguilla import IML as iiIML
import torch
from iipyper import Updater

from .utils import *

__all__ = ['IMLBase', 'IMLVec2Vec', 'IMLVec2Fun', 'IMLVec2OSC', 'IMLFun2Vec', 'IMLFun2Fun', 'IMLFun2OSC', 'IMLOSC2Vec', 'IMLOSC2OSC', 'IMLOSC2Fun', 'IMLDict', 'IML_TYPES']

IML_TYPES = ['vec2vec', 'vec2fun', 'vec2osc', 'fun2vec', 'fun2fun', 'fun2osc', 'osc2vec', 'osc2osc', 'osc2fun']

class IMLBase(iiIML):
    """IML mapping base class

    Args:
        tolvera: Tolvera instance.
        kwargs:
            size (tuple, required): (input, output) sizes.
            io (tuple, optional): (input, output) functions.
            config (dict, optional): {feature_size: size[0], emb:Identity, interp:Smooth, index:IndexBrute, k:10, verbose:False}.
            updater (cls, optional): See iipyper.osc.update (Updater, OSCUpdater, ...).
            update_rate (int, optional): Updater's update rate (defaults to 1).
            randomise (bool, optional): Randomise mapping on init (defaults to False).
            random_pairs (int, optional): Number of random pairs to add (defaults to 32).
            default_args (tuple, optional): Default args to use in update().
            default_kwargs (dict, optional): Default kwargs to use in update().
            TODO: add Lag/Interpolate
    """
    def __init__(self, **kwargs) -> None:
        assert 'size' in kwargs, f"IMLBase requires 'size' kwarg."
        self.size = kwargs['size']
        self.updater = kwargs.get('updater', Updater(self.update, kwargs.get('update_rate', 1)))
        self.config = kwargs.get('config', {'feature_size': self.size[0]})
        if self.size[0] is tuple:
            self.config['emb'] = 'ProjectAndSort'
        super().__init__(**self.config)
        self.data = dotdict()
        if 'randomise' in kwargs:
            self.random_pairs = kwargs.get('random_pairs', 32)
            self.randomise(self.random_pairs)
    def randomise(self, times):
        while len(self.pairs) < times:
            source = torch.rand(self.size[0])
            target = torch.rand(self.size[1])
            self.add(source, target)
    def update(self, *args, **kwargs):
        self.data.mapped = self.map(*args, **kwargs)
        return self.data.mapped
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Update mapping with args and kwargs. 
        
        Primarily facilitates calling IMLs from IMLDict without args or kwargs.
        
        args and kwargs are stored as prev_args and prev_kwargs.
        If args and kwargs are None, use prev_args and prev_kwargs.
        If prev_args and prev_kwargs are None, use default_args and default_kwargs.
        Otherwise, raise ValueError.
        """
        if args is not None and kwargs is None: self.data.prev_args = args
        if args is None and kwargs is not None: self.data.prev_kwargs = kwargs
        if args is None and kwargs is None:
            if self.data.prev_args is None and self.data.prev_kwargs is None:
                if self.default_args is None and self.default_kwargs is None:
                    raise ValueError(f"[tolvera._iml.IMLBase] All are None: (prev|default_)args, (prev|default_)kwargs.")
                if self.default_args is not None and self.default_kwargs is None:
                    args = self.default_args
                if self.default_args is None and self.default_kwargs is not None:
                    kwargs = self.default_kwargs
            if self.data.prev_args is not None and self.data.prev_kwargs is None:
                args = self.data.prev_args
            if self.data.prev_args is None and self.data.prev_kwargs is not None:
                kwargs = self.data.prev_kwargs
            args = self.data.prev_args
            kwargs = self.data.prev_kwargs
        return self.updater(*args, **kwargs)

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
        assert 'io' in kwargs, f"IMLVec2Fun requires 'io' kwarg."
        assert kwargs['io'][0] is None, f"IMLVec2Fun requires 'io' kwarg to be (None, outfun)."
        assert callable(kwargs['io'][1]), f"IMLVec2Fun requires 'io' kwarg to be (None, outfun)."
        self.outfun = kwargs['io'][1]
        super().__init__(**kwargs)
    def update(self, *args, **kwargs):
        self.data.mapped = self.outfun(self.map(*args, **kwargs))
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
        assert 'io' in kwargs, f"IMLVec2OSC requires 'io' kwarg."
        assert kwargs['io'][0] is None, f"IMLVec2OSC requires 'io' kwarg to be (None, osc_route)."
        assert type(kwargs['io'][1]) is str, f"IMLVec2OSC requires 'io' kwarg to be (None, osc_route)."
        self.route = kwargs['io'][1]
        super().__init__(**kwargs)
    def update(self, *args, **kwargs):
        raise NotImplementedError(f"[tolvera._iml.IMLVec2OSC] update() not implemented.")

class IMLFun2Vec(IMLBase):
    """IML function to vector mapping
    
    Args:
        kwargs:
            io (tuple, required): (callable, None) input function.
            see IMLBase kwargs.
    """
    def __init__(self, **kwargs) -> None:
        assert 'io' in kwargs, f"IMLFun2Vec requires 'io' kwarg."
        assert callable(kwargs['io'][0]), f"IMLFun2Vec requires 'io' kwarg to be (infun, None)."
        assert kwargs['io'][1] is None, f"IMLFun2Vec requires 'io' kwarg to be (infun, None)."
        self.infun = kwargs['io'][0]
        super().__init__(**kwargs)
    def update(self, *args, **kwargs):
        print(f"[tolvera._iml.IMLFun2Vec] update({args}, {kwargs})")
        self.data.mapped = self.map(self.infun(*args, **kwargs))
        return self.data.mapped

class IMLFun2Fun(IMLBase):
    """IML function to function mapping
    
    Args:
        kwargs:
            io (tuple, required): (callable, callable) input and output functions.
            see IMLBase kwargs.
    """
    def __init__(self, **kwargs) -> None:
        assert 'io' in kwargs, f"IMLFun2Fun requires 'io' kwarg."
        assert callable(kwargs['io'][0]), f"IMLFun2Fun requires 'io' kwarg to be (infun, outfun)."
        assert callable(kwargs['io'][1]), f"IMLFun2Fun requires 'io' kwarg to be (infun, outfun)."
        self.infun = kwargs['io'][0]
        self.outfun = kwargs['io'][1]
        super().__init__(**kwargs)
    def update(self, *args, **kwargs):
        self.data.mapped = self.outfun(self.map(self.infun(*args, **kwargs)))
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
    def update(self, *args, **kwargs):
        raise NotImplementedError(f"[tolvera._iml.IMLFun2OSC] update() not implemented.")

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
    def update(self, *args, **kwargs):
        raise NotImplementedError(f"[tolvera._iml.IMLOSC2Vec] update() not implemented.")

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
    def update(self, *args, **kwargs):
        raise NotImplementedError(f"[tolvera._iml.IMLOSC2Fun] update() not implemented.")

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
    def update(self, *args, **kwargs):
        """
        see iml.app.server.map?
        """
        raise NotImplementedError(f"[tolvera._iml.IMLOSC2OSC] update() not implemented.")

class IMLDict(dotdict):
    """IML mapping dict
    
    TODO: remove tolvera dependency?"""
    def __init__(self, tolvera) -> None:
        self.tv = tolvera
    def set(self, name, kwargs: dict) -> None:
        print(f"[tolvera._iml.IMLDict] set({name}, {kwargs})")
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
        print(f"[tolvera._iml.IMLDict] add({name}, {iml_type}, {kwargs})")
        match iml_type:
            case 'vec2vec': ins = IMLVec2Vec(**kwargs)
            case 'vec2fun': ins = IMLVec2Fun(**kwargs)
            case 'vec2osc': ins = IMLVec2OSC(self.tv.osc.map, **kwargs)
            case 'fun2vec': ins = IMLFun2Vec(**kwargs)
            case 'fun2fun': ins = IMLFun2Fun(**kwargs)
            case 'fun2osc': ins = IMLFun2OSC(self.tv.osc.map, **kwargs)
            case 'osc2vec': ins = IMLOSC2Vec(self.tv.osc.map, **kwargs)
            case 'osc2fun': ins = IMLOSC2Fun(self.tv.osc.map, **kwargs)
            case 'osc2osc': ins = IMLOSC2OSC(self.tv.osc.map, **kwargs)
            case _:
                raise ValueError(f"[tolvera._iml.IMLDict] Invalid IML_TYPE '{iml_type}'. Valid IML_TYPES: {IML_TYPES}.")
        self[name] = ins
    def __call__(self, name, *args: Any, **kwargs: Any) -> Any:
        if name in self:
            return self[name](*args, **kwargs)
        else:
            raise ValueError(f"[tolvera._iml.IMLDict] '{name}' not in dict.")

