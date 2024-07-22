import json
import os
import xml.etree.ElementTree as ET
from typing import Any, get_type_hints

import numpy as np
from iipyper.osc import OSC as iiOSC
from iipyper.util import *

from .maxmsp import MaxPatcher
from .pd import PdPatcher
from .update import OSCReceiveListUpdater, OSCReceiveUpdater, OSCSend, OSCSendUpdater


class OSCMap:
    """
    OSCMap maps OSC messages to functions
    It creates a Max/MSP patcher that can be used to control the OSCMap
    It uses OSCSendUpdater and OSCReceiveUpdater to decouple incoming messages
    """

    def __init__(
        self,
        osc: iiOSC,
        client_name="client",
        patch_type="Max",  # | "Pd"
        patch_filepath="osc_controls",
        create_patch=True,
        pd_net_or_udp="udp",
        pd_bela=False,
        export=None,  # 'JSON' | 'XML' | True
    ) -> None:
        self.osc = osc
        self.client_name = client_name
        self.client_address, self.client_port = self.osc.client_names[self.client_name]
        self.dict = {"send": {}, "receive": {}}
        self.create_patch = create_patch
        self.patch_filepath = patch_filepath
        self.patch_type = patch_type
        if create_patch is True:
            self.init_patcher(patch_type, patch_filepath, pd_net_or_udp, pd_bela)
        if export is not None:
            assert (
                export == "JSON" or export == "XML" or export == True
            ), "export must be 'JSON', 'XML' or True"
        self.export = export

    def init_patcher(self, patch_type, patch_filepath, pd_net_or_udp, pd_bela):
        # create self.patch_dir if it doesn't exist
        self.patch_dir = "pd" if patch_type == "Pd" else "max"
        if not os.path.exists(self.patch_dir):
            print(f"Creating {self.patch_dir} directory...")
            os.makedirs(self.patch_dir)
        self.patch_appendix = "_local" if self.osc.host == "127.0.0.1" else "_remote"
        self.patch_filepath = (
            self.patch_dir + "/" + patch_filepath + self.patch_appendix
        )
        if patch_type == "Max":
            self.patcher = MaxPatcher(self.osc, self.client_name, self.patch_filepath)
        elif patch_type == "Pd":
            if pd_bela is True:
                self.patcher = PdPatcher(
                    self.osc,
                    self.client_name,
                    self.patch_filepath,
                    net_or_udp=pd_net_or_udp,
                    bela=True,
                )
            else:
                self.patcher = PdPatcher(
                    self.osc,
                    self.client_name,
                    self.patch_filepath,
                    net_or_udp=pd_net_or_udp,
                )
        else:
            assert False, "`patch_type` must be 'Max' or 'Pd'"

    def add(self, **kwargs):
        print(
            "DeprecationError: OSCMap.add() has been split into separate functions: use `send_args`, `send_list`, `receive_args` or `receive_list` instead!"
        )
        exit()

    def map_func_to_dict(self, func, kwargs):
        if "name" not in kwargs:
            n = func.__name__
            address = "/" + n.replace("_", "/")
        else:
            if isinstance(kwargs["name"], str):
                n = kwargs["name"]
                address = "/" + kwargs["name"].replace("_", "/")
            else:
                raise TypeError(
                    f"OSC func name must be string, found {str(type(kwargs['name']))}"
                )
        # TODO: Move this into specific send/receive functions
        params = {
            k: v
            for k, v in kwargs.items()
            if k != "count" and k != "send_mode" and k != "length" and k != "name"
        }
        # TODO: turn params into dict with type hints (see export_dict)
        hints = get_type_hints(func)
        f = {"f": func, "name": n, "address": address, "params": params, "hints": hints}
        return f

    """
    send args
    """

    def send_args(self, **kwargs):
        def decorator(func):
            def wrapper(*args):
                self.add_send_args(func, kwargs)
                return func()

            default_args = [
                kwargs[a][0]
                for a in kwargs
                if a != "count" and a != "send_mode" and a != "name"
            ]
            wrapper(*default_args)
            return wrapper

        return decorator

    def add_send_args(self, func, kwargs):
        self.add_send_args_to_osc_map(func, kwargs)
        if self.create_patch is True:
            self.add_send_args_to_patcher(func)

    def add_send_args_to_osc_map(self, func, kwargs):
        f = self.map_func_to_dict(func, kwargs)
        if kwargs["send_mode"] == "broadcast":
            f["updater"] = OSCSendUpdater(
                self.osc,
                f["address"],
                f=func,
                count=kwargs["count"],
                client=self.client_name,
            )
        else:
            f["sender"] = OSCSend(
                self.osc,
                f["address"],
                f=func,
                # count=kwargs["count"],
                client=self.client_name,
            )
        f["type"] = "args"
        self.dict["send"][f["name"]] = f
        if self.export is not None:
            self.export_dict()

    def add_send_args_to_patcher(self, func):
        f = self.dict["send"][func.__name__]
        self.patcher.send_args_func(f)

    """
    send list
    """

    def send_list(self, **kwargs):
        def decorator(func):
            def wrapper(*args):
                self.add_send_list(func, kwargs)
                # TODO: This was originally here to sync defaults with client
                # but it causes init order isses in IMLFun2OSC.update
                # return func()

            default_arg = [
                kwargs[a][0]
                for a in kwargs
                if a != "count" and a != "send_mode" and a != "length" and a != "name"
            ]
            wrapper(default_arg)
            return wrapper

        return decorator

    def add_send_list(self, func, kwargs):
        f = self.add_send_list_to_osc_map(func, kwargs)
        if self.create_patch is True:
            self.add_send_list_to_patcher(f)

    def add_send_list_to_osc_map(self, func, kwargs):
        f = self.map_func_to_dict(func, kwargs)
        # TODO: Hack for send_list_inline which doesn't have a return type hint
        if "return" in f["hints"]:
            hint = f["hints"]["return"]
        else:
            hint = list[float]
        assert hint == list[float], "send_list can only send list[float], found " + str(
            hint
        )
        if kwargs["send_mode"] == "broadcast":
            f["updater"] = OSCSendUpdater(
                self.osc,
                f["address"],
                f=func,
                count=kwargs["count"],
                client=self.client_name,
            )
        else:
            f["sender"] = OSCSend(
                self.osc,
                f["address"],
                f=func,
                count=kwargs["count"],
                client=self.client_name,
            )
        f["type"] = "list"
        f["length"] = kwargs["length"]
        self.dict["send"][f["name"]] = f
        if self.export is not None:
            self.export_dict()
        return f

    def add_send_list_to_patcher(self, func):
        f = self.dict["send"][func["name"]]
        self.patcher.send_list_func(f)

    def send_list_inline(self, name: str, sender_func, length: int, send_mode="broadcast", count=1, **kwargs):
        kwargs = {**kwargs, **{"name": name, "vector": (0.0,0.0,1.0), "length": length, "send_mode": send_mode, "count": count}}
        self.send_list(**kwargs)(sender_func)

    """
    send kwargs
    """

    def send_kwargs(self, **kwargs):
        raise NotImplementedError("send_kwargs not implemented yet")

    """
    receive args
    """

    def receive_args(self, **kwargs):
        def decorator(func):
            def wrapper(*args):
                self.add_receive_args(func, kwargs)
                return func(*args)

            default_args = [
                kwargs[a][0] for a in kwargs if a != "count" and a != "name"
            ]
            wrapper(*default_args)
            return wrapper

        return decorator

    def add_receive_args(self, func, kwargs):
        f = self.add_receive_args_to_osc_map(func, kwargs)
        if self.create_patch is True:
            self.add_receive_args_to_patcher(f)

    def add_receive_args_to_osc_map(self, func, kwargs):
        f = self.map_func_to_dict(func, kwargs)
        f["updater"] = OSCReceiveUpdater(
            self.osc, f["address"], f=func, count=kwargs["count"]
        )
        f["type"] = "args"
        self.dict["receive"][f["name"]] = f
        return f

    def add_receive_args_to_patcher(self, func):
        f = self.dict["receive"][func["name"]]
        self.patcher.receive_args_func(f)

    def receive_args_inline(self, name: str, receiver_func, **kwargs):
        kwargs = {**kwargs, **{"count": 1, "name": name}}
        self.receive_args(**kwargs)(receiver_func)

    """
    receive list
    """

    def receive_list(self, **kwargs):
        def decorator(func):
            def wrapper(*args):
                self.add_receive_list(func, kwargs)
                # TODO: This was originally here to sync defaults with client
                # but it causes init order isses in IMLOSC2Vec.init
                # return func(*args)

            # TODO: This probably shouldn't be here...
            randomised_list = self.randomise_list(
                kwargs["length"], kwargs["vector"][1], kwargs["vector"][2]
            )
            wrapper(randomised_list)
            return wrapper

        return decorator

    def randomise_list(self, length, min, max):
        return min + (np.random.rand(length).astype(np.float32) * (max - min))

    def add_receive_list(self, func, kwargs):
        f = self.add_receive_list_to_osc_map(func, kwargs)
        if self.create_patch is True:
            self.add_receive_list_to_patcher(f)

    def add_receive_list_to_osc_map(self, func, kwargs):
        """
        TODO: Should this support list[float] only, or list[int] list[str] etc?
        """
        f = self.map_func_to_dict(func, kwargs)
        assert (
            len(f["params"]) == 1
        ), "receive_list can only receive one param (list[float])"
        hint = f["hints"][list(f["params"].keys())[0]]
        assert (
            hint == list[float]
        ), "receive_list can only receive list[float], found " + str(hint)
        f["updater"] = OSCReceiveListUpdater(
            self.osc, f["address"], f=func, count=kwargs["count"]
        )
        f["type"] = "list"
        f["length"] = kwargs["length"]
        self.dict["receive"][f["name"]] = f
        if self.export is not None:
            self.export_dict()
        return f

    def add_receive_list_to_patcher(self, func):
        f = self.dict["receive"][func["name"]]
        self.patcher.receive_list_func(f)

    def receive_list_inline(self, name: str, receiver_func, length: int, count=1, **kwargs):
        kwargs = {**kwargs, **{"name": name, "length": length, "count": count, "vector": (0, 0, 1)}}
        self.receive_list(**kwargs)(receiver_func)

    def receive_list_with_idx(
        self, name: str, receiver, idx_len: int, vec_len: int, attr=None
    ):
        """
        Create an OSC list handler that assumes that the first `idx_len` values are indices into some struct being modified by a receiver function, and the rest are args as a list, i.e.
            /name idx0 idx1 ... idxN arg0 arg1 ... argM
            ...
            receiver((idx0 idx1 ... idxN), args)
        Intended as a utility function to be used by external classes where it's not possible to use a decorator like `receive_list`.
        """

        def handler(vector: list[float]):
            arg_len = len(vector[idx_len:])
            assert (
                arg_len == vec_len
            ), f"len(args) != len(list) ({arg_len} != {vec_len})"
            if idx_len:
                indices = tuple([int(v) for v in vector[:idx_len]])
                if attr is None:
                    receiver(indices, vector[idx_len:])
                else:
                    receiver(indices, attr, vector[idx_len:])
            else:
                if attr is None:
                    receiver(vector)
                else:
                    receiver(attr, vector)

        kwargs = {
            "vector": (0, 0, 1),
            "length": vec_len + idx_len,
            "count": 1,
            "name": name,
        }
        self.receive_list(**kwargs)(handler)

    """
    receive kwargs
    """

    def receive_kwargs(self, **kwargs):
        """
        Same as receive_args but with named params
        """
        raise NotImplementedError("receive_kwargs not implemented yet")

    """
    xml / json export
    """

    def export_dict(self):
        """
        Save the OSCMap dict as XML
        """
        client_ip, client_port = self.osc.client_names[self.client_name]
        # TODO: This should be defined in the OSCMap dict / on init
        metadata = {
            "HostIP": self.osc.host,
            "HostPort": str(self.osc.port),
            "ClientName": self.client_name,
            "ClientIP": client_ip,
            "ClientPort": str(client_port),
        }
        root = ET.Element("OpenSoundControlSchema")
        metadata_element = ET.SubElement(root, "Metadata", **metadata)
        sends = self.dict["send"]
        receives = self.dict["receive"]
        for io in ["Send", "Receive"]:
            ET.SubElement(root, io)
        for io in ["send", "receive"]:
            for name in self.dict[io]:
                f = self.dict[io][name]
                if f["type"] == "args":
                    self.xml_add_args_params(root, name, io, f)
                elif f["type"] == "list":
                    self.xml_add_list_param(root, name, io, f)
                elif f["type"] == "kwargs":
                    raise NotImplementedError("kwargs not implemented yet")
        self.export_update(root)

    def xml_add_args_params(self, root, name, io, f):
        params = f["params"]
        hints = f["hints"]
        kw = {
            "Address": "/" + name.replace("_", "/"),
            "Type": f["type"],
            "Params": str(len(params)),
        }
        route = ET.SubElement(root.find(io.capitalize()), "Route", **kw)
        for i, p in enumerate(params):
            # TODO: This should already be defined by this point
            if io == "receive":
                p_type = hints[p].__name__
            elif io == "send":
                p_type = hints["return"].__args__[i].__name__
            kw = {
                "Name": p,
                "Type": p_type,
                "Default": str(params[p][0]),
                "Min": str(params[p][1]),
                "Max": str(params[p][2]),
            }
            ET.SubElement(route, "Param", **kw)

    def xml_add_list_param(self, root, name, io, f):
        params = f["params"]
        hints = f["hints"]
        length = f["length"]
        kw = {
            "Address": "/" + name.replace("_", "/"),
            "Type": f["type"],
            "Length": str(length),
        }
        route = ET.SubElement(root.find(io.capitalize()), "Route", **kw)
        p = list(params.keys())[0]
        if io == "receive":
            p_type = hints[p].__name__
        elif io == "send":
            p_type = hints["return"].__args__[0].__name__
        kw = {
            "Name": p,
            "Type": p_type,
            "Default": str(params[p][0]),
            "Min": str(params[p][1]),
            "Max": str(params[p][2]),
        }
        ET.SubElement(route, "ParamList", **kw)

    def export_update(self, root):
        tree = ET.ElementTree(root)
        ET.indent(tree, space="\t", level=0)
        if self.export == "XML":
            self.save_xml(tree, root)
        elif self.export == "JSON":
            self.save_json(root)
        elif self.export == True:
            self.save_xml(tree, root)
            self.save_json(root)

    def save_xml(self, tree, root):
        tree.write(self.patch_filepath + ".xml")
        print(f"Exported OSCMap to {self.patch_filepath}.xml")

    def save_json(self, xml_root):
        # TODO: params should be `params: []` and not `param: {}, param: {}, ...`
        json_dict = self.xml_to_json(
            ET.tostring(xml_root, encoding="utf8", method="xml")
        )
        with open(self.patch_filepath + ".json", "w") as f:
            f.write(json_dict)
        print(f"Exported OSCMap to {self.patch_filepath}.json")

    def etree_to_dict(self, t):
        tag = self.pascal_to_camel(t.tag)
        d = {tag: {} if t.attrib else None}
        children = list(t)
        if children:
            dd = {}
            for dc in map(self.etree_to_dict, children):
                for k, v in dc.items():
                    try:
                        dd[k].append(v)
                    except KeyError:
                        dd[k] = [v]
            d = {tag: {k: v[0] if len(v) == 1 else v for k, v in dd.items()}}
        if t.attrib:
            d[tag].update((self.pascal_to_camel(k), v) for k, v in t.attrib.items())
        if t.text:
            text = t.text.strip()
            if children or t.attrib:
                if text:
                    d[tag]["#text"] = text
            else:
                d[tag] = text
        return d

    def pascal_to_camel(self, s):
        return s[0].lower() + s[1:]

    def xml_to_json(self, xml_str):
        e = ET.ElementTree(ET.fromstring(xml_str))
        return json.dumps(self.etree_to_dict(e.getroot()), indent=4)

    def update(self):
        for k, v in self.dict["send"].items():
            if "updater" in v:
                ret = v["updater"]()
            # v['updater']()
        for k, v in self.dict["receive"].items():
            v["updater"]()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.update()
