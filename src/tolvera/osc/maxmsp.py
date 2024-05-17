import json
import typing


class MaxPatcher:
    """
    TODO: copy-paste using stdout
    TODO: add scale objects before send and after receive
    TODO: add default values via loadbangs
    TODO: move udpsend/udpreceive to the top left
    TODO: dict of object ids
    TODO: add abstraction i/o messages e.g. param names, state save/load/dumps
    """

    def __init__(
        self,
        osc,
        client_name="client",
        filepath="osc_controls",
        x=0.0,
        y=0.0,
        w=1600.0,
        h=900.0,
        v="8.5.4",
    ) -> None:
        self.patch = {
            "patcher": {
                "fileversion": 1,
                "appversion": {
                    "major": v[0],
                    "minor": v[2],
                    "revision": v[4],
                    "architecture": "x64",
                    "modernui": 1,
                },
                "classnamespace": "box",
                "rect": [x, y, w, h],
                "bglocked": 0,
                "openinpresentation": 0,
                "default_fontsize": 12.0,
                "default_fontface": 0,
                "default_fontname": "Arial",
                "gridonopen": 1,
                "gridsize": [15.0, 15.0],
                "gridsnaponopen": 1,
                "objectsnaponopen": 1,
                "statusbarvisible": 2,
                "toolbarvisible": 1,
                "lefttoolbarpinned": 0,
                "toptoolbarpinned": 0,
                "righttoolbarpinned": 0,
                "bottomtoolbarpinned": 0,
                "toolbars_unpinned_last_save": 0,
                "tallnewobj": 0,
                "boxanimatetime": 200,
                "enablehscroll": 1,
                "enablevscroll": 1,
                "devicewidth": 0.0,
                "description": "",
                "digest": "",
                "tags": "",
                "style": "",
                "subpatcher_template": "",
                "assistshowspatchername": 0,
                "boxes": [],
                "lines": [],
                "dependency_cache": [],
                "autosave": 0,
            }
        }
        self.types = {
            "print": "print",
            "message": "message",
            "object": "newobj",
            "comment": "comment",
            "slider": "slider",
            "float": "flonum",
            "int": "number",
            "bang": "button",
        }
        self.osc = osc
        self.client_name = client_name
        self.client_address, self.client_port = self.osc.client_names[self.client_name]
        self.filepath = filepath
        self.init()

    def init(self):
        self.w = 5.5  # default width (scaling factor)
        self.h = 22.0  # default height (pixels)
        self.s_x, self.s_y = 30, 125  # insertion point
        self.r_x, self.r_y = 30, 575  # insertion point
        self.patcher_ids = {}
        self.patcher_ids["send_id"] = self.osc_send(
            self.osc.host, self.osc.port, self.s_x, 30, print_label="sent"
        )
        self.patcher_ids["receive_id"] = self.osc_receive(
            self.client_port, self.s_x + 150, 30, print_label="received"
        )
        self.comment("Max → Python", self.s_x, self.s_y, 24)
        self.comment("Python → Max", self.r_x, self.r_y, 24)
        self.s_y += 50
        self.r_y += 50
        self.save(self.filepath)

    """
    basic objects
    """

    def box(self, box_type, inlets, outlets, x, y, w, h=None):
        if h is None:
            h = self.h
        box_id, box = self.create_box(box_type, inlets, outlets, x, y, w, h)
        return self._box(box)

    def _box(self, box):
        self.patch["patcher"]["boxes"].append(box)
        return self.id_from_str(box["box"]["id"])

    def create_box(self, box_type, inlets, outlets, x, y, w, h=None):
        if h is None:
            h = self.h
        box_id = len(self.patch["patcher"]["boxes"]) + 1
        box = {
            "box": {
                "id": "obj-" + str(box_id),
                "maxclass": self.types[box_type],
                "numinlets": inlets,
                "numoutlets": outlets,
                "patching_rect": [x, y, w, h],
            }
        }
        if outlets > 0:
            if outlets == 1:
                box["box"]["outlettype"] = [""]
            match box_type:
                case "int" | "float" | "bang":
                    box["box"]["outlettype"] = ["", "bang"]
        return box_id, box

    def object(self, text: str, inlets: int, outlets: int, x: float, y: float):
        box_id, box = self.create_box(
            "object", inlets, outlets, x, y, len(text) * self.w
        )
        box["box"]["text"] = text
        self._box(box)
        return box_id

    def message(self, text, x, y):
        box_id, box = self.create_box("message", 2, 1, x, y, len(text) * self.w)
        box["box"]["text"] = text
        self._box(box)
        return box_id

    def comment(self, text, x, y, fontsize=12):
        box_id, box = self.create_box("comment", 0, 0, x, y, len(text) * self.w)
        box["box"]["text"] = text
        box["box"]["fontsize"] = fontsize
        self._box(box)
        return box_id

    def bang(self, x, y):
        box_id, box = self.create_box("bang", 1, 1, x, y, 20.0)
        self._box(box)
        return box_id

    def slider(self, x, y, min_val, size, float=False):
        box_id, box = self.create_box("slider", 1, 1, x, y, 20.0, 140.0)
        if float:
            box["box"]["floatoutput"] = 1
        box["box"]["min"] = min_val
        box["box"]["size"] = size
        self._box(box)
        return box_id

    """
    connections
    """

    def connect(self, src, src_outlet, dst, dst_inlet):
        patchline = {
            "patchline": {
                "destination": ["obj-" + str(dst), dst_inlet],
                "source": ["obj-" + str(src), src_outlet],
            }
        }
        self.patch["patcher"]["lines"].append(patchline)
        return patchline

    """
    osc send/receive
    """

    def osc_send(self, ip, port, x, y, print=True, print_label=None):
        box_id_0 = self.object("r send", 0, 1, x, y)
        box_id = self.object("udpsend " + ip + " " + str(port), 1, 0, x, y + 25)
        if print:
            text = "print" if print_label is None else "print " + print_label
            print_id = self.object(text, 1, 0, x + 50, y)
            self.connect(box_id_0, 0, box_id, 0)
            self.connect(box_id_0, 0, print_id, 0)
            return box_id_0
        return box_id

    def osc_receive(self, port, x, y, print=True, print_label=None):
        box_id_0 = self.object("s receive", 0, 1, x, y + 25)
        box_id = self.object("udpreceive " + str(port), 1, 1, x, y)
        if print:
            text = "print" if print_label is None else "print " + print_label
            print_id = self.object(text, 1, 0, x + 60, y + 25)
            self.connect(box_id, 0, print_id, 0)
            self.connect(box_id, 0, box_id_0, 0)
            return box_id_0
        return box_id

    def osc_route(self, port, x, y, print=True, print_label=None):
        """
        [route path]
        [s name] [print]
        [unpack] ?
        [r name]
        """
        pass

    """
    osc send/receive args/list
    """

    def send_args_func(self, f):
        hints = typing.get_type_hints(f["f"])["return"].__args__
        f_p = f["params"]
        params = []
        if len(f_p) == 0:
            self.osc_receive_msg(self.r_x, self.r_y, f["address"])
        else:
            for i, p in enumerate(f_p):
                p_def, p_min, p_max = f_p[p][0], f_p[p][1], f_p[p][2]
                params.append(
                    {
                        "label": p,
                        "data": hints[i].__name__,
                        "min_val": p_min,
                        "size": p_max - p_min,
                    }
                )
            self.osc_receive_with_controls(self.r_x, self.r_y, f["address"], params)
        self.r_x += max(len(params) * 52.0 + 100.0, len(f["address"]) * 6.0 + 25.0)
        self.save(self.filepath)

    def send_list_func(self, f):
        raise NotImplementedError("send_list_func not implemented yet")

    def receive_args_func(self, f):
        hints = typing.get_type_hints(f["f"])
        f_p = f["params"]
        params = []
        if len(f_p) == 0:
            self.osc_send_msg(self.s_x, self.s_y, f["address"])
        else:
            for p in f_p:
                p_def, p_min, p_max = f_p[p][0], f_p[p][1], f_p[p][2]
                params.append(
                    {
                        "label": p,
                        "data": hints[p].__name__,
                        "min_val": p_min,
                        "size": p_max - p_min,
                    }
                )
            self.osc_send_with_controls(self.s_x, self.s_y, f["address"], params)
        self.s_x += max(len(params) * 52.0 + 100.0, len(f["address"]) * 6.0 + 25.0)
        self.save(self.filepath)

    def receive_list_func(self, f):
        self.osc_send_list(self.s_x, self.s_y, f["address"], f["params"])
        self.s_x += len(f["address"]) * 6.0 + 50.0
        self.save(self.filepath)

    """
    osc send/receive no args/list (msg)
    """

    def osc_send_msg(self, x, y, path):
        msg_id = self.message(path, x, y + 225 + self.h)
        send_id = self.object("s send", 1, 0, x, y + 250 + self.h)
        self.connect(msg_id, 0, send_id, 0)
        return msg_id

    def osc_receive_msg(self, x, y, path):
        receive_id = self.object("r receive", 0, 1, x, y + 225 + self.h)
        msg_id = self.message(path, x, y + 250 + self.h)
        self.connect(receive_id, 0, msg_id, 0)
        return msg_id

    """
    osc send/receive args with line, slider, rate-limiting, and change detection
    """

    def osc_send_with_controls(self, x, y, path, parameters):
        # TODO: add default param value and a loadbang
        """
        [comment path]
        [comment args]
        [r path_arg_name]
        sliders
        |                   |
        [pak $1 $2 $3 ...]
        |
        [msg /path $1 $2 $3 ...]
        |
        [s send]
        """
        y_off = 0
        # [comment path]
        path_comment_id = self.comment(path, x, y + y_off)
        y_off += 15
        param_comment_ids, _y_off = self.param_comments(x, y + y_off, parameters)

        # [r path_arg_name]
        y_off += 35
        receive_ids = [
            self.object(
                "r " + path.replace("/", "_")[1:] + "_" + p["label"][0:3],
                1,
                0,
                x + i * 52.0,
                y + y_off + (0 if i % 2 == 0 else 25),
            )
            for i, p in enumerate(parameters)
        ]
        y_off += 30

        # sliders
        slider_ids, slider_float_ids, _y_off = self.sliders(
            x, y + y_off, parameters
        )
        y_off += _y_off + 25
        # [pak $1 $2 $3 ...]
        pack_id = self.object(
            "pak " + self._pack_args(parameters), len(parameters) + 1, 1, x, y + y_off
        )
        pack_width = self.get_box_by_id(pack_id)["box"]["patching_rect"][2]
        # [msg /path $1 $2 $3 ...]
        y_off += 25
        msg_id = self.message(path + " " + self._msg_args(parameters), x, y + y_off)
        # [s send]
        y_off += 25
        send_id = self.object("s send", 1, 0, x, y + y_off)
        # connections
        [
            self.connect(receive_ids[i], 0, slider_ids[i], 0)
            for i in range(len(parameters))
        ]
        [
            self.connect(slider_ids[i], 0, slider_float_ids[i], 0)
            for i in range(len(parameters))
        ]
        [
            self.connect(slider_float_ids[i], 0, pack_id, i)
            for i in range(len(parameters))
        ]
        self.connect(pack_id, 0, msg_id, 0)
        self.connect(msg_id, 0, send_id, 0)
        return slider_ids, pack_id, msg_id

    def osc_receive_with_controls(self, x, y, path, parameters):
        # TODO: add default param value and a loadbang
        """
        [comment path]
        [r receive]
        |
        [route /path]
        |                  |
        [unpack f f f ...] [print /path]
        |
        [slider] ...
        |
        [number] ...
        |
        [s arg_name]
        [comment path_arg_name]
        [comment type min-max]
        """
        # [comment path]
        y_off = 0
        path_comment_id = self.comment(path, x, y + y_off)

        # [r receive]
        y_off += 25
        receive_id = self.object("r receive", 0, 1, x, y + y_off)

        # [route /path]
        y_off += 25
        route_id = self.object("route " + path, 1, 1, x, y + y_off)

        # [unpack f f f ...] [print /path]
        y_off += 25
        unpack_id = self.object(
            "unpack " + self._pack_args(parameters),
            len(parameters) + 1,
            1,
            x,
            y + y_off,
        )
        unpack_width = self.get_box_by_id(unpack_id)["box"]["patching_rect"][2]
        print_id = self.object(
            "print " + path, 1, 0, x + unpack_width + 10, y + y_off
        )

        # sliders
        y_off += 10
        slider_ids, float_ids, _y_off = self.sliders(x, y + y_off, parameters)

        # [s arg_name]
        y_off += _y_off + 25
        send_ids = [
            self.object(
                "s " + path.replace("/", "_")[1:] + "_" + p["label"][0:3],
                1,
                0,
                x + i * 52.0,
                y + y_off + (0 if i % 2 == 0 else 25),
            )
            for i, p in enumerate(parameters)
        ]

        # [comment params]
        y_off += 50
        param_comment_ids, _y_off = self.param_comments(x, y + y_off, parameters)

        # connections
        self.connect(receive_id, 0, route_id, 0)
        self.connect(route_id, 0, unpack_id, 0)
        self.connect(route_id, 0, print_id, 0)
        [self.connect(unpack_id, i, slider_ids[i], 0) for i in range(len(parameters))]
        [
            self.connect(slider_ids[i], 0, float_ids[i], 0)
            for i in range(len(parameters))
        ]
        [self.connect(float_ids[i], 0, send_ids[i], 0) for i in range(len(parameters))]

        return slider_ids, unpack_id

    """
    sliders
    """

    def sliders(self, x, y, sliders):
        """
        sliders = [
          { 'label': 'x', data: 'float', min_val: 0.0, size: 0.0 },
        ]

        [slider] ...
        |
        [number] ...
        """
        slider_ids = []
        float_ids = []
        y_off = 0
        for i, s in enumerate(sliders):
            y_off = 0
            x_i = x + (i * 52.0)
            y_off += self.h
            slider_id = self.slider(
                x_i, y + y_off, s["min_val"], s["size"], float=s["data"] == "float"
            )
            y_off += 150
            float_id = self.box("float", 1, 2, x_i, y + y_off, 50)
            slider_ids.append(slider_id)
            float_ids.append(float_id)
        return slider_ids, float_ids, y_off

    """
    comments
    """

    def param_comments(self, x, y, params):
        comment_ids = []
        y_off = 0
        for i, p in enumerate(params):
            y_off = 0
            x_i = x + (i * 52.0)
            p_max = (
                p["min_val"] + p["size"]
                if p["data"] == "float"
                else p["min_val"] + p["size"] - 1
            )
            comment_id1 = self.comment(f'{p["label"]}', x_i, y)
            y_off += 15
            comment_id2 = self.comment(
                f'{p["data"][0]} {p["min_val"]}-{p_max}', x_i, y + y_off
            )
            comment_ids.append(comment_id1)
            comment_ids.append(comment_id2)
        return comment_ids, y_off

    """
    lists
    """

    def osc_send_list(self, x, y, path, params):
        """
        [comment] path, list name, params
        [r] path
        [prepend path]
        [s send]
        """
        y_off = 0
        self.comment(path, x, y)
        y_off += 15
        l = list(params.items())[0]
        self.comment(f"{l[0]}", x, y + y_off)
        y_off += 15
        self.comment(f"l {l[1][1]} {l[1][2]}", x, y + y_off)
        y_off += self.h
        receive_id = self.object(f"r {self.path_to_snakecase(path)}", 0, 1, x, y + y_off)
        y_off += self.h + 3
        prepend_id = self.object(f"prepend {path}", 1, 1, x, y + y_off)
        y_off += self.h + 3
        send_id = self.object(f"s send", 0, 1, x, y + y_off)
        self.connect(receive_id, 0, prepend_id, 0)
        self.connect(prepend_id, 0, send_id, 0)

    def osc_receive_list(self, x, y, path, params):
        """
        [comment] path
        [r receive.from.iipyper]
        [routeOSC path]
        [s path]
        [comment] params
        """
        # y_off = 0
        # self.comment(path, x, y)
        # y_off += self.h
        # receive_id = self.object(f"r receive.from.iipyper", x, y + y_off)
        # y_off += self.h
        # route_id = self.object(f"routeOSC {path}", x, y + y_off)
        # y_off += self.h
        # send_id = self.object(f"s {self.path_to_snakecase(path)}", x, y + y_off)
        # y_off += self.h
        # l = list(params.items())[0]
        # self.comment(f"{l[0]}", x, y + y_off)
        # y_off += 15
        # self.comment(f"l {l[1][1]} {l[1][2]}", x, y + y_off)
        # self.connect(receive_id, 0, route_id, 0)
        # self.connect(route_id, 0, send_id, 0)
        pass

    """
    utils
    """

    def get_box_by_id(self, id):
        for box in self.patch["patcher"]["boxes"]:
            if self.id_from_str(box["box"]["id"]) == id:
                return box
        return None

    def str_from_id(self, id):
        return "obj-" + str(id)

    def id_from_str(self, obj_str):
        return int(obj_str[4:])

    def _msg_args(self, args):
        return " ".join(["$" + str(i + 1) for i in range(len(args))])

    def _pack_args(self, args):
        arg_types = []
        for a in args:
            match a["data"]:
                case "int":
                    arg_types.append("i")
                case "float":
                    arg_types.append("f")
                case "string":
                    arg_types.append("s")
        return " ".join(arg_types)

    def path_to_snakecase(self, path):
        return path.replace("/", "_")[1:]  # +'_'+label[0:3]

    """
    save/load
    """

    def save(self, name):
        with open(name + ".maxpat", "w") as f:
            f.write(json.dumps(self.patch, indent=2))

    def load(self, name):
        with open(name + ".maxpat", "r") as f:
            self.patch = json.loads(f.read())
