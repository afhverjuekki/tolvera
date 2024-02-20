from iipyper import OSC as iiOSC

from .oscmap import OSCMap


class OSC:
    def __init__(self, context, **kwargs) -> None:
        self.ctx = context
        self.kwargs = kwargs
        self.osc = kwargs.get("osc", False)
        self.init_osc(**kwargs)
        self.init_map(**kwargs)
        print(f"[Tölvera.OSC] OSC initialization complete.")

    def init_osc(self, **kwargs):
        self.host_ip = kwargs.get("host", "127.0.0.1")
        self.client_ip = kwargs.get("client", "127.0.0.1")
        self.client_name = kwargs.get("client_name", self.ctx.name_clean)
        self.receive_port = kwargs.get("receive_port", 5001)
        self.send_port = kwargs.get("send_port", 5000)
        self.trace = kwargs.get("osc_trace", False)
        self.verbose = kwargs.get("osc_verbose", False)
        print(
            f"[Tölvera.OSC] Initialising OSC host '{self.host_ip}:{self.receive_port}'."
        )
        self.host = iiOSC(
            self.host_ip, self.receive_port, verbose=self.verbose, concurrent=True
        )
        print(
            f"[Tölvera.OSC] Initialising OSC client '{self.client_name}' at '{self.client_ip}:{self.send_port}'."
        )
        self.host.create_client(self.client_name, self.client_ip, self.send_port)
        if self.trace:

            def trace(address, *args):
                print(f"[Tölvera.OSC.trace] '{address}' {args}")

            self.host.args("/*")(trace)

    def init_map(self, **kwargs):
        print(f"[Tölvera.OSC] Initialising OSCMap for '{self.client_name}'.")
        self.create_patch = kwargs.get("create_patch", False)
        self.patch_type = kwargs.get("patch_type", "Pd")
        self.patch_filepath = kwargs.get("patch_filepath", self.client_name)
        self.export_patch = kwargs.get("export_patch", None)
        if self.create_patch:
            print(
                f"[Tölvera.OSC] Creating {self.patch_type} patch '{self.patch_filepath}'."
            )
        self.map = OSCMap(
            self.host,
            self.client_name,
            self.patch_type,
            self.patch_filepath,
            self.create_patch,
            export=self.export_patch,
        )
