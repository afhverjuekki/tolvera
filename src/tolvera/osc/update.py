class Updater:
    """
    Rate-limited function call
    """

    def __init__(self, f, count: int = 1):
        self.f = f
        self.count = int(count)
        self.counter = 0

    def __call__(self, *args, **kwargs):
        self.counter += 1
        if self.counter >= self.count:
            self.counter = 0
            return self.f(*args, **kwargs)
        return None


class ReceiveUpdater:
    """
    Decouples event handling from updating
    Updating is rate-limited by a counter
    TODO: Rename to ReceiveArgsUpdater?
    """

    def __init__(self, f, state=None, count=5, update=False):
        self.f = f
        self.count = count
        self.counter = 0
        self.update = update
        self.state = state

    def set(self, state):
        """
        Set the Updater's state
        """
        self.state = state
        self.update = True

    def __call__(self):
        """
        Update the target function with internal state
        """
        self.counter += 1
        if not (self.update and self.counter > self.count and self.state is not None):
            return
        self.ret = self.f(*self.state)
        """
        if ret is not None:
            route = self.pascal_to_path(kwargs['name'])
            print('wrapper', route, ret, self.client_name)
            self.osc.return_to_sender_by_name((route, ret), self.client_name)
        """
        self.counter = 0
        self.update = False
        return self.ret


class ReceiveListUpdater:
    """
    Decouples event handling from updating
    Updating is rate-limited by a counter
    Assumes a list[float] instead of *args
    """

    def __init__(self, f, state=None, count=5, update=False):
        self.f = f
        self.count = count
        self.counter = 0
        self.update = update
        self.state = state

    def set(self, state):
        """
        Set the Updater's state
        """
        self.state = state
        self.update = True

    def __call__(self):
        """
        Update the target function with internal state
        """
        self.counter += 1
        if not (self.update and self.counter > self.count and self.state is not None):
            return
        self.f(self.state)
        self.counter = 0
        self.update = False


class OSCReceiveUpdater(ReceiveUpdater):
    """
    ReceiveUpdater with an OSC handler
    """

    def __init__(self, osc, address: str, f, state=None, count=10, update=False):
        super().__init__(f, state, count, update)
        self.osc = osc
        self.address = address
        osc.add_handler(self.address, self.receive)

    def receive(self, address, *args):
        # FIXME: ip:port/args
        """
        v: first argument to the handler is the IP:port of the sender
        v: or you can use dispatcher.map directly
           and not set needs_reply_address=True
        j: can I get ip:port from osc itself?
        v: if you know the sender ahead of time yeah,
           but that lets you respond to different senders dynamically
        """
        self.set(args[1:])


class OSCReceiveListUpdater(ReceiveListUpdater):
    """
    ReceiveListUpdater with an OSC handler
    """

    def __init__(self, osc, address: str, f, state=None, count=10, update=False):
        super().__init__(f, state, count, update)
        self.osc = osc
        self.address = address
        osc.add_handler(self.address, self.receive)

    def receive(self, address, *args):
        self.set(list(args[1:]))


class OSCSend:
    """
    Non rate-limited OSC send
    """

    def __init__(self, osc, address: str, f, client=None):
        self.osc = osc
        self.address = address
        self.f = f
        self.client = client

    def __call__(self, *args):
        self.osc.send(self.address, *self.f(*args), client=self.client)


class OSCSendUpdater:
    """
    Rate-limited OSC send
    """

    def __init__(self, osc, address: str, f, count=30, client=None):
        self.osc = osc
        self.address = address
        self.f = f
        self.count = count
        self.counter = 0
        self.client = client

    def __call__(self):
        self.counter += 1
        if self.counter >= self.count:
            self.osc.send(self.address, *self.f(), client=self.client)
            self.counter = 0


class OSCReceiveUpdaters:
    """
    o = OSCReceiveUpdaters(osc,
        {"/tolvera/particles/pos": s.osc_set_pos,
         "/tolvera/particles/vel": s.osc_set_vel})
    """

    def __init__(self, osc, receives=None, count=10):
        self.osc = osc
        self.receives = []
        self.count = count
        if receives is not None:
            self.add_dict(receives, count=self.count)

    def add_dict(self, receives, count=None):
        if count is None:
            count = self.count
        {a: self.add(a, f, count=count) for a, f in receives.items()}

    def add(self, address, function, state=None, count=None, update=False):
        if count is None:
            count = self.count
        self.receives.append(
            OSCReceiveUpdater(self.osc, address, function, state, count, update)
        )

    def __call__(self):
        [r() for r in self.receives]


class OSCSendUpdaters:
    """
    o = OSCSendUpdaters(osc, client="particles", count=10,
        sends={
            "/tolvera/particles/get/pos/all": s.osc_get_pos_all
        })
    """

    def __init__(self, osc, sends=None, count=10, client=None):
        self.osc = osc
        self.sends = []
        self.count = count
        self.client = client
        if sends is not None:
            self.add_dict(sends, self.count, self.client)

    def add_dict(self, sends, count=None, client=None):
        if count is None:
            count = self.count
        if client is None:
            client = self.client
        {a: self.add(a, f, count=count, client=client) for a, f in sends.items()}

    def add(self, address, function, state=None, count=None, update=False, client=None):
        if count is None:
            count = self.count
        if client is None:
            client = self.client
        self.sends.append(OSCSendUpdater(self.osc, address, function, count, client))

    def __call__(self):
        [s() for s in self.sends]


class OSCUpdaters:
    """
    o = OSCUpdaters(osc, client="boids", count=10,
        receives={
            "/tolvera/boids/pos": b.osc_set_pos,
            "/tolvera/boids/vel": b.osc_set_vel
        },
        sends={
            "/tolvera/boids/pos/all": b.osc_get_all_pos
        }
    )
    """

    def __init__(
        self,
        osc,
        sends=None,
        receives=None,
        send_count=60,
        receive_count=10,
        client=None,
    ):
        self.osc = osc
        self.client = client
        self.send_count = send_count
        self.receive_count = receive_count
        self.sends = OSCSendUpdaters(
            self.osc, count=self.send_count, client=self.client
        )
        self.receives = OSCReceiveUpdaters(self.osc, count=self.receive_count)
        if sends is not None:
            self.add_sends(sends)
        if receives is not None:
            self.add_receives(receives)

    def add_sends(self, sends, count=None, client=None):
        if count is None:
            count = self.send_count
        if client is None:
            client = self.client
        self.sends.add_dict(sends, count, client)

    def add_send(self, send, count=None, client=None):
        if count is None:
            count = self.send_count
        if client is None:
            client = self.client
        self.sends.add(send, client=client, count=count)

    def add_receives(self, receives, count=None):
        if count is None:
            count = self.receive_count
        self.receives.add_dict(receives, count=count)

    def add_receive(self, receive, count=None):
        if count is None:
            count = self.receive_count
        self.receives.add(receive, count=count)

    def __call__(self):
        self.sends()
        self.receives()
