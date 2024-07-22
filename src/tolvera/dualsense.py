"""
TODO: SignalFlow microphone and speaker demo
TODO: List available output methods for LEDs, rumble, etc.
"""

import os
import time
import threading

if os.uname().sysname == 'Darwin':
    # Setting up the environment variable for the library path
    homebrew_lib_path = "/opt/homebrew/lib"
    current_dyld_library_path = os.environ.get('DYLD_LIBRARY_PATH', '')
    os.environ['DYLD_LIBRARY_PATH'] = f"{homebrew_lib_path}:{current_dyld_library_path}"

from dualsense_controller import DualSenseController, Mapping, UpdateLevel

DEFAULTS = {
    'device_index_or_device_info': 0,
    'mapping': Mapping.NORMALIZED,
    'left_joystick_deadzone': 0.2,
    'right_joystick_deadzone': 0.2,
    'left_trigger_deadzone': 0.05,
    'right_trigger_deadzone': 0.05,
    'mapping': Mapping.RAW,
    'left_joystick_deadzone': 5,
    'right_joystick_deadzone': 5,
    'left_trigger_deadzone': 1,
    'right_trigger_deadzone': 1,
    'gyroscope_threshold': 0,
    'accelerometer_threshold': 0,
    'orientation_threshold': 0,
    'update_level': UpdateLevel.PAINSTAKING,
    'update_level': UpdateLevel.HAENGBLIEM,
    'update_level': UpdateLevel.DEFAULT,
    'microphone_initially_muted': True,
    'microphone_invert_led': False,
}

CONTROLLER_INPUTS = [
    'connection',
    'battery',
    'btn_ps',
    'btn_options',
    'btn_create',
    'btn_mute',
    'btn_touchpad',
    'btn_triangle',
    'btn_cross',
    'btn_circle',
    'btn_square',
    'btn_left',
    'btn_up',
    'btn_right',
    'btn_down',
    'btn_l1',
    'btn_r1',
    'btn_l2',
    'btn_r2',
    'btn_r3',
    'btn_l3',
    'left_trigger',
    'right_trigger',
    'left_stick',
    'right_stick',
    'touch_finger_1',
    'touch_finger_2',
    'gyroscope',
    'accelerometer',
    'orientation',
]

class DualSense:
    def __init__(self, **kwargs):
        self.device_infos = DualSenseController.enumerate_devices()
        self.is_running = False
        print(f"[DualSense] Device Infos: {self.device_infos}")
        self.init(**kwargs)

    def init(self, **kwargs):
        self.device_infos = DualSenseController.enumerate_devices()
        if len(self.device_infos) < 1:
            raise Exception('[DualSense.init] No DualSense Controller available.')
        print(f"[DualSense.init] Device Infos: {self.device_infos}")
        config = DEFAULTS
        config.update(kwargs)
        self.controller = DualSenseController(**config)
        self.is_running = False

    def start(self):
        if hasattr(self, 'update_thread') and self.update_thread is not None:
            raise Exception('[DualSense.start] Thread already exists.')
        self.update_thread = threading.Thread(target=self.update)
        self.update_thread.start()
        self.is_running = True
        print(f"[DualSense.start] Thread started.")

    def stop(self):
        if self.update_thread is None:
            return
        self.is_running = False
        self.update_thread.join()
        self.update_thread = None
        print(f"[DualSense.stop] Thread stopped.")

    def update(self):
        self.controller.exceptions.on_change(self.on_exception)
        self.controller.activate()
        while self.is_running:
            time.sleep(1)
        self.controller.deactivate()

    def on_exception(self, exception: Exception):
        print(f'[DualSense.on_exception] Exception occured:', exception)
        self.stop()

    def handle(self, event_type: str):
        """Decorator for handling DualSense events.
        It has a 'type' argument where you specify the event type.
        For event types, see:
        https://github.com/yesbotics/dualsense-controller-python/blob/main/src/examples/example.py
        https://github.com/yesbotics/dualsense-controller-python/blob/main/src/dualsense_controller/core/state/read_state/value_type.py
        To list available inputs, use `list_inputs()`.

        Args:
            event_type (str): The event type.

        Example:
            ```py
            @ds.handle('left_stick')
            def _(left_stick):
                print(left_stick) # JoyStick(x=0.0, y=0.0)
            ```
        """
        def decorator(func):
            if hasattr(self.controller, event_type):
                getattr(self.controller, event_type).on_change(func)
            return func
        return decorator

    def list_defaults(self):
        print(f"[DualSense] Default config:\n{DEFAULTS}")
        return DEFAULTS

    def list_inputs(self):
        print(f"[DualSense] Available inputs:")
        [print(f"  - {i}") for i in CONTROLLER_INPUTS]
        return CONTROLLER_INPUTS
