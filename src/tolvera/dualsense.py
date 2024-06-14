"""
TODO: @ti.dataclasses/tv.s for DualSense types
TODO: add onupdate callbacks - overridable?
TODO: listener pattern with Taichi...?
TODO: demos
TODO: init pattern/DEFAULTS
TODO: test start/stop/disconnect/reconnect polling etc.
TODO: test `time.sleep()` vs @tv.render
"""

import os
import time
import threading
import taichi as ti

if os.uname().sysname == 'Darwin':
    # Setting up the environment variable for the library path
    homebrew_lib_path = "/opt/homebrew/lib"
    current_dyld_library_path = os.environ.get('DYLD_LIBRARY_PATH', '')
    os.environ['DYLD_LIBRARY_PATH'] = f"{homebrew_lib_path}:{current_dyld_library_path}"

from dualsense_controller import Accelerometer, Battery, Benchmark, Connection, ConnectionType, DeviceInfo, \
    DualSenseController, \
    Gyroscope, \
    JoyStick, \
    Mapping, Number, \
    Orientation, TouchFinger, UpdateLevel

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

@ti.data_oriented
class DualSense:
    def __init__(self):
        self.device_infos = DualSenseController.enumerate_devices()
        print(f"[DualSense] Device Infos: {self.device_infos}")

    def init(self, **kwargs):
        self.device_infos = DualSenseController.enumerate_devices()
        if len(self.device_infos) < 1:
            raise Exception('No DualSense Controller available.')
        print(f"[DualSense.init] Device Infos: {self.device_infos}")
        config = DEFAULTS
        config.update(kwargs)
        self.controller = DualSenseController(**config)

    def update(self):
        self.controller.activate()
        while self.is_running:
            time.sleep(1)
        self.controller.deactivate()

    def start(self):
        if self.update_thread is not None:
            raise Exception('[DualSense.start] Thread already exists.')
        self.update_thread = threading.Thread(target=self.update)
        self.update_thread.start()

    def stop(self):
        if self.update_thread is None:
            return
        self.is_running = False
        self.update_thread.join()
        self.update_thread = None

