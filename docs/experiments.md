---
title: Experiments
hide:
  - navigation
---

# Experiments

## Live Coding

Tölvera can be live coded via [Sardine](https://sardine.raphaelforment.fr/).
Note that this only works so far using VSCode's inbuilt Python REPL, and not with the Sardine VSCode extension:

```py
tv.s.params = {'state':{'evap': (ti.f32, 0., 0.99)}}

@swim
def gui_loop(p=0.5, i=0):
  tv.px.diffuse(tv.s.params.field[0].evap)
  tv.px.particles(tv.p, tv.s.species())
  tv.show(tv.px)
  again(gui_loop, p=1/64, i=i+1)

@swim
def control_loop(p=4, i=0):
  tv.s.params.field[0].evap = P('0.9 0.99 0.1', i)
  again(control_loop, p=1/2, i=i+1)
```

## Sonification

Sonification of and with Tölvera can be achieved via [SignalFlow](https://signalflow.dev). See examples [here](https://github.com/Intelligent-Instruments-Lab/iil-examples/tree/main/tolvera/signalflow).

## GPU Audio

GPU audio rendering can be achieved using Taichi alone:

```py
from iipyper import Audio, run
import sounddevice as sd
import taichi as ti
from math import pi

@ti.kernel
def generate_data(outdata: ti.ext_arr(), 
                  frames: ti.i32, 
                  start_idx: ti.template(), 
                  fs: ti.f32):
    for I in ti.grouped(ti.ndrange(frames)):
        i = I[0]
        t = (start_idx[None] + i) / fs
        s = ti.sin(2 * pi * 440 * t)
        start_idx[None] += 1
        outdata[i,0] = s

def main():
    ti.init(arch=ti.vulkan)
    device = sd.default.device
    fs = sd.query_devices(sd.default.device, 'output')['default_samplerate']
    buffer_size = 128
    start_idx = ti.field(ti.i32, ())
    def callback(indata, outdata, frames, time, status):
        if status: print(status, file=sys.stderr)
        generate_data(outdata, frames, start_idx, fs)
    Audio(device=device, channels=1, blocksize=buffer_size,
        samplerate=fs, callback=callback)

if __name__=='__main__':
    run(main)
```

## 3D Printing

Exploration of 3D printing with Tölvera can be achieved via [FullControl](https://fullcontrol.xyz). See example [here](https://github.com/Intelligent-Instruments-Lab/iil-examples/tree/main/tolvera/3dprinting).

