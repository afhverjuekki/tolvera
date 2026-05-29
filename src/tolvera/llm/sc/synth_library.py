"""SynthDef bodies for the Tölvera SuperCollider companion.

Each entry is (name, body). The body is inserted between the
`arg freq=440, amp=0.0, cutoff=2000;` declaration and `}).add;` produced
by `templates/supercollider/synthdef.scd.j2`. Bodies must end with `Out.ar`.

Lag smoothing is applied to control inputs because the OSC stream from
`src/tolvera/llm/context/exemplars/boids_osc.py` updates every ~2 frames
and would otherwise produce zipper noise when freq/amp/cutoff change.
"""

_SAW = """
    var f = Lag.kr(freq, 0.1);
    var a = Lag.kr(amp, 0.05);
    var c = Lag.kr(cutoff.clip(40, 18000), 0.1);
    var sig = Saw.ar(f);
    sig = RLPF.ar(sig, c, 0.5);
    sig = sig * a;
    Out.ar(0, sig ! 2);
"""

_FM = """
    var f = Lag.kr(freq, 0.1);
    var a = Lag.kr(amp, 0.05);
    var c = Lag.kr(cutoff.clip(40, 18000), 0.1);
    var modulator = SinOsc.ar(f * 1.5, 0, f * 2);
    var sig = SinOsc.ar(f + modulator);
    sig = LPF.ar(sig, c);
    sig = sig * a;
    Out.ar(0, sig ! 2);
"""

_PULSE = """
    var f = Lag.kr(freq, 0.1);
    var a = Lag.kr(amp, 0.05);
    var c = Lag.kr(cutoff.clip(40, 18000), 0.1);
    var width = SinOsc.kr(0.3).range(0.2, 0.8);
    var sig = Pulse.ar(f, width);
    sig = RLPF.ar(sig, c, 0.4);
    sig = sig * a;
    Out.ar(0, sig ! 2);
"""

_FM_BELL = """
    var f = Lag.kr(freq, 0.1);
    var a = Lag.kr(amp, 0.05);
    var c = Lag.kr(cutoff.clip(40, 18000), 0.1);
    var modIndex = Decay2.kr(Impulse.kr(0.5), 0.01, 2.0);
    var modulator = SinOsc.ar(f * 3.5, 0, f * 4 * modIndex);
    var sig = SinOsc.ar(f + modulator);
    sig = HPF.ar(sig, 100);
    sig = LPF.ar(sig, c);
    sig = sig * a;
    Out.ar(0, sig ! 2);
"""

_PLUCK = """
    var f = Lag.kr(freq, 0.1);
    var a = Lag.kr(amp, 0.05);
    var c = Lag.kr(cutoff.clip(40, 18000), 0.1);
    var trig = Impulse.kr(2);
    var exciter = WhiteNoise.ar(0.3) * Decay.kr(trig, 0.05);
    var sig = Pluck.ar(exciter, trig, 0.2, f.reciprocal, 4, 0.5);
    sig = LPF.ar(sig, c);
    sig = sig * a * 4;
    Out.ar(0, sig ! 2);
"""

_GRANULAR = """
    var f = Lag.kr(freq, 0.1);
    var a = Lag.kr(amp, 0.05);
    var c = Lag.kr(cutoff.clip(40, 18000), 0.1);
    var grainRate = 30;
    var trig = Impulse.kr(grainRate);
    var env = Decay2.kr(trig, 0.001, 0.04);
    var detune = LFNoise1.kr(grainRate).range(0.95, 1.05);
    var sig = SinOsc.ar(f * detune) * env;
    sig = LPF.ar(sig, c);
    sig = sig * a * 2;
    Out.ar(0, sig ! 2);
"""

SYNTH_CATALOG: list[tuple[str, str]] = [
    ("saw", _SAW),
    ("fm", _FM),
    ("pulse", _PULSE),
    ("fm_bell", _FM_BELL),
    ("pluck", _PLUCK),
    ("granular", _GRANULAR),
]
