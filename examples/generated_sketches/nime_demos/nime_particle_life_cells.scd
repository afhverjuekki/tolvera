// Particle Life: Cells — SuperCollider companion
// ===============================================
// Four FM-bell voices, one per species, driven by nime_particle_life_cells.py
// over UDP 5000. As cells form, chase and dissolve, the voices move in pitch,
// brighten, and swell with the motion of each species.
//
//   /metrics/<s>/x    centroid x  -> pitch
//   /metrics/<s>/y    centroid y  -> FM index (brightness)
//   /metrics/<s>/vel  mean speed  -> amplitude

s.options.numInputBusChannels = 0;

s.waitForBoot {
    s.latency = 0.05;
    thisProcess.openUDPPort(5000);

    SynthDef(\cellVoice, {
        arg freq = 220, amp = 0.0, fmindex = 2.0;
        var f = Lag.kr(freq, 0.1);
        var a = Lag.kr(amp.clip(0, 1), 0.1);
        var idx = Lag.kr(fmindex.clip(0.2, 12), 0.15);
        var mod = SinOsc.ar(f * 2.0) * f * idx;
        var sig = SinOsc.ar(f + mod);
        sig = sig * a * 0.3;
        sig = sig + FreeVerb.ar(sig, mix: 0.32, room: 0.75, damp: 0.35);
        Out.ar(0, sig ! 2);
    }).add;

    s.sync;

    ~cells = ();
    // A pentatonic-ish stack so overlapping voices stay consonant.
    ~base = [110.0, 165.0, 220.0, 330.0];
    4.do { |i| ~cells[i] = Synth(\cellVoice, [\freq, ~base[i]]) };

    4.do { |i|
        OSCdef(("cell_x" ++ i).asSymbol, { |msg|
            ~cells[i].set(\freq, msg[1].linexp(0, 1, ~base[i] * 0.85, ~base[i] * 2.0));
        }, "/metrics/" ++ i ++ "/x", nil, 5000);

        OSCdef(("cell_y" ++ i).asSymbol, { |msg|
            ~cells[i].set(\fmindex, msg[1].linexp(0, 1, 0.4, 9.0));
        }, "/metrics/" ++ i ++ "/y", nil, 5000);

        OSCdef(("cell_v" ++ i).asSymbol, { |msg|
            ~cells[i].set(\amp, msg[1].linlin(0, 1, 0.05, 0.45));
        }, "/metrics/" ++ i ++ "/vel", nil, 5000);
    };

    "Particle Life: Cells companion ready. Listening on UDP port 5000.".postln;
};

ServerQuit.add({
    OSCdef.freeAll;
    ~cells.do { |v| if(v.notNil) { v.free } };
});
