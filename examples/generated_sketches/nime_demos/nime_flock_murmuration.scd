// Flock Murmuration — SuperCollider companion
// ============================================
// Four voices, one per flock species, driven by nime_flock_murmuration.py over
// UDP 5000. The flock never stops moving, so every voice always sings; a voice
// swells when its flock is moving fast and brightens as it rises on screen.
//
//   /metrics/<s>/x    centroid x  -> pitch (each species in its own register)
//   /metrics/<s>/y    centroid y  -> filter cutoff
//   /metrics/<s>/vel  mean speed  -> amplitude (with a floor)

s.options.numInputBusChannels = 0;

s.waitForBoot {
    s.latency = 0.05;
    thisProcess.openUDPPort(5000);

    SynthDef(\flockVoice, {
        arg freq = 220, amp = 0.0, cutoff = 1800, detune = 0.012;
        var f = Lag.kr(freq, 0.12);
        var a = Lag.kr(amp.clip(0, 1), 0.1);
        var c = Lag.kr(cutoff.clip(80, 14000), 0.1);
        var sig;
        sig = Mix([
            Saw.ar(f * (1 - detune)),
            Saw.ar(f * (1 + detune)),
            Pulse.ar(f * 0.5, 0.4) * 0.5
        ]) * 0.22;
        sig = RLPF.ar(sig, c, 0.45);
        sig = sig * a;
        sig = sig + FreeVerb.ar(sig, mix: 0.3, room: 0.7, damp: 0.4);
        Out.ar(0, sig ! 2);
    }).add;

    s.sync;

    ~flock = ();
    // Four registers forming a stacked chord (root, fifth, octave, tenth).
    ~base = [98.0, 147.0, 196.0, 311.0];
    4.do { |i| ~flock[i] = Synth(\flockVoice, [\freq, ~base[i]]) };

    4.do { |i|
        OSCdef(("flk_x" ++ i).asSymbol, { |msg|
            ~flock[i].set(\freq, msg[1].linexp(0, 1, ~base[i] * 0.8, ~base[i] * 2.2));
        }, "/metrics/" ++ i ++ "/x", nil, 5000);

        OSCdef(("flk_y" ++ i).asSymbol, { |msg|
            ~flock[i].set(\cutoff, msg[1].linexp(0, 1, 400.0, 8000.0));
        }, "/metrics/" ++ i ++ "/y", nil, 5000);

        OSCdef(("flk_v" ++ i).asSymbol, { |msg|
            ~flock[i].set(\amp, msg[1].linlin(0, 1, 0.05, 0.45));
        }, "/metrics/" ++ i ++ "/vel", nil, 5000);
    };

    "Flock Murmuration companion ready. Listening on UDP port 5000.".postln;
};

ServerQuit.add({
    OSCdef.freeAll;
    ~flock.do { |v| if(v.notNil) { v.free } };
});
