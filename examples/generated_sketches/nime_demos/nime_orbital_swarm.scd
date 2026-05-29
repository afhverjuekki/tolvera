// Orbital Swarm — SuperCollider companion
// ========================================
// Three glassy pad voices, one per species, driven by nime_orbital_swarm.py over
// UDP 5000. As each species orbits and streams, its voice drifts in pitch,
// opens in brightness, and swells with orbital energy.
//
//   /metrics/<s>/x    centroid x  -> pitch
//   /metrics/<s>/y    centroid y  -> filter cutoff
//   /metrics/<s>/vel  mean speed  -> amplitude

s.options.numInputBusChannels = 0;

s.waitForBoot {
    s.latency = 0.05;
    thisProcess.openUDPPort(5000);

    SynthDef(\orbitVoice, {
        arg freq = 220, amp = 0.0, cutoff = 2000, detune = 0.008;
        var f = Lag.kr(freq, 0.18);
        var a = Lag.kr(amp.clip(0, 1), 0.12);
        var c = Lag.kr(cutoff.clip(80, 14000), 0.14);
        var sig;
        sig = Mix([
            SinOsc.ar(f),
            SinOsc.ar(f * (1 + detune)) * 0.7,
            SinOsc.ar(f * 2.01) * 0.25,
            Saw.ar(f * 0.5) * 0.2
        ]) * 0.3;
        sig = LPF.ar(sig, c);
        sig = sig * a;
        sig = sig + FreeVerb.ar(sig, mix: 0.4, room: 0.85, damp: 0.25);
        Out.ar(0, sig ! 2);
    }).add;

    s.sync;

    ~orbit = ();
    ~base = [110.0, 164.0, 220.0];   // triad
    3.do { |i| ~orbit[i] = Synth(\orbitVoice, [\freq, ~base[i]]) };

    3.do { |i|
        OSCdef(("orb_x" ++ i).asSymbol, { |msg|
            ~orbit[i].set(\freq, msg[1].linexp(0, 1, ~base[i] * 0.8, ~base[i] * 2.2));
        }, "/metrics/" ++ i ++ "/x", nil, 5000);

        OSCdef(("orb_y" ++ i).asSymbol, { |msg|
            ~orbit[i].set(\cutoff, msg[1].linexp(0, 1, 400.0, 7500.0));
        }, "/metrics/" ++ i ++ "/y", nil, 5000);

        OSCdef(("orb_v" ++ i).asSymbol, { |msg|
            ~orbit[i].set(\amp, msg[1].linlin(0, 1, 0.05, 0.45));
        }, "/metrics/" ++ i ++ "/vel", nil, 5000);
    };

    "Orbital Swarm companion ready. Listening on UDP port 5000.".postln;
};

ServerQuit.add({
    OSCdef.freeAll;
    ~orbit.do { |v| if(v.notNil) { v.free } };
});