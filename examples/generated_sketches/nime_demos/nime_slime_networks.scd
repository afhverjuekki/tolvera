// Slime Networks — SuperCollider companion
// =========================================
// Three warm voices, one per pheromone species, driven by nime_slime_networks.py
// over UDP 5000. The network is always crawling, so every voice always has some
// motion; a voice swells when its species is actively reaching and rerouting.
//
//   /metrics/<s>/x    centroid x  -> pitch (each species in its own register)
//   /metrics/<s>/y    centroid y  -> filter cutoff (brightness)
//   /metrics/<s>/vel  mean crawl  -> amplitude (with a floor, so always audible)

s.options.numInputBusChannels = 0;

s.waitForBoot {
    s.latency = 0.05;
    thisProcess.openUDPPort(5000);

    // A warm, slightly detuned pad voice through a resonant low-pass and reverb.
    SynthDef(\slimeVoice, {
        arg freq = 220, amp = 0.0, cutoff = 1800, detune = 0.01, room = 0.6;
        var f = Lag.kr(freq, 0.15);
        var a = Lag.kr(amp.clip(0, 1), 0.12);
        var c = Lag.kr(cutoff.clip(80, 14000), 0.12);
        var sig;
        sig = Mix([
            Saw.ar(f * (1 - detune)),
            Saw.ar(f * (1 + detune)),
            SinOsc.ar(f * 0.5) * 0.6,         // sub for warmth
            SinOsc.ar(f * 2) * 0.2            // gentle shimmer
        ]) * 0.25;
        sig = RLPF.ar(sig, c, 0.5);
        sig = sig * a;
        sig = FreeVerb.ar(sig, mix: 0.35, room: room, damp: 0.4);
        Out.ar(0, sig ! 2);
    }).add;

    s.sync;

    ~slime = ();
    // Three registers so the species stack into a chord rather than crowd one octave.
    ~base = [55.0, 110.0, 220.0];     // low, mid, high roots
    3.do { |i| ~slime[i] = Synth(\slimeVoice, [\freq, ~base[i]]) };

    3.do { |i|
        OSCdef(("slime_x" ++ i).asSymbol, { |msg|
            // x sweeps the voice across a fifth+octave around its register root
            ~slime[i].set(\freq, msg[1].linexp(0, 1, ~base[i], ~base[i] * 3.0));
        }, "/metrics/" ++ i ++ "/x", nil, 5000);

        OSCdef(("slime_y" ++ i).asSymbol, { |msg|
            ~slime[i].set(\cutoff, msg[1].linexp(0, 1, 350.0, 7000.0));
        }, "/metrics/" ++ i ++ "/y", nil, 5000);

        OSCdef(("slime_v" ++ i).asSymbol, { |msg|
            // crawl -> amplitude, with a floor so the network is always heard
            ~slime[i].set(\amp, msg[1].linlin(0, 1, 0.06, 0.5));
        }, "/metrics/" ++ i ++ "/vel", nil, 5000);
    };

    "Slime Networks companion ready. Listening on UDP port 5000.".postln;
};

ServerQuit.add({
    OSCdef.freeAll;
    ~slime.do { |v| if(v.notNil) { v.free } };
});
