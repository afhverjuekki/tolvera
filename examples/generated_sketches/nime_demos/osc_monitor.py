"""Bind an OSC server to UDP 5000 (in place of sclang), launch a sketch, and
report which /metrics messages actually arrive and their value ranges."""
import subprocess, sys, threading, time
from collections import defaultdict
from pathlib import Path
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer

REPO = Path("/Users/mclemens/Development/tolvera")
DEMO_DIR = Path(__file__).resolve().parent           # demos live next to this script
_arg = sys.argv[1] if len(sys.argv) > 1 else "nime_flocking_timbres.py"
sketch = Path(_arg) if Path(_arg).exists() else DEMO_DIR / _arg
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 5000  # isolated port avoids a running demo on 5000
SECONDS = 12

stats = defaultdict(lambda: {"n": 0, "min": 1e9, "max": -1e9, "last": None})


def handle(addr, *args):
    v = args[0] if args else None
    s = stats[addr]
    s["n"] += 1
    if isinstance(v, (int, float)):
        s["min"] = min(s["min"], v); s["max"] = max(s["max"], v); s["last"] = v


disp = Dispatcher()
disp.set_default_handler(handle)
server = ThreadingOSCUDPServer(("127.0.0.1", PORT), disp)
t = threading.Thread(target=server.serve_forever, daemon=True)
t.start()
print(f"OSC monitor listening on 127.0.0.1:{PORT}; launching {sketch.name} for {SECONDS}s ...", flush=True)

proc = subprocess.Popen(
    [sys.executable, str(sketch), "--headless=True",
     f"--send_port={PORT}", f"--receive_port={PORT + 1}"],
    cwd=str(REPO), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(SECONDS)
proc.terminate()
try:
    proc.wait(timeout=5)
except subprocess.TimeoutExpired:
    proc.kill()
server.shutdown()

print(f"\n=== OSC received: {len(stats)} distinct addresses ===")
total = 0
for addr in sorted(stats):
    s = stats[addr]; total += s["n"]
    rng = f"min={s['min']:.3f} max={s['max']:.3f} last={s['last']}" if s["last"] is not None else "(no numeric)"
    print(f"  {addr:24s} n={s['n']:4d}  {rng}")
print(f"total messages: {total}")
if total == 0:
    print("RESULT: NO OSC RECEIVED (sketch is not emitting / not reaching 5000)")
else:
    vel = [a for a in stats if a.endswith('/vel')]
    nonzero = any(stats[a]['max'] > 0.02 for a in vel)
    print(f"RESULT: OSC FLOWING; velocity(amp) above floor: {nonzero}")
