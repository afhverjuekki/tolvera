r"""Subprocess manager for the SuperCollider language interpreter (sclang).

The Textual UI uses this to boot a single long-lived sclang process, evaluate
the user's `.scd` companion patch by asking sclang to `.load` the file, and
shut down cleanly on exit.

Why file-load instead of piping code over stdin
-----------------------------------------------
The patches we generate are multi-kilobyte and contain `\identifier` symbols
(`\freq`, `\amp`, `\species0`). Piping them directly through stdin requires
careful escaping and can be brittle across sclang versions. Asking sclang to
load the patch from disk avoids all of that: we write the buffer, send a
one-liner `("/abs/path.scd").load;` followed by the form-feed evaluator, and
let sclang parse the file itself. The form-feed character (`\x0c`) is sclang's
"evaluate the preceding block" trigger when reading piped stdin — the same
keystroke the SuperCollider IDE sends on Ctrl-Enter.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import sys
from enum import Enum
from pathlib import Path
from typing import Awaitable, Callable, Optional


# sclang's interactive REPL emits ANSI cursor/screen-control escapes (e.g.
# `\x1b[H\x1b[2J` to redraw its prompt). If those reach the Textual TextArea
# and then get rendered to the terminal, they execute as real escape codes
# and blank the entire TUI. Strip them at the source.
_ANSI_ESCAPE_RE = re.compile(
    r"\x1b"            # ESC
    r"(?:"
    r"\[[0-?]*[ -/]*[@-~]"  # CSI sequences (cursor/colour/erase)
    r"|\][^\x07]*\x07"      # OSC sequences terminated by BEL
    r"|[@-Z\\-_]"           # 2-byte ESC sequences
    r")"
)


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", text)


# macOS often installs sclang inside the SuperCollider.app bundle rather than
# on PATH. Try the bundle path as a fallback so the user doesn't have to
# tweak their shell setup just to hear the synths.
_MACOS_BUNDLE_SCLANG = "/Applications/SuperCollider.app/Contents/MacOS/sclang"


def _resolve_sclang_path() -> Optional[str]:
    """Find sclang on PATH or in the macOS app bundle. ``None`` if not found."""
    on_path = shutil.which("sclang")
    if on_path:
        return on_path
    if os.path.exists(_MACOS_BUNDLE_SCLANG) and os.access(_MACOS_BUNDLE_SCLANG, os.X_OK):
        return _MACOS_BUNDLE_SCLANG
    return None


class SclangStatus(str, Enum):
    STOPPED = "stopped"
    BOOTING = "booting"
    RUNNING = "running"
    ERROR = "error"
    NOT_INSTALLED = "not_installed"


# Form-feed terminates a code block in piped sclang stdin (analogous to
# pressing Ctrl-Enter in the IDE). Without this, sclang buffers the input
# and never executes it.
_EVAL_TERMINATOR = b"\x0c\n"

# Graceful shutdown command. ``Server.killAll`` releases scsynth nodes,
# ``0.exit`` quits sclang cleanly with exit code 0.
_SHUTDOWN_COMMAND = b"Server.killAll; 0.exit;\n"


LogCallback = Callable[[str], None]


class SclangRunner:
    """Manage a single sclang subprocess with boot / eval / stop lifecycle."""

    def __init__(self, log_callback: Optional[LogCallback] = None):
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._status: SclangStatus = SclangStatus.STOPPED
        self._log = log_callback or (lambda msg: None)
        self._reader_tasks: list[asyncio.Task] = []
        self._status_callback: Optional[Callable[[SclangStatus], None]] = None
        # When True, the stream drainer drops noisy shutdown output (sclang
        # prints "Exiting sclang (ctrl-D)" and the "sc3>" prompt many times
        # while it's tearing down after `0.exit`).
        self._suppress_shutdown_noise = False

    @property
    def status(self) -> SclangStatus:
        return self._status

    @property
    def is_running(self) -> bool:
        return self._status == SclangStatus.RUNNING and self._proc is not None

    def set_status_callback(self, cb: Callable[[SclangStatus], None]) -> None:
        """Register a callback fired every time status changes."""
        self._status_callback = cb

    def _set_status(self, status: SclangStatus) -> None:
        self._status = status
        if self._status_callback is not None:
            try:
                self._status_callback(status)
            except Exception:
                pass

    async def boot(self) -> bool:
        """Boot sclang. Returns True if running, False on failure.

        Idempotent: if already running, returns True immediately. If sclang
        is not on PATH, sets status to ``NOT_INSTALLED`` and returns False
        without raising — the UI surfaces the install hint instead.
        """
        if self.is_running:
            return True

        self._set_status(SclangStatus.BOOTING)
        self._log("Booting sclang...")

        sclang_path = _resolve_sclang_path()
        if sclang_path is None:
            self._set_status(SclangStatus.NOT_INSTALLED)
            self._log(
                "sclang not found. Install SuperCollider "
                "(macOS: `brew install --cask supercollider`) or ensure sclang "
                "is on PATH."
            )
            return False

        env = {**os.environ}
        # On Linux, sclang's Qt build supports an "offscreen" platform plugin
        # that suppresses any GUI surface. macOS sclang only ships the
        # "cocoa" plugin — setting QT_QPA_PLATFORM=offscreen there causes it
        # to abort on launch. So only apply the hint on Linux.
        if sys.platform.startswith("linux"):
            env.setdefault("QT_QPA_PLATFORM", "offscreen")

        try:
            self._proc = await asyncio.create_subprocess_exec(
                sclang_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            self._log(f"sclang process started ({sclang_path})")
        except FileNotFoundError:
            # Race: file disappeared between resolve and exec.
            self._set_status(SclangStatus.NOT_INSTALLED)
            self._log(f"sclang executable vanished at {sclang_path}")
            return False
        except Exception as exc:
            self._set_status(SclangStatus.ERROR)
            self._log(f"sclang failed to start: {exc}")
            return False

        # Spawn drainers so stdout/stderr never block the subprocess buffer
        # and so the user sees the SC post window in the UI status log.
        self._reader_tasks = [
            asyncio.create_task(self._drain_stream(self._proc.stdout, "SC")),
            asyncio.create_task(self._drain_stream(self._proc.stderr, "SC ERR")),
        ]

        self._set_status(SclangStatus.RUNNING)
        self._log("sclang booted.")
        return True

    async def eval_file(self, path: Path) -> bool:
        """Ask the running sclang to load and evaluate a .scd file from disk."""
        if not self.is_running or self._proc is None or self._proc.stdin is None:
            self._log("sclang is not running. Boot it first.")
            return False

        abs_path = str(Path(path).resolve())
        # Quote with SC string syntax (escape backslashes and quotes).
        sc_string = abs_path.replace("\\", "\\\\").replace('"', '\\"')
        command = f'("{sc_string}").load;\n'.encode("utf-8")

        try:
            self._proc.stdin.write(command + _EVAL_TERMINATOR)
            await self._proc.stdin.drain()
            self._log(f"sclang loaded {abs_path}")
            return True
        except Exception as exc:
            self._log(f"Failed to send eval command to sclang: {exc}")
            return False

    async def stop(self, timeout: float = 3.0) -> None:
        """Stop sclang gracefully, then forcibly if it does not exit in time."""
        if self._proc is None:
            self._set_status(SclangStatus.STOPPED)
            return

        # Switch the stream drainer into "suppress shutdown noise" mode so
        # the log isn't flooded with sclang's farewell chatter.
        self._suppress_shutdown_noise = True

        # Try graceful shutdown first.
        try:
            if self._proc.stdin is not None and not self._proc.stdin.is_closing():
                self._proc.stdin.write(_SHUTDOWN_COMMAND + _EVAL_TERMINATOR)
                await self._proc.stdin.drain()
                self._proc.stdin.close()
        except Exception:
            pass

        try:
            await asyncio.wait_for(self._proc.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            self._log("sclang did not exit in time; terminating.")
            try:
                self._proc.terminate()
                await asyncio.wait_for(self._proc.wait(), timeout=1.0)
            except (asyncio.TimeoutError, ProcessLookupError):
                try:
                    self._proc.kill()
                    await self._proc.wait()
                except ProcessLookupError:
                    pass
        except Exception as exc:
            self._log(f"Error stopping sclang: {exc}")

        for task in self._reader_tasks:
            task.cancel()
        self._reader_tasks.clear()

        self._proc = None
        self._set_status(SclangStatus.STOPPED)
        self._log("sclang stopped.")

    async def _drain_stream(self, stream: Optional[asyncio.StreamReader], prefix: str) -> None:
        if stream is None:
            return
        # Lines we never want to surface to the user — sclang spams these
        # during shutdown as it re-reads its closed stdin.
        SHUTDOWN_NOISE = ("Exiting sclang (ctrl-D)", "sc3>")
        while True:
            try:
                line = await stream.readline()
            except asyncio.CancelledError:
                raise
            except Exception:
                return
            if not line:
                # Stream closed. If we were running, sclang died.
                if self._status == SclangStatus.RUNNING:
                    self._set_status(SclangStatus.STOPPED)
                    self._log(f"{prefix}: process exited.")
                return
            text = _strip_ansi(line.decode("utf-8", errors="replace")).rstrip()
            if not text:
                continue
            if self._suppress_shutdown_noise and any(n in text for n in SHUTDOWN_NOISE):
                continue
            self._log(f"{prefix}: {text}")
