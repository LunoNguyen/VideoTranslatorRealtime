"""Per-process audio capture via WASAPI process loopback.

Decision (spike, 2026-10-28): wrap PyPI ``proc-tap`` 1.1.1 (import name
``proctap``, MIT, prebuilt cp312 win_amd64 wheel -> no compiler for users).
Its native ``ProcessLoopback`` calls ``ActivateAudioInterfaceAsync`` with
``AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK`` +
``PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE`` and delivers 48 kHz
stereo float32 (int16 44.1 kHz fallback). Spike on Win11 26200: a 440 Hz tone
process captured at RMS 0.14 / FFT peak 440 Hz, a silent sibling process read
exactly 0 while the tone (and other apps) played, and children are included.
Fallbacks (ctypes/comtypes COM handler, C++ helper) were not needed.

Caveats handled here:
- proctap silently falls back to *system-wide* loopback when per-process
  activation fails; we check ``is_process_specific()`` and raise
  ``LoopbackUnsupportedError`` instead (never leak other apps' audio).
- The native object calls ``CoInitializeEx(STA)`` on its thread, so it is
  created and polled on a private worker thread.
- We use ``proctap._native`` directly (the high-level class hides the
  fallback flag); pinned to ``proc-tap>=1.1.1,<1.2``.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from typing import Protocol, runtime_checkable

import numpy as np

log = logging.getLogger(__name__)

MIN_WINDOWS_BUILD = 19041  # Windows 10 2004
POLL_S = 0.005
QUEUE_CHUNKS = 400  # ~4 s of 10 ms packets; oldest dropped on overflow


class LoopbackUnsupportedError(RuntimeError):
    """Per-process loopback capture is not available on this OS."""


@runtime_checkable
class AudioSource(Protocol):
    pid: int
    sample_rate: int
    channels: int

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def read(self, timeout: float) -> np.ndarray | None:
        """float32 array shape (n, channels), or None if nothing within timeout."""
        ...


def _check_os() -> None:
    if sys.platform != "win32":
        raise LoopbackUnsupportedError("Process loopback requires Windows")
    build = sys.getwindowsversion().build
    if build < MIN_WINDOWS_BUILD:
        raise LoopbackUnsupportedError(
            f"Windows build {build} lacks process loopback (need >= {MIN_WINDOWS_BUILD})"
        )


class ProcessLoopbackSource:
    """Captures audio rendered by ``pid`` (and its child processes)."""

    def __init__(self, pid: int, include_tree: bool = True) -> None:
        if not include_tree:
            raise NotImplementedError("proctap only supports INCLUDE_TARGET_PROCESS_TREE")
        self.pid = pid
        self.include_tree = include_tree
        self.sample_rate = 48000
        self.channels = 2
        self._dtype: type = np.float32
        self._queue: queue.Queue[np.ndarray] = queue.Queue(QUEUE_CHUNKS)
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Activate capture; raises LoopbackUnsupportedError/RuntimeError on failure."""
        if self._thread is not None:
            return
        _check_os()
        self._stop.clear()
        self._ready.clear()
        self._error = None
        self._thread = threading.Thread(
            target=self._run, name=f"loopback-{self.pid}", daemon=True
        )
        self._thread.start()
        self._ready.wait(15)
        if self._error is not None or not self._ready.is_set():
            err = self._error or RuntimeError("Process loopback activation timed out")
            self.stop()
            raise err

    @property
    def error(self) -> BaseException | None:
        """Exception that ended capture (activation or runtime), else None."""
        return self._error

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2)
            self._thread = None

    def read(self, timeout: float) -> np.ndarray | None:
        try:
            chunks = [self._queue.get(timeout=timeout)]
        except queue.Empty:
            return None
        while True:
            try:
                chunks.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks)

    def _open(self):  # noqa: ANN202 - native type
        try:
            from proctap._native import ProcessLoopback
        except ImportError as e:  # pragma: no cover - install problem
            raise LoopbackUnsupportedError(f"proctap native module unavailable: {e}") from e
        try:
            native = ProcessLoopback(self.pid)
        except RuntimeError as e:
            raise LoopbackUnsupportedError(str(e)) from e
        if not native.is_process_specific():
            detail = native.get_last_error() or "per-process activation failed"
            raise LoopbackUnsupportedError(f"Process loopback unavailable: {detail}")
        fmt = native.get_format()
        bits = fmt["bits_per_sample"]
        if bits not in (16, 32):
            raise RuntimeError(f"Unsupported loopback sample width: {bits}")
        self.sample_rate = fmt["sample_rate"]
        self.channels = fmt["channels"]
        self._dtype = np.float32 if bits == 32 else np.int16
        return native

    def _run(self) -> None:
        try:
            native = self._open()
            native.start()
        except BaseException as e:  # noqa: BLE001 - surfaced to start()
            self._error = e
            self._ready.set()
            return
        self._ready.set()
        try:
            while not self._stop.is_set():
                data = native.read()
                if not data:
                    time.sleep(POLL_S)
                    continue
                self._put(self._decode(data))
        except Exception as e:  # noqa: BLE001 - exposed via .error
            log.exception("Loopback capture for pid %s failed", self.pid)
            self._error = e
        finally:
            try:
                native.stop()
            except RuntimeError:
                pass

    def _decode(self, data: bytes) -> np.ndarray:
        arr = np.frombuffer(data, dtype=self._dtype)
        if self._dtype is np.int16:
            arr = arr.astype(np.float32) / 32768.0
        return arr.reshape(-1, self.channels).astype(np.float32, copy=False)

    def _put(self, chunk: np.ndarray) -> None:
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(chunk)
