import time

import numpy as np
import pytest

from backend.capture import source as source_mod
from backend.capture.source import ProcessLoopbackSource


class FakeNative:
    """Mimics proctap._native.ProcessLoopback: a few packets, then raises."""

    def __init__(self, packets: int = 3):
        self.packets = packets
        self.stopped = False

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self.stopped = True

    def read(self) -> bytes | None:
        if self.packets:
            self.packets -= 1
            return np.zeros((480, 2), np.float32).tobytes()
        raise RuntimeError("Failed to read data: HRESULT=0x88890004")


@pytest.fixture
def no_os_check(monkeypatch):
    monkeypatch.setattr(source_mod, "_check_os", lambda: None)


def test_worker_sets_error_when_native_read_raises(no_os_check, monkeypatch):
    native = FakeNative()
    monkeypatch.setattr(ProcessLoopbackSource, "_open", lambda self: native)
    src = ProcessLoopbackSource(1234)
    src.start()
    assert src.error is None or isinstance(src.error, RuntimeError)
    deadline = time.monotonic() + 2
    while src.error is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert isinstance(src.error, RuntimeError)
    assert native.stopped
    data = src.read(0.1)  # packets captured before the failure stay readable
    assert data is not None and data.shape[1] == 2
    src.stop()


def test_worker_sets_error_on_decode_mismatch(no_os_check, monkeypatch):
    class OddNative(FakeNative):
        def read(self) -> bytes:
            return b"\x00" * 12  # not a multiple of one stereo float32 frame

    monkeypatch.setattr(ProcessLoopbackSource, "_open", lambda self: OddNative())
    src = ProcessLoopbackSource(1234)
    src.start()
    deadline = time.monotonic() + 2
    while src.error is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert isinstance(src.error, ValueError)
    src.stop()


@pytest.mark.parametrize("build,ok", [(19041, True), (19045, True), (18363, False)])
def test_os_check_accepts_windows10_2004(monkeypatch, build, ok):
    from types import SimpleNamespace

    monkeypatch.setattr(source_mod.sys, "platform", "win32")
    monkeypatch.setattr(source_mod.sys, "getwindowsversion", lambda: SimpleNamespace(build=build), raising=False)
    if ok:
        source_mod._check_os()
    else:
        with pytest.raises(source_mod.LoopbackUnsupportedError, match="19041"):
            source_mod._check_os()
