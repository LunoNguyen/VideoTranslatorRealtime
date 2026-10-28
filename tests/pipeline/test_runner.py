import asyncio
import threading
import time

import numpy as np

from backend.asr.whisper import Transcript
from backend.pipeline.runner import Pipeline, Subtitle
from backend.vad.segmenter import Segment


class FakeSeg:
    """Every pushed frame becomes one segment whose audio value is the frame's index."""

    def push(self, frame):
        v = float(frame[0])
        return [Segment(frame, v, v + 1)]

    def flush(self):
        return []


class FakeASR:
    def __init__(self, delay=0.0, fail_on=None, texts=None):
        self.delay, self.fail_on, self.texts = delay, fail_on, texts or {}
        self.langs = []

    def transcribe(self, audio, src_lang):
        self.langs.append(src_lang)
        time.sleep(self.delay)
        v = int(audio[0])
        if v == self.fail_on:
            raise RuntimeError("boom")
        return Transcript(self.texts.get(v, f"text{v}"), "en", 0.0)


class FakeMT:
    def translate(self, text, src_lang, tgt_lang="vi"):
        return f"vi:{text}:{src_lang}"


class FakeTTS:
    def __init__(self):
        self.said, self.enabled = [], True

    def set_enabled(self, e):
        self.enabled = e

    def say(self, text):
        self.said.append(text)


def frames(n):
    return [np.full(320, i, dtype=np.float32) for i in range(n)]


async def run_pipe(n, asr, tts=None, queue_max=4, setup=None, sub_raises=False, status_raises=False):
    subs, statuses = [], []

    async def on_sub(s: Subtitle):
        subs.append(s)
        if sub_raises and len(subs) == 1:
            raise ConnectionError("ws closed")

    async def on_status(k, m):
        statuses.append((k, m))
        if status_raises:
            raise ConnectionError("ws closed")

    p = Pipeline(frames(n), FakeSeg(), asr, FakeMT(), tts, on_sub, on_status, queue_max)
    if setup:
        setup(p)
    await asyncio.wait_for(p.run(), 15)
    return subs, statuses


async def test_end_to_end_emits_subtitle():
    subs, _ = await run_pipe(3, FakeASR())
    assert [s.id for s in subs] == [1, 2, 3]
    assert subs[0].vi_text == "vi:text0:en" and subs[0].src_text == "text0"
    assert subs[0].src_lang == "en" and (subs[0].t_start, subs[0].t_end) == (0.0, 1.0)


async def test_src_lang_explicit_used():
    asr = FakeASR()
    subs, _ = await run_pipe(1, asr, setup=lambda p: p.set_src_lang("ja"))
    assert asr.langs == ["ja"] and subs[0].src_lang == "ja"


async def test_hallucination_not_emitted():
    subs, _ = await run_pipe(2, FakeASR(texts={0: "Thanks for watching!"}))
    assert [s.src_text for s in subs] == ["text1"]


async def test_full_queue_drops_oldest():
    subs, st = await run_pipe(10, FakeASR(delay=0.5))
    assert ("warn", "dropped_segment") in st
    assert subs[-1].src_text == "text9"
    assert "text1" not in [s.src_text for s in subs]
    assert len(subs) < 10


async def test_stage_exception_continues():
    subs, st = await run_pipe(3, FakeASR(fail_on=1))
    assert [s.src_text for s in subs] == ["text0", "text2"]
    assert ("error", "boom") in st


async def test_tts_called_only_when_enabled():
    tts = FakeTTS()
    await run_pipe(2, FakeASR(), tts)
    assert tts.said == ["vi:text0:en", "vi:text1:en"]
    tts2 = FakeTTS()
    await run_pipe(2, FakeASR(), tts2, setup=lambda p: p.set_tts_enabled(False))
    assert tts2.said == [] and tts2.enabled is False


async def test_stop_ends_run_cleanly():
    def endless():
        i = 0
        while True:
            time.sleep(0.01)
            yield np.full(320, i % 100, dtype=np.float32)
            i += 1

    async def noop(*a):
        pass

    p = Pipeline(endless(), FakeSeg(), FakeASR(), FakeMT(), None, noop, noop)
    task = asyncio.create_task(p.run())
    await asyncio.sleep(0.2)
    p.stop()
    await asyncio.wait_for(task, 5)
    assert not any(t.name == "capture-vad" for t in threading.enumerate())


async def test_on_subtitle_exception_continues():
    subs, st = await run_pipe(3, FakeASR(), sub_raises=True)
    assert [s.src_text for s in subs] == ["text0", "text1", "text2"]
    assert any(k == "error" for k, _ in st)


async def test_on_status_exception_does_not_break_capture():
    subs, st = await run_pipe(10, FakeASR(delay=0.2), status_raises=True)
    assert st and subs[-1].src_text == "text9"


async def test_stop_before_run_is_respected():
    async def noop(*a):
        pass

    p = Pipeline(frames(5), FakeSeg(), FakeASR(), FakeMT(), None, noop, noop)
    p.stop()
    await asyncio.wait_for(p.run(), 5)
