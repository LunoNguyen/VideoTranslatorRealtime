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


# ---------------------------------------------------------------------------
# Streaming lane + speaker tagging (Task 5)
# ---------------------------------------------------------------------------
from backend.vad.segmenter import Partial  # noqa: E402


def A(n):
    """Audio of n samples (content irrelevant; fakes key on length)."""
    return np.zeros(n, dtype=np.float32)


def P(sid, n):
    return Partial(sid, A(n), 0.0, n / 16000)


def S(sid, n):
    return Segment(A(n), 0.0, 1.0, sid)


class ScriptSeg:
    """push(frame i) returns script[i]; records set_partial_every calls."""

    def __init__(self, script):
        self.script, self.every = script, []

    def push(self, frame):
        return list(self.script.get(int(frame[0]), []))

    def flush(self):
        return []

    def set_partial_every(self, s):
        self.every.append(s)


def timeline(gaps):
    """Yield frame i after sleeping gaps[i] seconds."""
    for i, g in enumerate(gaps):
        time.sleep(g)
        yield np.full(320, i, dtype=np.float32)


class LenASR:
    """Text derived from audio length; optional per-length delay / fake-clock advance."""

    def __init__(self, texts=None, delays=None, clock=None, advance=None):
        self.texts, self.delays = texts or {}, delays or {}
        self.clock, self.advance = clock, advance or {}
        self.lens = []

    def transcribe(self, audio, src_lang):
        n = len(audio)
        self.lens.append(n)
        time.sleep(self.delays.get(n, 0.0))
        if self.clock is not None:
            self.clock.t += self.advance.get(n, 0.0)
        return Transcript(self.texts.get(n, f"len{n}"), "en", 0.0)


class RecMT:
    def __init__(self):
        self.calls = []

    def translate(self, text, src_lang, tgt_lang="vi", **kw):
        self.calls.append((text, kw))
        return f"vi:{text}"


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def unit(i, dim=4):
    v = np.zeros(dim, dtype=np.float32)
    v[i] = 1.0
    return v


class FakeEmbedder:
    def __init__(self, vecs=None, fail=False):
        self.vecs, self.fail, self.lens = vecs or {}, fail, []

    def embed(self, audio):
        self.lens.append(len(audio))
        if self.fail:
            raise RuntimeError("onnx gone")
        return self.vecs.get(len(audio), unit(3))


async def run_stream(script, gaps, asr=None, mt=None, tts=None, speaker=None,
                     clock=None, setup=None, queue_max=4):
    subs, statuses = [], []

    async def on_sub(s):
        subs.append(s)

    async def on_status(k, m):
        statuses.append((k, m))

    seg = ScriptSeg(script)
    kw = {"clock": clock} if clock is not None else {}
    p = Pipeline(timeline(gaps), seg, asr or LenASR(), mt or RecMT(), tts, on_sub, on_status,
                 queue_max, speaker=speaker, **kw)
    if setup:
        setup(p)
    await asyncio.wait_for(p.run(), 20)
    return subs, statuses, seg, p


def view(subs):
    return [(s.id, s.final, s.rev, s.src_text) for s in subs]


async def test_partial_emits_nonfinal_with_commit_false():
    mt = RecMT()
    subs, _, _, _ = await run_stream({0: [P(1, 100)], 1: [S(1, 300)]}, [0, 0.3], mt=mt)
    assert view(subs) == [(1, False, 0, "len100"), (1, True, 1, "len300")]
    assert subs[0].vi_text == "vi:len100"
    # final path keeps the old call shape (no commit kwarg)
    assert mt.calls == [("len100", {"commit": False}), ("len300", {})]


async def test_final_after_partials_same_id_rev_increments():
    subs, _, _, _ = await run_stream(
        {0: [P(1, 100)], 1: [P(1, 200)], 2: [S(1, 300)]}, [0, 0.3, 0.3])
    assert view(subs) == [(1, False, 0, "len100"), (1, False, 1, "len200"), (1, True, 2, "len300")]


async def test_segment_preferred_over_partial():
    # Both arrive in one push; the partial (of the next sentence) is listed first.
    subs, _, _, _ = await run_stream({0: [P(2, 100), S(1, 300)]}, [0])
    assert view(subs) == [(1, True, 0, "len300"), (2, False, 0, "len100")]


async def test_partial_overwritten_by_newer():
    asr = LenASR(delays={500: 0.6})
    subs, _, _, _ = await run_stream(
        {0: [S(1, 500)], 1: [P(2, 100)], 2: [P(2, 200)]}, [0, 0.1, 0.1], asr=asr)
    assert view(subs) == [(1, True, 0, "len500"), (2, False, 0, "len200")]
    assert 100 not in asr.lens


async def test_late_partial_dropped():
    asr = LenASR(delays={100: 0.5})
    subs, st, _, _ = await run_stream({0: [P(1, 100)], 1: [S(1, 300)]}, [0, 0.15], asr=asr)
    assert view(subs) == [(1, True, 0, "len300")]
    assert ("warn", "streaming_slow") not in st


async def test_partial_of_finalized_segment_dropped_silently():
    asr = LenASR()
    subs, _, _, _ = await run_stream({0: [S(1, 300), P(1, 100)]}, [0], asr=asr)
    assert view(subs) == [(1, True, 0, "len300")]
    assert 100 not in asr.lens


async def test_unchanged_partial_text_skips_mt():
    mt = RecMT()
    asr = LenASR(texts={100: "hello", 200: "hello", 300: "hello there"})
    subs, _, _, _ = await run_stream(
        {0: [P(1, 100)], 1: [P(1, 200)], 2: [S(1, 300)]}, [0, 0.3, 0.3], asr=asr, mt=mt)
    assert mt.calls == [("hello", {"commit": False}), ("hello there", {})]
    assert view(subs) == [(1, False, 0, "hello"), (1, True, 1, "hello there")]


async def test_tts_only_on_final():
    tts = FakeTTS()
    await run_stream({0: [P(1, 100)], 1: [P(1, 200)], 2: [S(1, 300)]}, [0, 0.3, 0.3], tts=tts)
    assert tts.said == ["vi:len300"]


async def test_streaming_off_emits_no_partials():
    subs, _, seg, _ = await run_stream(
        {0: [P(1, 100)], 1: [S(1, 300)]}, [0.2, 0.3], setup=lambda p: p.set_streaming(False))
    assert view(subs) == [(1, True, 0, "len300")]
    assert seg.every == [None]


async def test_set_streaming_false_clears_pending_partial():
    seg = ScriptSeg({})
    p = Pipeline([], seg, LenASR(), RecMT(), None, None, None)
    p._partial = P(1, 100)
    p.set_streaming(False)
    assert p._partial is None and seg.every == [None]
    p.set_streaming(True)
    assert seg.every == [None, 0.7]


async def test_adaptive_interval_grows_and_caps_2s():
    clock = FakeClock()
    asr = LenASR(clock=clock, advance={100: 1.0, 200: 3.0, 300: 0.1, 400: 0.0})
    script = {0: [P(1, 100)], 1: [P(2, 200)], 2: [P(3, 300)], 3: [P(4, 400)]}
    _, _, seg, _ = await run_stream(script, [0, 0.3, 0.3, 0.3], asr=asr, clock=clock)
    assert [round(x, 3) for x in seg.every] == [1.2, 2.0, 1.8, 1.62]


async def test_adaptive_interval_floor_0_7():
    clock = FakeClock()
    _, _, seg, _ = await run_stream({0: [P(1, 100)], 1: [P(2, 200)]}, [0, 0.3],
                                    asr=LenASR(clock=clock), clock=clock)
    assert seg.every == [0.7, 0.7]


def _late_block(script, gaps, sid):
    """Slow partial of `sid`, then its Segment arrives while the partial is still in ASR."""
    script[len(gaps)] = [P(sid, 100)]
    gaps.append(0.5 if gaps else 0)
    script[len(gaps)] = [S(sid, 300)]
    gaps.append(0.15)


def _ontime_block(script, gaps, sid):
    script[len(gaps)] = [P(sid, 200)]
    gaps.append(0.5 if gaps else 0)
    script[len(gaps)] = [S(sid, 300)]
    gaps.append(0.3)


async def test_three_late_partials_disable_streaming():
    script, gaps = {}, []
    for sid in (1, 2, 3):
        _late_block(script, gaps, sid)
    _ontime_block(script, gaps, 9)  # would be on time, but streaming is off now
    subs, st, seg, _ = await run_stream(script, gaps, asr=LenASR(delays={100: 0.4}))
    assert all(s.final for s in subs) and [s.id for s in subs] == [1, 2, 3, 9]
    assert st.count(("warn", "streaming_slow")) == 1
    assert seg.every[-1] is None


async def test_on_time_partial_resets_late_count():
    script, gaps = {}, []
    _late_block(script, gaps, 1)
    _late_block(script, gaps, 2)
    _ontime_block(script, gaps, 3)
    _late_block(script, gaps, 4)
    _late_block(script, gaps, 5)
    subs, st, _, _ = await run_stream(script, gaps, asr=LenASR(delays={100: 0.4}))
    assert ("warn", "streaming_slow") not in st
    assert (3, False, 0, "len200") in view(subs)


async def test_final_speaker_assigned():
    emb = FakeEmbedder({300: unit(0), 500: unit(1)})
    subs, _, _, _ = await run_stream(
        {0: [S(1, 300)], 1: [S(2, 500)], 2: [S(3, 300)]}, [0, 0.1, 0.1], speaker=emb)
    assert [s.speaker for s in subs] == [0, 1, 0]
    assert emb.lens == [300, 500, 300]


async def test_partial_uses_match_not_assign():
    emb = FakeEmbedder({24000: unit(0), 48000: unit(0), 64000: unit(1)})
    script = {
        0: [P(1, 32000)], 1: [P(1, 40000)], 2: [S(1, 48000)],
        3: [P(2, 32000)], 4: [S(2, 64000)],
    }
    subs, _, _, p = await run_stream(script, [0, 0.3, 0.3, 0.3, 0.3], speaker=emb)
    # first partial: no clusters yet -> match gives None and creates nothing
    assert [(s.id, s.final, s.speaker) for s in subs] == [
        (1, False, None), (1, False, None), (1, True, 0), (2, False, 0), (2, True, 1)]
    # provisional embed once per seg_id, on the first 1.5 s only
    assert emb.lens == [24000, 48000, 24000, 64000]
    assert p._clusterer.count == 2


async def test_speaker_error_disables_and_warns():
    emb = FakeEmbedder(fail=True)
    subs, st, _, _ = await run_stream(
        {0: [P(1, 100)], 1: [S(1, 300)], 2: [S(2, 300)]}, [0, 0.3, 0.1], speaker=emb)
    assert view(subs) == [(1, False, 0, "len100"), (1, True, 1, "len300"), (2, True, 0, "len300")]
    assert all(s.speaker is None for s in subs)
    assert st.count(("warn", "speaker_unavailable")) == 1
    assert len(emb.lens) == 1


async def test_no_speaker_gives_none():
    subs, _, _, _ = await run_stream({0: [P(1, 100)], 1: [S(1, 300)]}, [0, 0.3])
    assert [s.speaker for s in subs] == [None, None]


async def test_speaker_disabled_gives_none():
    emb = FakeEmbedder({300: unit(0)})
    subs, _, _, _ = await run_stream({0: [S(1, 300)]}, [0], speaker=emb,
                                     setup=lambda p: p.set_speaker_enabled(False))
    assert subs[0].speaker is None and emb.lens == []


async def test_reset_speakers_clears_clusters():
    emb = FakeEmbedder({300: unit(0), 500: unit(1)})
    _, _, _, p = await run_stream({0: [S(1, 300)], 1: [S(2, 500)]}, [0, 0.1], speaker=emb)
    assert p._clusterer.count == 2
    p.reset_speakers()
    assert p._clusterer.count == 0


async def test_unchanged_ontime_partial_does_not_reset_late_count():
    script, gaps = {0: [P(3, 200)]}, [0]  # sent "same" (count 0)
    _late_block(script, gaps, 1)
    _late_block(script, gaps, 2)
    script[len(gaps)] = [P(3, 250)]  # on time, text unchanged -> not sent, no reset
    gaps.append(0.5)
    script[len(gaps)] = [S(3, 300)]
    gaps.append(0.3)
    _late_block(script, gaps, 4)
    asr = LenASR(texts={200: "same", 250: "same"}, delays={100: 0.4})
    subs, st, _, _ = await run_stream(script, gaps, asr=asr)
    assert st.count(("warn", "streaming_slow")) == 1
    assert [s.rev for s in subs if s.id == 3] == [0, 1]


async def test_partial_on_subtitle_error_only_logged():
    subs, statuses = [], []

    async def on_sub(s):
        subs.append(s)
        raise ConnectionError("ws closed")

    async def on_status(k, m):
        statuses.append((k, m))

    p = Pipeline(timeline([0, 0.3]), ScriptSeg({0: [P(1, 100)], 1: [S(1, 300)]}), LenASR(),
                 RecMT(), None, on_sub, on_status)
    await asyncio.wait_for(p.run(), 10)
    assert [s.final for s in subs] == [False, True]
    assert statuses == [("error", "subtitle: ws closed")]  # only from the final


async def test_partial_in_flight_not_emitted_after_streaming_off():
    subs = []

    async def on_sub(s):
        subs.append(s)

    async def noop(*a):
        pass

    seg = ScriptSeg({0: [P(1, 100)], 1: [S(1, 300)]})
    p = Pipeline(timeline([0, 0.6]), seg, LenASR(delays={100: 0.4}), RecMT(), None, on_sub, noop)
    task = asyncio.create_task(p.run())
    await asyncio.sleep(0.2)  # partial is inside ASR
    p.set_streaming(False)
    await asyncio.wait_for(task, 10)
    assert view(subs) == [(1, True, 0, "len300")]
    assert seg.every[-1] is None


async def test_batch_routed_before_drop_status():
    seen = []
    p = None

    async def on_status(k, m):
        # at status time the whole batch must already be routed
        seen.append((m, [s.seg_id for s in p._queue], p._partial.seg_id if p._partial else None))
        await asyncio.sleep(0)

    p = Pipeline([], ScriptSeg({}), LenASR(), RecMT(), None, None, on_status, seg_queue_max=1)
    p._wake = asyncio.Event()
    await p._enqueue([P(3, 100), S(1, 300), S(2, 400)])
    assert seen == [("dropped_segment", [2], 3)]


async def test_speaker_cfg_builds_clusterer():
    p = Pipeline([], ScriptSeg({}), LenASR(), RecMT(), None, None, None,
                 speaker=FakeEmbedder(), speaker_cfg={"threshold": 0.8, "max_speakers": 3})
    assert p._clusterer.threshold == 0.8 and p._clusterer.max_speakers == 3


async def test_partial_every_s_kwarg_is_floor():
    clock = FakeClock()
    seg = ScriptSeg({0: [P(1, 100)], 1: [P(2, 200)]})

    async def nop(*a):
        pass

    p = Pipeline(timeline([0, 0.3]), seg, LenASR(clock=clock), RecMT(), None, nop, nop,
                 partial_every_s=1.0, clock=clock)
    p.set_streaming(True)
    await asyncio.wait_for(p.run(), 20)
    assert seg.every[0] == 1.0 and all(x == 1.0 for x in seg.every)


# ---- final-review fixes ----
class EventEmbedder(FakeEmbedder):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.started = threading.Event()

    def embed(self, audio):
        self.started.set()
        return super().embed(audio)


async def test_final_embed_runs_concurrently_with_mt():
    emb = EventEmbedder({300: unit(0)})
    seen = []

    class CheckMT(RecMT):
        def translate(self, text, src_lang, tgt_lang="vi", **kw):
            seen.append(emb.started.wait(2.0))
            return super().translate(text, src_lang, tgt_lang, **kw)

    subs, _, _, _ = await run_stream({0: [S(1, 300)]}, [0], speaker=emb, mt=CheckMT())
    assert seen == [True]
    assert subs[0].speaker == 0


async def test_long_audio_capped_to_middle_4s():
    emb = FakeEmbedder()
    await run_stream({0: [S(1, 16000 * 7)]}, [0], speaker=emb)
    assert emb.lens == [4 * 16000]


async def test_empty_asr_final_does_not_leak_embed_task():
    emb = FakeEmbedder()

    class NoneASR:
        def transcribe(self, audio, src_lang):
            return None

    subs, st, _, _ = await run_stream({0: [S(1, 300)]}, [0], speaker=emb, asr=NoneASR())
    assert subs == []
    assert [t for t in asyncio.all_tasks() if t is not asyncio.current_task()] == []


async def test_final_embed_failure_disables_and_warns_once():
    emb = FakeEmbedder(fail=True)
    subs, st, _, _ = await run_stream({0: [S(1, 300)], 1: [S(2, 300)]}, [0, 0.1], speaker=emb)
    assert [s.src_text for s in subs] == ["len300", "len300"]
    assert all(s.speaker is None for s in subs)
    assert st.count(("warn", "speaker_unavailable")) == 1


async def test_late_partial_does_not_spend_embed():
    emb = FakeEmbedder()

    async def noop(*a):
        pass

    p = Pipeline(iter([]), ScriptSeg({}), LenASR(), RecMT(), None, noop, noop, speaker=emb)
    p._queue.append(S(1, 300))  # the sentence's final is already waiting
    assert await p._provisional_speaker(P(1, 32000)) is None
    assert emb.lens == []
    assert await p._provisional_speaker(P(2, 32000)) is None  # no cluster yet, but embeds
    assert emb.lens == [24000]
