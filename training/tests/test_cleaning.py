from training.cleaning import (
    clean_pair,
    clean_text,
    dedup_norm,
    dedup_pairs,
    lang_ok,
    leak_keys,
    make_context_pairs,
    norm_key,
    should_sync,
    split_segments,
    touches_test,
)


def test_strip_tags_and_music():
    assert clean_pair("<i>♪ Hello {\\an8}there</i>", "<b>Xin chào</b> ♪ ♫") == ("Hello there", "Xin chào")


def test_unescape_entities_and_whitespace():
    assert clean_text("It &apos;s  a &quot;test&quot;\t") == "It 's a \"test\""


def test_empty_after_cleaning_is_dropped():
    assert clean_pair("♪ ♪", "Xin chào") is None
    assert clean_pair("Hello", "<i></i>") is None


def test_length_ratio_filter():
    assert clean_pair("Hi", "Xin chào bạn, rất vui được gặp bạn hôm nay") is None  # > 3
    assert clean_pair("Nice to meet you again, my old friend", "Chào") is None  # < 1/3
    assert clean_pair("Nice to meet you.", "Rất vui được gặp bạn.") is not None


def test_too_long_dropped():
    long = "word " * 45  # > 200 chars
    assert clean_pair(long, long) is None


def test_japanese_ratio_uses_weighted_length():
    # 5 ja chars vs 18 vi chars: plain char ratio is 3.6 but the pair is a valid translation
    assert clean_pair("元気ですか", "Bạn có khỏe không?") == ("元気ですか", "Bạn có khỏe không?")


def test_mojibake_dropped():
    assert clean_pair("Science", "Khoa h�c") is None


def test_make_context_pairs_n1():
    rows = [("a", "A"), ("b", "B"), ("c", "C")]
    out = make_context_pairs(rows, n=1)
    assert out[0] == ("a", "A")
    assert out[1] == ("a b", "A B")
    assert out[2] == ("b c", "B C")


def test_make_context_pairs_n2_and_empty():
    rows = [("a", "A"), ("b", "B"), ("c", "C")]
    assert make_context_pairs(rows, n=2)[2] == ("a b c", "A B C")
    assert make_context_pairs(rows, n=0) == rows
    assert make_context_pairs([], n=1) == []


def test_lang_ok():
    assert lang_ok("こんにちは、元気ですか?", "ja")
    assert not lang_ok("Hello, how are you?", "ja")
    assert lang_ok("Hello, how are you?", "en")
    assert not lang_ok("こんにちは", "en")
    assert lang_ok("Tôi đang đi chợ hôm nay.", "vi")
    assert lang_ok("OK.", "vi")  # short lines may lack diacritics
    assert not lang_ok("I am going to the market today.", "vi")  # untranslated English
    assert not lang_ok("元気です", "vi")


def test_split_segments_breaks_on_drop_and_doc_change():
    items = [("d1", ("a", "A")), ("d1", ("b", "B")), ("d1", None), ("d1", ("c", "C")),
             ("d2", ("e", "E")), (None, ("f", "F"))]
    assert split_segments(items) == [[("a", "A"), ("b", "B")], [("c", "C")], [("e", "E")], [("f", "F")]]


def test_hangul_is_not_japanese():
    assert not lang_ok("안녕하세요", "ja")


def test_norm_key_ignores_case_punct_space():
    assert norm_key("  Hello, World! ") == norm_key("hello world") == "helloworld"
    assert norm_key("元気です。") == norm_key("元気 です")
    assert norm_key("...") == ""


def test_touches_test_on_source_or_target():
    keys = leak_keys([("How are you?", "Bạn khỏe không?"), ("!!!", "...")])
    assert touches_test([("how are you", "khác")], keys)  # source match
    assert touches_test([("other", "BẠN KHỎE KHÔNG")], keys)  # target match
    assert touches_test([("prev", "P"), ("How are you", "x")], keys)  # any row of a context window
    assert not touches_test([("Fine.", "Ổn.")], keys)
    assert not touches_test([("?", "!")], keys)  # empty keys never match


def test_dedup_norm_on_source_key():
    pairs = [("Hello!", "Chào!"), ("hello", "Xin chào"), ("Bye.", "Tạm biệt."), ("...", "..."), ("?", "!")]
    assert dedup_norm(pairs) == [("Hello!", "Chào!"), ("Bye.", "Tạm biệt.")]


def test_should_sync():
    h = 3600.0
    assert should_sync(2000, 9000, None, 0.0, 60)  # first save always syncs
    assert not should_sync(4000, 9000, 0.0, 0.5 * h, 60)  # too soon
    assert should_sync(6000, 9000, 0.0, 1.0 * h, 60)
    assert not should_sync(9000, 9000, 0.0, 5 * h, 60)  # final step: the final model is saved right after
    assert not should_sync(9000, 9000, None, 0.0, 60)
    assert should_sync(100, 0, None, 0.0, 60)  # unknown max_steps
    assert not should_sync(6000, 9000, 0.0, 2 * h, 60, n_synced=2, max_syncs=2)  # cap reached
    assert should_sync(6000, 9000, 0.0, 2 * h, 60, n_synced=1, max_syncs=2)


def test_dedup_pairs_keeps_first_order():
    assert dedup_pairs([("a", "A"), ("b", "B"), ("a", "A"), ("a", "X")]) == [("a", "A"), ("b", "B"), ("a", "X")]
