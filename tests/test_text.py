from app.detect.text import analyze_text
from app.clean.text import clean_text


def tag(s):
    return "".join(chr(0xE0000 + ord(c)) for c in s)


def ids(rep):
    return {f.id for f in rep["findings"]}


def test_detect_and_clean():
    text = ("Hello\u200b world\u00a0!\n"
            "p\u0430ssword reset\n"
            + tag("id=173")
            + "line with spaces   \n"
            "soft\u00adhyph\u00aden\n")
    rep = analyze_text(text)
    fids = ids(rep)
    assert "zero-width" in fids
    assert "unusual-spaces" in fids
    assert "homoglyphs" in fids
    assert "tag-characters" in fids
    assert "whitespace-pattern" in fids
    assert "soft-hyphen" in fids
    tf = [f for f in rep["findings"] if f.id == "tag-characters"][0]
    assert "id=173" in tf.evidence

    out, actions = clean_text(text, None)
    after = analyze_text(out)
    remaining = ids(after) - {"typographic-variants"}
    assert not remaining, remaining
    assert "p" in out and "\u0430" not in out
    assert "id=173" not in out


def test_emoji_preserved():
    text = "family 👨‍👩‍👧 and ❤️ emoji"
    out, _ = clean_text(text, None)
    assert "👨‍👩‍👧" in out
    assert "❤️" in out
    after = analyze_text(out)
    assert "variation-selectors" not in ids(after)


def test_bidi_and_pua():
    text = "ab\u202ecdef\ue001g"
    rep = analyze_text(text)
    fids = ids(rep)
    assert "bidi-controls" in fids
    assert "private-use" in fids
    out, _ = clean_text(text, None)
    assert "\u202e" not in out and "\ue001" not in out


def test_options_off():
    text = "“quote” — dash…"
    out, _ = clean_text(text, {"normalize_typography": True})
    assert "“" not in out and "—" not in out
    rep = analyze_text(text)
    assert "typographic-variants" in ids(rep)
