import re
import unicodedata

from ..detect.text import (ZERO_WIDTH, BIDI, UNUSUAL_SPACES, PUA_BMP,
                           CONFUSABLES, _is_emoji)

DEFAULTS = {
    "strip_invisible": True,
    "strip_bidi": True,
    "normalize_spaces": True,
    "remove_soft_hyphen": True,
    "fix_homoglyphs": True,
    "strip_private_use": True,
    "normalize_whitespace": True,
    "normalize_typography": False,
    "nfkc": False,
}

TAG_CHARS = lambda c: 0xE0000 <= ord(c) <= 0xE007F
VS_SUPP = lambda c: 0xE0100 <= ord(c) <= 0xE01EF


def _keep_char(text, i):
    """Preserve ZWJ/VS16 when adjacent to emoji."""
    ch = text[i]
    if ch not in ("‍", "️"):
        return False
    for j in (i - 1, i + 1):
        if 0 <= j < len(text) and _is_emoji(text[j]):
            return True
    return False


def clean_text(text: str, options: dict | None = None):
    opt = {**DEFAULTS, **(options or {})}
    actions = []
    out = text

    if opt["nfkc"]:
        out = unicodedata.normalize("NFKC", out)
        actions.append("Applied NFKC normalization")

    if opt["strip_invisible"]:
        n = 0
        buf = []
        for i, ch in enumerate(out):
            o = ord(ch)
            if TAG_CHARS(ch) or VS_SUPP(ch):
                n += 1
                continue
            if o in ZERO_WIDTH or 0xFE00 <= o <= 0xFE0F:
                if _keep_char(out, i):
                    buf.append(ch)
                else:
                    n += 1
                continue
            buf.append(ch)
        out = "".join(buf)
        if n:
            actions.append(f"Removed {n} invisible/tag/variation-selector characters")

    if opt["strip_bidi"]:
        n = sum(1 for c in out if ord(c) in BIDI)
        out = "".join(c for c in out if ord(c) not in BIDI)
        if n:
            actions.append(f"Removed {n} bidi control characters")

    if opt["remove_soft_hyphen"]:
        n = out.count("\u00ad")
        out = out.replace("\u00ad", "")
        if n:
            actions.append(f"Removed {n} soft hyphens")

    if opt["normalize_spaces"]:
        n = 0
        def rep(m):
            nonlocal n
            n += len(m.group(0))
            return " " * len(m.group(0))
        out = re.sub(r"[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]+", rep, out)
        if n:
            actions.append(f"Normalized {n} unusual space characters to U+0020")

    if opt["fix_homoglyphs"]:
        n = 0
        def fix_word(m):
            nonlocal n
            w = m.group(0)
            if not any(c.isascii() and c.isalpha() for c in w):
                return w
            res = "".join(CONFUSABLES.get(c, c) for c in w)
            n += sum(1 for c in w if c in CONFUSABLES)
            return res
        out = re.sub(r"[^\W\d_]+", fix_word, out, flags=0)
        if n:
            actions.append(f"Mapped {n} confusable characters to Latin")

    if opt["strip_private_use"]:
        n = sum(1 for c in out if ord(c) in PUA_BMP
                or 0xF0000 <= ord(c) <= 0xFFFFD or 0x100000 <= ord(c) <= 0x10FFFD)
        out = "".join(c for c in out if not (ord(c) in PUA_BMP
                      or 0xF0000 <= ord(c) <= 0xFFFFD or 0x100000 <= ord(c) <= 0x10FFFD))
        if n:
            actions.append(f"Removed {n} private-use characters")

    if opt["normalize_whitespace"]:
        crlf = out.count("\r\n")
        out = out.replace("\r\n", "\n").replace("\r", "\n")
        t = len(re.findall(r"[ \t]+$", out, re.M))
        out = re.sub(r"[ \t]+$", "", out, flags=re.M)
        d = len(re.findall(r"(?<=\S)  (?=\S)", out))
        out = re.sub(r"(?<=\S) {2,}(?=\S)", " ", out)
        parts = []
        if crlf:
            parts.append(f"{crlf} CRLF→LF")
        if t:
            parts.append(f"trailing whitespace on {t} lines")
        if d:
            parts.append(f"{d} double spaces collapsed")
        if parts:
            actions.append("Normalized whitespace: " + ", ".join(parts))

    if opt["normalize_typography"]:
        n = 0
        table = {"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
                 "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u00a0": " "}
        for a, b in table.items():
            c = out.count(a)
            n += c
            out = out.replace(a, b)
        if n:
            actions.append(f"Normalized {n} typographic variants")

    return out, actions
