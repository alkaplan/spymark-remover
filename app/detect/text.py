import re
import unicodedata

from ..findings import Finding

ZERO_WIDTH = {0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0x2060, 0x2061, 0x2062,
              0x2063, 0x2064, 0xFEFF, 0x180E, 0x034F, 0x061C, 0x115F, 0x1160,
              0x17B4, 0x17B5, 0x3164, 0xFFA0, 0x2028, 0x2029}
BIDI = set(range(0x202A, 0x202F)) | set(range(0x2066, 0x206A))
UNUSUAL_SPACES = {0x00A0, 0x1680, *range(0x2000, 0x200B), 0x202F, 0x205F, 0x3000}
PUA_BMP = range(0xE000, 0xF900)

CONFUSABLES = {}
for s, d in {
    "\u0430": "a", "\u0435": "e", "\u043e": "o", "\u0440": "p", "\u0441": "c",
    "\u0443": "y", "\u0445": "x", "\u0456": "i", "\u0458": "j", "\u0455": "s",
    "\u04bb": "h", "\u0501": "d", "\u0261": "g",
    "\u0410": "A", "\u0412": "B", "\u0415": "E", "\u041a": "K", "\u041c": "M",
    "\u041d": "H", "\u041e": "O", "\u0420": "P", "\u0421": "C", "\u0422": "T",
    "\u0425": "X",
    "\u03bf": "o", "\u03bd": "v",
    "\u0391": "A", "\u0392": "B", "\u0395": "E", "\u0397": "H", "\u0399": "I",
    "\u039a": "K", "\u039c": "M", "\u039d": "N", "\u039f": "O", "\u03a1": "P",
    "\u03a4": "T", "\u03a7": "X", "\u0396": "Z",
}.items():
    CONFUSABLES[s] = d
for c in range(0xFF01, 0xFF5F):
    CONFUSABLES[chr(c)] = chr(c - 0xFEE0)
for c in range(0x1D400, 0x1D800):
    try:
        n = unicodedata.normalize("NFKC", chr(c))
        if len(n) == 1 and n.isalpha():
            CONFUSABLES[chr(c)] = n
    except Exception:
        pass


def _is_emoji(ch):
    o = ord(ch)
    return (0x1F000 <= o <= 0x1FAFF or 0x2600 <= o <= 0x27BF
            or 0x2300 <= o <= 0x23FF or 0x1F1E6 <= o <= 0x1F1FF)


def _pos(text, idx):
    line = text.count("\n", 0, idx) + 1
    col = idx - text.rfind("\n", 0, idx)
    return f"{line}:{col}"


def _chname(ch):
    try:
        return unicodedata.name(ch)
    except ValueError:
        return "<unnamed>"


def _examples(text, pred, n=5):
    out = []
    for i, ch in enumerate(text):
        if pred(ch, i):
            out.append(f"{_pos(text, i)} U+{ord(ch):04X} {_chname(ch)}")
            if len(out) == n:
                break
    return out


def analyze_text(text: str) -> dict:
    findings = []
    notes = []

    def add(fid, name, sev, conf, ev, removable=True, removal=""):
        findings.append(Finding(id=fid, category="text", name=name, severity=sev,
                                confidence=conf, evidence=ev, removable=removable,
                                removal=removal))

    zw = [ch for ch in text if ord(ch) in ZERO_WIDTH]
    if zw:
        ex = _examples(text, lambda c, i: ord(c) in ZERO_WIDTH)
        add("zero-width", "Zero-width / invisible characters", "high", 1.0,
            f"{len(zw)} invisible chars ({', '.join(sorted({f'U+{ord(c):04X}' for c in zw}))}); e.g. {'; '.join(ex)}",
            removal="Removed by strip_invisible")

    bd = [ch for ch in text if ord(ch) in BIDI]
    if bd:
        ex = _examples(text, lambda c, i: ord(c) in BIDI)
        add("bidi-controls", "Bidirectional control characters", "medium", 1.0,
            f"{len(bd)} bidi controls; e.g. {'; '.join(ex)}",
            removal="Removed by strip_bidi")

    vs_e = [ch for ch in text if 0xE0100 <= ord(ch) <= 0xE01EF]
    vs_f = [text[i] for i in range(1, len(text))
            if 0xFE00 <= ord(text[i]) <= 0xFE0F and ord(text[i - 1]) < 0x80
            and not _is_emoji(text[i - 1])]
    if vs_e or vs_f:
        add("variation-selectors", "Variation selectors (steganographic)", "high", 0.95,
            f"{len(vs_e) + len(vs_f)} variation selectors on non-emoji bases",
            removal="Removed by strip_invisible (emoji VS16 preserved)")

    tags = [ord(ch) for ch in text if 0xE0000 <= ord(ch) <= 0xE007F]
    if tags:
        decoded = "".join(chr(c - 0xE0000) for c in tags if 0xE0001 <= c <= 0xE007F)
        ev = f"{len(tags)} tag characters"
        if decoded.strip():
            ev += f"; hidden ASCII: \"{decoded[:120]}\""
        add("tag-characters", "Unicode tag characters (ASCII smuggling)", "high", 1.0,
            ev, removal="Removed by strip_invisible")

    us = [ch for ch in text if ord(ch) in UNUSUAL_SPACES]
    if us:
        kinds = sorted({f"U+{ord(c):04X}" for c in us})
        sev = "low" if set(map(ord, us)) == {0x00A0} and len(us) < 4 else "medium"
        add("unusual-spaces", "Unusual space characters", sev, 0.9,
            f"{len(us)} unusual spaces ({', '.join(kinds)})",
            removal="Normalized to U+0020")

    sh = [ch for ch in text if ch == "\u00ad"]
    if sh:
        add("soft-hyphen", "Soft hyphens", "medium", 1.0,
            f"{len(sh)} soft hyphens (U+00AD)", removal="Removed")

    mixed = []
    for m in re.finditer(r"[^\W\d_]+", text, re.UNICODE):
        w = m.group(0)
        conf = [c for c in w if c in CONFUSABLES]
        latin = [c for c in w if c.isascii() and c.isalpha()]
        if conf and latin:
            mixed.append((m.start(), w, conf))
    if mixed:
        ex = "; ".join(f"{_pos(text, s)} \"{w}\" ({len(c)} confusable)"
                       for s, w, c in mixed[:5])
        add("homoglyphs", "Mixed-script homoglyphs", "high", 0.95,
            f"{len(mixed)} words mix Latin with confusable lookalikes; e.g. {ex}",
            removal="Confusables mapped to Latin in mixed-script words")

    pua = [ch for ch in text if ord(ch) in PUA_BMP
           or 0xF0000 <= ord(ch) <= 0xFFFFD or 0x100000 <= ord(ch) <= 0x10FFFD]
    if pua:
        add("private-use", "Private-use characters", "medium", 1.0,
            f"{len(pua)} private-use codepoints",
            removal="Removed by strip_private_use")

    trailing = len(re.findall(r"[ \t]+$", text, re.M))
    doubles = len(re.findall(r"(?<=\S)  (?=\S)", text))
    crlf = text.count("\r\n")
    lf = text.count("\n") - crlf
    mixed_indent = bool(re.search(r"^\t* +\t| \t", text, re.M))
    if trailing or doubles or (crlf and lf) or mixed_indent:
        parts = []
        if trailing:
            parts.append(f"{trailing} lines with trailing whitespace")
        if doubles:
            parts.append(f"{doubles} double spaces")
        if crlf and lf:
            parts.append(f"mixed line endings ({crlf} CRLF, {lf} LF)")
        if mixed_indent:
            parts.append("mixed tab/space indentation")
        add("whitespace-pattern", "Whitespace pattern anomalies", "low", 0.8,
            "; ".join(parts) + " — possible whitespace steganography (SNOW) carrier",
            removal="Normalized by normalize_whitespace")

    typo = []
    counts = {"curly quotes": sum(text.count(c) for c in "\u201c\u201d\u2018\u2019"),
              "dashes": sum(text.count(c) for c in "\u2013\u2014"),
              "ellipses": text.count("\u2026"),
              "NBSP": text.count("\u00a0")}
    typo = [f"{v} {k}" for k, v in counts.items() if v]
    if typo:
        add("typographic-variants", "Typographic variants", "info", 1.0,
            ", ".join(typo) + " — style fingerprint, not a mark itself",
            removal="Optional: normalize_typography")

    if re.search(r"</?[a-z]", text, re.I):
        hidden = re.findall(r"<span[^>]*display\s*:\s*none[^>]*>|font-size\s*:\s*0|"
                            r"color\s*:\s*transparent|<div[^>]*hidden", text, re.I)
        if hidden:
            add("invisible-html", "Invisible HTML content", "low", 0.9,
                f"{len(hidden)} hidden-content constructs; e.g. {hidden[0][:100]}",
                removal="Manual review; not auto-removed")

    return {"findings": findings, "notes": notes}
