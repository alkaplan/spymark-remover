import base64
import io
import json
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from ..findings import Finding

SDXL_BITS = [int(b) for b in bin(0b101100111110110010010000011110111011000110011110)[2:]]
SDV2_BYTES = b"SDV2"

SERIAL_TAGS = {
    "bodyserialnumber", "serialnumber", "lensserialnumber",
    "internalserialnumber", "cameraserialnumber", "lensid",
}
IDENTITY_TAGS = {
    "ownername", "artist", "creator", "by-line", "copyright",
    "cameraownername", "author", "creatorcontactinfo", "personinimage",
}
UNIQUEID_TAGS = {
    "imageuniqueid", "documentid", "instanceid", "originaldocumentid",
    "preservedfilename", "originalrawfilename",
}
SOFTWARE_TAGS = {"software", "creatortool", "processingsoftware", "historysoftwareagent"}
AI_RE = re.compile(
    r"Google AI|Made with Google AI|Gemini|DALL|OpenAI|Midjourney|Firefly|"
    r"Stable Diffusion|ComfyUI|FLUX|Imagen|Grok", re.I)
UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
C2PA_STR_RE = re.compile(
    rb"c2pa\.[a-z_.]+|urn:uuid:[0-9a-f-]+|\"?issuer\"?\s*[:=][^,}\"']*|CN=[^,\x00]+|O=[^,\x00]+|"
    rb"(Adobe|Google|OpenAI|Microsoft|Truepic|Leica|Samsung|Stability|Black Forest|"
    rb"ByteDance|Volcano)[A-Za-z ]*")


def _trunc(s, n=200):
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def _run_exiftool(data: bytes, suffix: str):
    if not shutil.which("exiftool"):
        return None
    fd, path = tempfile.mkstemp(suffix=suffix or ".bin")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        out = subprocess.run(
            ["exiftool", "-j", "-G", "-n", "-a", path],
            capture_output=True, text=True, timeout=30)
        arr = json.loads(out.stdout or "[]")
        return arr[0] if arr else {}
    except Exception:
        return None
    finally:
        os.unlink(path)


SKIP_GROUPS = {"File", "Composite", "ExifTool"}
SKIP_TAGS = {"imagewidth", "imageheight", "bitdepth", "colortype", "compression",
             "filter", "interlace", "imagesize", "megapixels", "pixelsperunitx",
             "pixelsperunity", "pixelunits", "xresolution", "yresolution",
             "resolutionunit", "ycbcrpositioning", "ycbcrsubsampling",
             "bitspersample", "colorcomponents", "encodingprocess", "exifbyteorder",
             "backgroundcolor", "significantbits", "imagedata"}


def _flat_tags(raw: dict):
    """Return list of (group, tag, value), excluding structural/file-system tags."""
    tags = []
    for k, v in (raw or {}).items():
        if ":" in k:
            g, t = k.split(":", 1)
        else:
            g, t = "", k
        if t == "SourceFile" or g in SKIP_GROUPS or t.lower() in SKIP_TAGS:
            continue
        tags.append((g, t, v))
    return tags


def _pil_fallback_tags(data: bytes):
    tags = []
    try:
        img = Image.open(io.BytesIO(data))
        for k, v in img.getexif().items():
            tags.append(("EXIF", str(k), v))
        for k, v in (img.info or {}).items():
            tags.append((img.format or "", str(k), v))
    except Exception:
        pass
    return tags


def _metadata_findings(data: bytes, filename: str, notes: list):
    suffix = os.path.splitext(filename or "")[1] or ".img"
    raw = _run_exiftool(data, suffix)
    if raw is None:
        tags = _pil_fallback_tags(data)
        notes.append("exiftool unavailable; used Pillow fallback (reduced metadata coverage)")
    else:
        tags = _flat_tags(raw)

    gps, serials, identity, uids, software, times = [], [], [], [], [], []
    ai_evidence, other = [], {}
    tag_total = len(tags)

    for group, tag, val in tags:
        tl = tag.lower()
        vs = str(val)
        if tl.startswith("gps"):
            gps.append(f"{tag}={_trunc(vs, 40)}")
            continue
        if tl in SERIAL_TAGS:
            serials.append(f"{tag}={_trunc(vs, 40)}")
            continue
        if tl in IDENTITY_TAGS or (tl == "creator" and group.lower() in ("dc", "xmp", "xmp-dc")):
            identity.append(f"{tag}={_trunc(vs, 40)}")
            continue
        if tl in UNIQUEID_TAGS or "derivedfrom" in tl or tl.startswith("history") and "software" not in tl:
            uids.append(f"{tag}={_trunc(vs, 40)}")
            continue
        if tl in SOFTWARE_TAGS:
            software.append(f"{tag}={_trunc(vs, 40)}")
            continue
        if "date" in tl or tl.endswith("time"):
            times.append(f"{tag}={_trunc(vs, 40)}")
            continue
        other[group or "?"] = other.get(group or "?", 0) + 1

        # AI generator markers (value-driven, checked against every tag)
        if "digitalsourcetype" in tl and ("trainedalgorithmicmedia" in vs.lower()
                                        or "compositewithtrainedalgorithmicmedia" in vs.lower()):
            ai_evidence.append(f"{tag}={_trunc(vs, 80)}")
        if tl == "aisystemused" and vs:
            ai_evidence.append(f"AISystemUsed={_trunc(vs, 80)}")
        if AI_RE.search(vs) and tl in {"credit", "description", "software", "creatortool",
                                       "title", "subject", "comment", "parameters",
                                       "prompt", "workflow", "usercomment", "imagedescription",
                                       "signature", "artist", "make", "model", "historysoftwareagent"}:
            ai_evidence.append(f"{tag}={_trunc(vs, 80)}")
        if tl in {"parameters", "prompt", "workflow", "comment"} and (
                "Steps:" in vs or "Sampler" in vs or "steps" in vs and "seed" in vs):
            ai_evidence.append(f"{tag}: {_trunc(vs, 100)}")
        if "hf-job-id" in tl or "hf-job-id" in vs.lower():
            ai_evidence.append(f"{tag}={_trunc(vs, 80)}")
        if tl == "signature":
            ai_evidence.append(f"Signature={_trunc(vs, 60)}")
        if tl == "artist" and UUID_RE.match(vs.strip()):
            ai_evidence.append(f"Artist (UUID-like)={vs.strip()}")

    # TC260 AIGC label: XMP/PNG 'AIGC' key or a JPEG blob containing AIGC JSON
    aigc_blob = re.search(rb'"?AIGC"?\s*[:={][^\x00]{0,400}?(Label|ContentProducer|ProduceID)', data)
    if aigc_blob or any(t.lower() == "aigc" for _, t, _ in tags):
        ev = _trunc(aigc_blob.group(0).decode("utf-8", "replace"), 120) if aigc_blob else "AIGC tag present"
        ai_evidence.append(f"TC260 AIGC label: {ev}")

    findings = []

    def add(fid, name, sev, ev, rem=True, removal="Stripped with all metadata on re-encode"):
        ev = list(dict.fromkeys(ev))  # dedupe identical Tag=Value pairs
        findings.append(Finding(id=fid, category="metadata", name=name, severity=sev,
                                confidence=1.0, evidence=_trunc("; ".join(ev)),
                                removable=rem, removal=removal))

    if gps:
        add("exif-gps", "GPS location metadata", "high", gps)
    if serials:
        add("device-serial", "Device serial number", "high", serials)
    if identity:
        add("owner-identity", "Owner/creator identity", "high", identity)
    if uids:
        add("unique-ids", "Unique document/image IDs", "medium", uids)
    if software:
        add("software-tags", "Generator software tags", "low", software)
    if times:
        add("timestamps", "Creation timestamps", "low", times[:8])
    if ai_evidence:
        add("ai-generator-label", "AI generator label", "medium", ai_evidence)
    if other:
        ev = ", ".join(f"{g}:{c}" for g, c in sorted(other.items()))
        add("other-metadata", "Other metadata", "info", [f"{ev} ({tag_total} tags total)"])

    return findings, tag_total


def _jpeg_segments(data: bytes):
    """Yield (marker, payload) for JPEG APPn segments."""
    if data[:2] != b"\xff\xd8":
        return
    i = 2
    n = len(data)
    while i + 4 <= n:
        if data[i] != 0xFF:
            break
        marker = data[i + 1]
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        seglen = struct.unpack(">H", data[i + 2:i + 4])[0]
        payload = data[i + 4:i + 2 + seglen]
        yield marker, payload
        if marker == 0xDA:  # SOS: stop
            break
        i += 2 + seglen


def _png_chunks(data: bytes):
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return
    i = 8
    n = len(data)
    while i + 12 <= n:
        length = struct.unpack(">I", data[i:i + 4])[0]
        ctype = data[i + 4:i + 8]
        yield ctype, data[i + 8:i + 8 + length]
        i += 12 + length
        if ctype == b"IEND":
            break


def _webp_chunks(data: bytes):
    if data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        return
    i = 12
    n = len(data)
    while i + 8 <= n:
        ctype = data[i:i + 4]
        length = struct.unpack("<I", data[i + 4:i + 8])[0]
        yield ctype, data[i + 8:i + 8 + length]
        i += 8 + length + (length & 1)


def _c2pa_evidence(blob: bytes):
    hits = C2PA_STR_RE.findall(blob)
    out = []
    for h in hits[:10]:
        if isinstance(h, tuple):
            h = b"".join(h)
        out.append(h.decode("utf-8", "replace"))
    return out[:5]


def _c2pa_findings(data: bytes, tags_flat):
    findings = []
    manifest = None
    for marker, payload in _jpeg_segments(data):
        if marker == 0xEB and (b"jumb" in payload or b"c2pa" in payload):
            manifest = payload
            break
    if manifest is None:
        for ctype, payload in _png_chunks(data) or []:
            if ctype == b"caBX":
                manifest = payload
                break
    if manifest is None:
        for ctype, payload in _webp_chunks(data) or []:
            if ctype == b"C2PA":
                manifest = payload
                break

    jumbf_tags = [t for g, t, _ in tags_flat if g.upper().startswith("JUMBF")]

    if manifest is not None or jumbf_tags:
        ev = []
        if manifest is not None:
            ev.append(f"manifest {len(manifest)/1024:.1f} KB")
            ev += _c2pa_evidence(manifest)
        if jumbf_tags:
            ev.append(f"{len(jumbf_tags)} JUMBF tags")
        findings.append(Finding(
            id="c2pa-manifest", category="provenance", name="C2PA Content Credentials",
            severity="high", confidence=1.0, evidence=_trunc("; ".join(ev)),
            removable=True,
            removal="Dropped on re-encode (manifest segment not copied)"))
        blob = manifest or b" ".join(t.encode() for t in jumbf_tags)
        if re.search(rb"TrustMark|Digimarc|Imatag|Steg\.?AI", blob, re.I):
            findings.append(Finding(
                id="c2pa-soft-binding", category="provenance",
                name="C2PA soft-binding watermark pointer", severity="info",
                confidence=0.9,
                evidence="Manifest references a soft-binding vendor (TrustMark/Digimarc/Imatag/Steg.AI) — an invisible watermark may be present",
                removable=False,
                removal="Pixel transforms degrade soft-binding watermarks but cannot be verified"))

    # remote manifest reference in metadata
    remote = []
    for g, t, v in tags_flat:
        if t.lower() == "provenance" or re.search(r"c2pa|contentcredentials", str(v), re.I):
            remote.append(f"{t}={_trunc(v, 80)}")
    if remote:
        findings.append(Finding(
            id="c2pa-remote-manifest", category="provenance",
            name="Remote C2PA manifest reference", severity="medium", confidence=0.9,
            evidence=_trunc("; ".join(remote[:3])), removable=True,
            removal="Stripped with XMP metadata"))
    return findings


def _decode_sd_bits(bgr):
    """Return (frac48, frac_sdV2) bit-match fractions, or (None, None)."""
    try:
        from imwatermark import WatermarkDecoder
        frac48 = None
        try:
            dec = WatermarkDecoder("bits", len(SDXL_BITS))
            bits = dec.decode(bgr, "dwtDct")
            bits = [int(b) for b in np.asarray(bits).ravel().tolist()]
            if len(bits) == len(SDXL_BITS):
                frac48 = 1 - sum(a != b for a, b in zip(bits, SDXL_BITS)) / len(SDXL_BITS)
        except Exception:
            pass
        frac_sd = None
        try:
            dec = WatermarkDecoder("bytes", 32)
            out = dec.decode(bgr, "dwtDct")
            ob = bytes(bytearray(out))
            if len(ob) >= len(SDV2_BYTES):
                ob = ob[:len(SDV2_BYTES)]
                ob_bits = [int(b) for c in ob for b in f"{c:08b}"]
                sb_bits = [int(b) for c in SDV2_BYTES for b in f"{c:08b}"]
                frac_sd = 1 - sum(a != b for a, b in zip(ob_bits, sb_bits)) / len(sb_bits)
        except Exception:
            pass
        return frac48, frac_sd
    except Exception:
        return None, None


def _sd_watermark(bgr):
    h, w = bgr.shape[:2]
    if min(h, w) < 256:
        return None, None
    frac48, frac_sd = _decode_sd_bits(bgr)
    best = max(x for x in (frac48, frac_sd) if x is not None) if (frac48 or frac_sd) else None
    if best is None:
        return None, None
    ev = []
    if frac48 is not None and frac48 >= 0.7:
        ev.append(f"SDXL 48-bit payload matched {round(frac48*48)}/48 bits")
    if frac_sd is not None and frac_sd >= 0.7:
        ev.append(f"SDV2 32-bit payload matched {round(frac_sd*32)}/32 bits")
    if best >= 0.85:
        return Finding(id="sd-dwtdct", category="pixel",
                       name="Stable Diffusion invisible watermark", severity="high",
                       confidence=best, evidence="; ".join(ev), removable=True,
                       removal="Warp + resample + recompress destroys the DWT-DCT payload"), best
    if best >= 0.70:
        return Finding(id="sd-dwtdct-weak", category="pixel",
                       name="Possible SD watermark (weak match)", severity="low",
                       confidence=best, evidence="; ".join(ev), removable=True,
                       removal="Same transforms reduce payload further"), best
    return None, best


def _spectrum_analysis(bgr):
    """Return (finding_or_None, peaks, spectrum_png_b64)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    resid = gray - cv2.medianBlur(gray.astype(np.float32), 5).astype(np.float64)
    F = np.abs(np.fft.fftshift(np.fft.fft2(resid)))
    log = np.log1p(F)
    h, w = log.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dc = (Y - cy) ** 2 + (X - cx) ** 2 <= 64
    mask = ~(dc | (Y == cy) | (X == cx))
    bg = ndimage.median_filter(log, size=21)
    diff = log - bg
    vals = diff[mask]
    mad = np.median(np.abs(vals - np.median(vals)))
    rstd = max(1.4826 * mad, 1e-9)
    z = diff / rstd
    z[~mask] = 0

    loc = (z == ndimage.maximum_filter(z, size=3)) & (z > 6)
    ys, xs = np.nonzero(loc)
    pts = sorted(zip(ys.tolist(), xs.tolist()), key=lambda p: -z[p])
    used, pairs = set(), []
    for (y, x) in pts:
        if (y, x) in used:
            continue
        py, px = 2 * cy - y, 2 * cx - x
        if 0 <= py < h and 0 <= px < w and z[py, px] > 6:
            pairs.append((y, x, z[y, x]))
            used.add((y, x))
            used.add((py, px))
    n = len(pairs)
    mean_z = float(np.mean([p[2] for p in pairs])) if pairs else 0.0
    score = min(1.0, n / 6) * min(1.0, mean_z / 12) if n else 0.0

    peaks = [(int(y), int(x)) for y, x, _ in pairs]

    finding = None
    if n >= 3:
        y, x, zp = pairs[0]
        fx, fy = abs(x - cx) / w, abs(y - cy) / h
        finding = Finding(
            id="periodic-carrier", category="pixel",
            name="Periodic carrier signal", severity="medium",
            confidence=round(float(min(1, score + 0.4)), 3),
            evidence=f"{n} symmetric spectral peaks (strongest at fx={fx:.3f}, fy={fy:.3f} cycles/px, z={zp:.1f})",
            removable=True,
            removal="FFT notch attenuates peak frequencies on luma")

    vis = cv2.normalize(log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_VIRIDIS)
    for y, x, _ in pairs:
        cv2.circle(vis, (x, y), 7, (0, 0, 255), 1)
    ok, png = cv2.imencode(".png", vis)
    spec64 = "data:image/png;base64," + base64.b64encode(png.tobytes()).decode() if ok else None
    return finding, peaks, spec64


def _printer_dots(bgr):
    h, w = bgr.shape[:2]
    if min(h, w) < 600:
        return None, None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.int16)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    bg = cv2.blur(rgb.astype(np.float32), (31, 31))
    yellow = (r > 150) & (g > 150) & (b < np.minimum(r, g) - 50) & (bg[..., 0] > 180)
    mask = (yellow.astype(np.uint8)) * 255
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    good = []
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if 1 <= area <= 40 and cw <= 8 and ch <= 8:
            good.append((cents[i][0], cents[i][1]))
    if len(good) < 40:
        return None, None
    pts = np.array(good)
    quads = set()
    for x, y in pts:
        quads.add((x >= w / 2, y >= h / 2))
    if len(quads) < 4:
        return None, None
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    spacing = float(np.median(d.min(axis=1)))
    return Finding(
        id="printer-dots", category="tracking", name="Printer tracking dots",
        severity="high", confidence=0.9,
        evidence=f"{len(pts)} faint yellow dots, median spacing {spacing:.0f} px, present in all 4 quadrants",
        removable=True, removal="Inpainted (TELEA) before pixel transforms"), mask


def _residual_visual(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    resid = (gray - cv2.medianBlur(gray, 5)) * 12 + 128
    vis = np.clip(resid, 0, 255).astype(np.uint8)
    h, w = vis.shape
    scale = 512 / max(h, w)
    if scale < 1:
        vis = cv2.resize(vis, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    ok, png = cv2.imencode(".png", vis)
    return "data:image/png;base64," + base64.b64encode(png.tobytes()).decode() if ok else None


def analyze_image(data: bytes, filename: str = "") -> dict:
    img = Image.open(io.BytesIO(data))
    bgr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    notes = []

    meta_findings, meta_count = _metadata_findings(data, filename, notes)
    tags_raw = _run_exiftool(data, os.path.splitext(filename or "")[1] or ".img") or {}
    tags_flat = _flat_tags(tags_raw)

    findings = meta_findings
    findings += _c2pa_findings(data, tags_flat)

    sd_finding, sd_frac = _sd_watermark(bgr)
    if sd_finding:
        findings.append(sd_finding)

    periodic_finding, peaks, spectrum = _spectrum_analysis(bgr)
    if periodic_finding:
        findings.append(periodic_finding)

    dots_finding, dots_mask = _printer_dots(bgr)
    if dots_finding:
        findings.append(dots_finding)

    return {
        "findings": findings,
        "visual": _residual_visual(bgr),
        "visual_spectrum": spectrum,
        "notes": notes,
        "_peaks": peaks,
        "_printer_mask": dots_mask,
        "_sd_frac": sd_frac,
        "_meta_count": meta_count,
        "_has_c2pa": any(f.id == "c2pa-manifest" for f in findings),
    }
