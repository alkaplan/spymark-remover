import base64
import io
import os
import re
import subprocess
import tempfile

os.environ.setdefault("NO_TORCH_COMPILE", "1")  # torch inductor needs Python.h; run eager

import cv2
import numpy as np
from scipy import signal as ss

from ..findings import Finding

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".aiff",
              ".aif"}
PURCHASE_KEYS = {"apid", "ownr", "purd", "cprt", "xid ", "atid", "plid", "cnid",
                 "sfid", "akid", "geid", "ufid", "priv"}
UUID_RE = re.compile(r"[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?"
                     r"[0-9a-fA-F]{4}-?[0-9a-fA-F]{8,}")
VORBIS_ID_RE = re.compile(r"^(PURCHASE|TRANSACTION|USER|ACCOUNT|.*UUID|^ID$|"
                          r"CUSTOMER|ORDER)", re.I)

_seal_det = None
_wavmark_model = None


def _ffprobe(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-show_format", path],
            capture_output=True, text=True, timeout=30)
        import json
        return json.loads(out.stdout or "{}")
    except Exception:
        return {}


def _stream_info(probe):
    for s in probe.get("streams", []):
        if s.get("codec_type") == "audio":
            return int(s.get("sample_rate") or 44100), int(s.get("channels") or 1)
    return 44100, 1


def _decode(path, sr=None, channels=None, seconds=None):
    """Decode to float32 numpy (frames, ch)."""
    cmd = ["ffmpeg", "-v", "error"]
    if seconds:
        cmd += ["-t", str(seconds)]
    cmd += ["-i", path, "-f", "f32le", "-acodec", "pcm_f32le"]
    if sr:
        cmd += ["-ar", str(sr)]
    if channels:
        cmd += ["-ac", str(channels)]
    cmd.append("-")
    out = subprocess.run(cmd, capture_output=True, timeout=120)
    if out.returncode != 0 or not out.stdout:
        return None
    raw = np.frombuffer(out.stdout, dtype=np.float32)
    ch = channels or _stream_info(_ffprobe(path))[1] or 1
    if len(raw) % ch:
        raw = raw[: len(raw) - len(raw) % ch]
    return raw.reshape(-1, ch) if ch > 1 else raw[:, None]


def _seal_detector():
    global _seal_det
    if _seal_det is None:
        from audioseal import AudioSeal
        _seal_det = AudioSeal.load_detector("audioseal_detector_16bits")
    return _seal_det


def _seal_prob(sig16k: np.ndarray):
    """Max detection probability over ≤3 windows of 20 s."""
    import torch
    det = _seal_detector()
    sr = 16000
    dur = len(sig16k)
    if dur <= 60 * sr:
        windows = [sig16k]
    else:
        windows = [sig16k[: 20 * sr], sig16k[dur // 2 - 10 * sr: dur // 2 + 10 * sr],
                   sig16k[-20 * sr:]]
    best, msg = 0.0, None
    for w in windows:
        x = torch.from_numpy(np.array(w, dtype=np.float32))[None, None, :]
        result, message = det.detect_watermark(x, 16000)
        p = float(result.flatten().max())
        if p > best:
            best, msg = p, message
    payload = None
    if msg is not None:
        try:
            m = np.asarray(msg.detach().cpu().flatten(), dtype=int)
            payload = int("".join(str(int(b)) for b in m), 2) if m.size else None
        except Exception:
            payload = None
    return best, payload


def _wavmark_decode(sig16k: np.ndarray):
    global _wavmark_model
    import wavmark
    if _wavmark_model is None:
        _wavmark_model = wavmark.load_model()
    dur = len(sig16k)
    if dur > 60 * 16000:
        sig16k = np.concatenate([sig16k[: 20 * 16000],
                                 sig16k[dur // 2 - 10 * 16000: dur // 2 + 10 * 16000],
                                 sig16k[-20 * 16000:]])
    payload, info = wavmark.decode_watermark(_wavmark_model,
                                           np.asarray(sig16k, np.float32),
                                           show_progress=False)
    return payload, info


def _tag_findings(path, data):
    import mutagen
    findings = []
    try:
        f = mutagen.File(path, easy=False)
    except Exception:
        f = None
    if f is None or not getattr(f, "tags", None):
        return findings, 0

    owner, uids, software, other = [], [], [], {}
    purchase, picture = [], False
    n_tags = 0
    for key in f.tags.keys():
        n_tags += 1
        kl = str(key).lower()
        try:
            frames = f.tags.getall(key) if hasattr(f.tags, "getall") else None
        except Exception:
            frames = None
        vals = frames if frames else [f.tags[key]]
        for v in vals:
            vs = str(v)[:200]
            if kl.split(":")[0] in PURCHASE_KEYS or kl in PURCHASE_KEYS:
                purchase.append(f"{key}={vs}")
            elif kl.startswith("priv") or kl == "ufid":
                purchase.append(f"{key}={vs}")
            elif kl.startswith("txxx") or kl in ("woaf", "wxxx") or kl.startswith("w"):
                if "http" in vs or "=" in vs:
                    uids.append(f"{key}={vs}")
                else:
                    other.setdefault("tags", 0)
                    other["tags"] += 1
            elif kl in ("tenc", "tsse", "encoder", "encodedby", "software",
                        "encoded_by", "lavf", "itunsmpb", "encoding_tool"):
                software.append(f"{key}={vs}")
            elif kl.startswith("comm") or kl == "comment":
                if re.search(r"http|id[=:]|uuid", vs, re.I):
                    uids.append(f"{key}={vs}")
                else:
                    other.setdefault("comments", 0)
                    other["comments"] += 1
            elif VORBIS_ID_RE.match(str(key).split("=")[0]):
                uids.append(f"{key}={vs}")
            elif UUID_RE.search(vs):
                uids.append(f"{key}={vs}")
            elif kl in ("artist", "author", "albumartist", "composer",
                        "copyright", "tcop", "tpe1", "tpe2", "owner"):
                owner.append(f"{key}={vs}")
            elif kl in ("apic", "covr", "metadata_block_picture") or "apic" in kl:
                picture = True
            else:
                g = str(key).split(":")[0][:20]
                other[g] = other.get(g, 0) + 1

    def add(fid, name, sev, conf, ev, rem="Stripped via -map_metadata -1 re-encode"):
        findings.append(Finding(id=fid, category="metadata", name=name,
                                severity=sev, confidence=conf, evidence=ev[:200],
                                removable=True, removal=rem))

    if purchase:
        add("purchase-identity", "Purchase/account identity atoms", "high", 1.0,
            "; ".join(purchase[:5]))
    if owner:
        add("owner-identity", "Artist/author tags", "high", 1.0,
            "; ".join(owner[:5]))
    if uids:
        add("unique-ids", "IDs / custom frames", "medium", 0.9,
            "; ".join(uids[:6]))
    if software:
        add("software-tags", "Encoder software tags", "low", 1.0,
            "; ".join(software[:5]))
    if picture:
        add("embedded-picture", "Embedded cover art", "low", 1.0,
            "Cover art carries its own metadata", "Dropped via -vn -map_metadata -1")
    if other:
        add("other-metadata", "Other tags", "info", 1.0,
            ", ".join(f"{k}:{v}" for k, v in sorted(other.items()))
            + f" ({n_tags} tags total)")
    return findings, n_tags


def _ultrasonic(y, sr):
    if sr < 32000 or len(y) < sr:
        return None
    mono = y[:, 0] if y.ndim > 1 else y
    nseg = min(len(mono) // sr, 30)
    if nseg < 2:
        return None
    presence = {}
    for i in range(nseg):
        seg = mono[i * sr:(i + 1) * sr]
        f, p = ss.welch(seg, sr, nperseg=4096)
        hi = f > 16000
        if not hi.any():
            return None
        pf = p[hi]
        ff = f[hi]
        db = 10 * np.log10(pf + 1e-20)
        med = np.median(db)
        mid = (f > 300) & (f < 15000)
        floor = np.median(10 * np.log10(p[mid] + 1e-20)) if mid.any() else med
        # must stand out locally, carry real energy vs the audible band,
        # and exceed an absolute floor (a −60dB residue is not a working beacon)
        peaks = np.nonzero((db > med + 20) & (db > floor + 12) & (db > -55))[0]
        for j in peaks:
            presence[ff[j]] = presence.get(ff[j], 0) + 1
    hits = {fr: c for fr, c in presence.items() if c >= 0.7 * nseg}
    if not hits:
        return None
    freqs = sorted(hits)[:5]
    return Finding(id="ultrasonic-tone", category="tracking",
                   name="Persistent ultrasonic tone", severity="medium",
                   confidence=0.9,
                   evidence="Persistent narrowband energy at "
                            + ", ".join(f"{fr/1000:.2f} kHz" for fr in freqs)
                            + " (>16 kHz, in ≥70% of 1-s windows) — possible "
                              "ultrasonic cross-device beacon",
                   removable=True,
                   removal="Low-pass filtered below 16–18 kHz on clean")


def _periodic_spectrum(y, sr):
    mono = y[:, 0] if y.ndim > 1 else y
    if len(mono) < 4 * sr:
        return None
    f, p = ss.welch(mono, sr, nperseg=4096)
    keep = f > 300
    s = np.log10(p[keep] + 1e-20)
    s = s - s.mean()
    n = len(s)
    if n < 64:
        return None
    ac = np.correlate(s, s, "full")[n - 1:]
    ac /= ac[0]
    lag = np.arange(n) * (f[1] - f[0])
    lo = np.searchsorted(np.arange(n), 8)
    if n <= lo + 1:
        return None
    peak_i = lo + int(np.argmax(ac[lo:min(n, lo + 400)]))
    if ac[peak_i] > 0.6:
        return Finding(id="periodic-spectral-pattern", category="audio",
                       name="Periodic spectral pattern", severity="low",
                       confidence=float(min(1, ac[peak_i])),
                       evidence=f"Comb-like periodicity in long-term spectrum "
                                f"(autocorr {ac[peak_i]:.2f} at lag "
                                f"{peak_i} bins ≈ {peak_i * (f[1]-f[0]):.0f} Hz)",
                       removable=True,
                       removal="Phase jitter + stretch decorrelate the pattern")
    return None


def _spectrogram(y, sr):
    import librosa
    mono = y[:, 0] if y.ndim > 1 else y
    S = np.abs(librosa.stft(mono[: sr * 60], n_fft=1024, hop_length=256))
    db = librosa.amplitude_to_db(S, ref=np.max)
    vis = cv2.normalize(db, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)[::-1]
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_VIRIDIS)
    h, w = vis.shape[:2]
    sc = 512 / max(h, w)
    if sc < 1:
        vis = cv2.resize(vis, (round(w * sc), round(h * sc)),
                         interpolation=cv2.INTER_AREA)
    ok, png = cv2.imencode(".png", vis)
    return "data:image/png;base64," + base64.b64encode(png.tobytes()).decode() if ok else None


def analyze_audio(data: bytes, filename: str = "") -> dict:
    ext = os.path.splitext(filename or "")[1] or ".bin"
    fd, path = tempfile.mkstemp(suffix=ext)
    findings, notes = [], []
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        probe = _ffprobe(path)
        sr, ch = _stream_info(probe)

        tag_findings, n_tags = _tag_findings(path, data)
        findings += tag_findings

        sig16 = _decode(path, sr=16000, channels=1, seconds=60)
        if sig16 is not None:
            try:
                prob, payload = _seal_prob(sig16[:, 0])
                if prob >= 0.5:
                    ev = f"AudioSeal detector probability {prob:.2f}"
                    if payload is not None:
                        ev += f", 16-bit payload 0x{payload:04X}"
                    findings.append(Finding(
                        id="audioseal", category="audio",
                        name="AudioSeal AI watermark", severity="high",
                        confidence=round(prob, 3), evidence=ev, removable=True,
                        removal="Stretch/pitch/phase-jitter degrade it; verified after clean"))
            except Exception as e:
                notes.append(f"AudioSeal check failed: {e}")
            try:
                payload, info = _wavmark_decode(sig16[:, 0])
                if payload is not None:
                    pb = "".join(str(int(b)) for b in np.asarray(payload).ravel())
                    findings.append(Finding(
                        id="wavmark", category="audio",
                        name="WavMark audio watermark", severity="high",
                        confidence=0.95,
                        evidence=f"decoded payload bits {pb[:32]} (info: {str(info)[:80]})",
                        removable=True,
                        removal="Temporal/spectral perturbation destroys the payload"))
            except Exception as e:
                notes.append(f"WavMark check failed: {e}")
        else:
            notes.append("Could not decode audio for watermark analysis")

        y = _decode(path)
        if y is not None:
            f = _ultrasonic(y, sr)
            if f:
                findings.append(f)
            f = _periodic_spectrum(y, sr)
            if f:
                findings.append(f)
            visual = _spectrogram(y, sr)
            dur = len(y) / sr
        else:
            visual, dur = None, None
    finally:
        os.unlink(path)

    return {"findings": findings, "visual": visual, "notes": notes,
            "_sr": sr, "_ch": ch, "_duration": dur, "_n_tags": n_tags,
            "_has_ultrasonic": any(f.id == "ultrasonic-tone" for f in findings),
            "_audioseal_prob": max((f.confidence for f in findings
                                    if f.id == "audioseal"), default=0.0)}
