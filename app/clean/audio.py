import os
import subprocess
import tempfile

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ..detect.audio import _decode, _stream_info, _ffprobe, _seal_prob

LOSSLESS = {".wav": (".wav", ["-c:a", "pcm_s16le"]),
            ".aiff": (".wav", ["-c:a", "pcm_s16le"]),
            ".aif": (".wav", ["-c:a", "pcm_s16le"]),
            ".flac": (".flac", ["-c:a", "flac"])}
LOSSY = {".ogg": (".ogg", ["-c:a", "libvorbis", "-q:a", "6"]),
         ".opus": (".ogg", ["-c:a", "libvorbis", "-q:a", "6"]),
         ".m4a": (".m4a", ["-c:a", "aac", "-b:a", "192k"]),
         ".aac": (".m4a", ["-c:a", "aac", "-b:a", "192k"])}


def _phase_jitter(y, sigma, rng):
    import librosa
    S = librosa.stft(y, n_fft=2048)
    mag, ph = np.abs(S), np.angle(S)
    ph = ph + rng.normal(0, sigma, ph.shape).astype(np.float32)
    S2 = mag * np.exp(1j * ph)
    return librosa.istft(S2, length=len(y))


def _process(y, sr, strength, has_ultrasonic, rng):
    import librosa
    ch = y.shape[1] if y.ndim > 1 else 1
    if strength == "light":
        out = np.stack([librosa.resample(y[:, c], orig_sr=sr,
                                       target_sr=int(sr * 1.01)) for c in range(ch)], -1)
        out = np.stack([librosa.resample(out[:, c], orig_sr=int(sr * 1.01),
                                         target_sr=sr, res_type="soxr_hq")
                        for c in range(ch)], -1)
        out = out * (10 ** (-0.5 / 20))
        desc = "resampled ×1.01→back, −0.5 dB gain"
    else:
        if strength == "standard":
            rate, pitch_hi, jitter, noise_db = (1.006, 1.012), 0.008, 0.15, -70
        else:
            rate, pitch_hi, jitter, noise_db = (1.015, 1.03), 0.018, 0.35, -60
        out = np.stack([
            librosa.effects.time_stretch(y[:, c], rate=float(rng.uniform(*rate)))
            for c in range(ch)], -1)
        p = rng.uniform(0.004, pitch_hi)
        n_steps = 12 * np.log2(1 + p)
        out = np.stack([
            librosa.effects.pitch_shift(out[:, c], sr=sr,
                                        n_steps=float(n_steps))
            for c in range(ch)], -1)
        out = np.stack([_phase_jitter(out[:, c], jitter, rng) for c in range(ch)], -1)
        rms = np.sqrt(np.mean(out ** 2) + 1e-20)
        out += rng.normal(0, rms * 10 ** (noise_db / 20), out.shape).astype(np.float32)
        if strength == "heavy":
            # mild dynamic EQ: random ±1.5 dB tilt over 3 bands
            S = np.fft.rfft(out[:, 0])
            freqs = np.fft.rfftfreq(len(out[:, 0]), 1 / sr)
            bands = np.array_split(freqs, 3)
            tilt = np.zeros_like(freqs)
            for b in bands:
                tilt[b[0]:b[-1] + 1] = rng.uniform(-1.5, 1.5)
            out = np.stack([np.fft.irfft(np.fft.rfft(out[:, c]) * 10 ** (tilt / 20),
                                       n=len(out[:, 0])) for c in range(ch)], -1).astype(np.float32)
        # low-pass
        cutoff = 16000 if strength == "heavy" else (18000 if has_ultrasonic else None)
        if cutoff and sr > 2 * cutoff:
            sos = butter(8, cutoff / (sr / 2), "low", output="sos")
            out = sosfiltfilt(sos, out, axis=0).astype(np.float32)
        desc = f"stretch, pitch shift {p*100:.2f}%, phase jitter σ={jitter}, noise {noise_db} dBFS"
        if cutoff:
            desc += f", low-pass {cutoff//1000} kHz"
        if strength == "heavy":
            desc += ", 3-band random EQ"
    return out.astype(np.float32), desc


def _encode(y, sr, in_path, ext_in):
    ext = ext_in.lower()
    if ext in LOSSLESS:
        out_ext, codec = LOSSLESS[ext]
    elif ext in LOSSY:
        out_ext, codec = LOSSY[ext]
    else:
        out_ext, codec = ".mp3", ["-c:a", "libmp3lame", "-b:a", "192k"]
    fd, raw = tempfile.mkstemp(suffix=".f32")
    os.close(fd)
    fd, outp = tempfile.mkstemp(suffix=out_ext)
    os.close(fd)
    try:
        y.astype(np.float32).tofile(raw)
        ch = y.shape[1] if y.ndim > 1 else 1
        cmd = ["ffmpeg", "-v", "error", "-y",
               "-f", "f32le", "-ar", str(sr), "-ac", str(ch), "-i", raw,
               "-map_metadata", "-1", "-vn", "-fflags", "+bitexact"] + codec + [outp]
        r = subprocess.run(cmd, capture_output=True, timeout=300)
        if r.returncode != 0:
            return None, out_ext
        return open(outp, "rb").read(), out_ext
    finally:
        for p in (raw, outp):
            try:
                os.unlink(p)
            except OSError:
                pass


def clean_audio(data: bytes, strength: str = "standard", ctx: dict | None = None):
    ctx = ctx or {}
    ext = os.path.splitext(ctx.get("filename") or "x.wav")[1] or ".wav"
    fd, path = tempfile.mkstemp(suffix=ext)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        sr, ch = _stream_info(_ffprobe(path))
        y = _decode(path)
        if y is None:
            raise ValueError("undecodable audio")
        in_dur = len(y) / sr
        rng = np.random.default_rng(int.from_bytes(os.urandom(8), "little"))
        strength = strength if strength in ("light", "standard", "heavy") else "standard"
        actions = []
        if ctx.get("_n_tags"):
            actions.append(f"Stripped {ctx['_n_tags']} metadata tags + cover art")
        else:
            actions.append("Stripped metadata tags + cover art")
        out, desc = _process(y, sr, strength, ctx.get("_has_ultrasonic"), rng)
        actions.append("Applied " + desc)

        # AudioSeal persistence → escalate once
        if strength != "heavy" and ctx.get("_audioseal_prob", 0) >= 0.5:
            try:
                import librosa
                s16 = np.stack([librosa.resample(out[:, c], orig_sr=sr, target_sr=16000)
                                for c in range(out.shape[1])], -1)
                prob, _ = _seal_prob(s16[:, 0])
                if prob >= 0.5:
                    out, desc = _process(y, sr,
                                         "heavy" if strength == "standard" else "standard",
                                         ctx.get("_has_ultrasonic"), rng)
                    actions.append(f"AudioSeal persisted at {prob:.2f} — "
                                   f"escalated one step ({desc})")
            except Exception:
                pass

        buf, out_ext = _encode(out, sr, path, ext)
        if buf is None:
            raise ValueError("ffmpeg encode failed")
        snr = None
        if len(out) == len(y):
            noise = y - out
            snr = float(10 * np.log10(np.mean(y ** 2) / (np.mean(noise ** 2) + 1e-20)))
        metrics = {"duration_s": round(in_dur, 2),
                   "snr_db": round(snr, 1) if snr is not None else None,
                   "in_bytes": len(data), "out_bytes": len(buf)}
        return buf, out_ext, actions, metrics
    finally:
        os.unlink(path)
