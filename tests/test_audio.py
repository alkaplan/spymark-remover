import io
import os
import subprocess
import tempfile

import numpy as np
import pytest

from app.detect.audio import analyze_audio, _seal_prob, _decode
from app.clean.audio import clean_audio


def noise_audio(sr=16000, dur=8, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(dur * sr) / sr
    y = rng.normal(0, 0.05, dur * sr).astype(np.float32)
    y += 0.1 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.sin(2 * np.pi * 990 * t)
    return np.clip(y, -1, 1)


def write_wav(y, sr, path):
    subprocess.run(["ffmpeg", "-v", "error", "-y",
                    "-f", "f32le", "-ar", str(sr), "-ac", "1",
                    "-i", "pipe:0", path], input=y.astype(np.float32).tobytes(),
                   check=True, capture_output=True)


def test_audioseal_detect_and_clean():
    import torch
    from audioseal import AudioSeal
    y = noise_audio()
    wav = torch.from_numpy(y)[None, None, :]
    gen = AudioSeal.load_generator("audioseal_wm_16bits")
    wmed = (wav + gen.get_watermark(wav, 16000)).detach()
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    write_wav(wmed[0, 0].numpy().astype(np.float32), 16000, path)
    data = open(path, "rb").read()
    os.unlink(path)

    rep = analyze_audio(data, "clip.wav")
    fids = {f.id for f in rep["findings"]}
    assert "audioseal" in fids, fids

    ctx = {k: rep[k] for k in rep if k.startswith("_")}
    ctx["filename"] = "clip.wav"
    out, ext, actions, metrics = clean_audio(data, "standard", ctx)
    assert ext == ".wav"
    fd, p2 = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    open(p2, "wb").write(out)
    sig16 = _decode(p2, sr=16000, channels=1)
    os.unlink(p2)
    prob, _ = _seal_prob(sig16[:, 0])
    if prob >= 0.5:  # escalate check with heavy directly
        rep2 = analyze_audio(data, "clip.wav")
        ctx2 = {k: rep2[k] for k in rep2 if k.startswith("_")}
        ctx2["filename"] = "clip.wav"
        out, _, _, _ = clean_audio(data, "heavy", ctx2)
        fd, p3 = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        open(p3, "wb").write(out)
        sig16 = _decode(p3, sr=16000, channels=1)
        os.unlink(p3)
        prob, _ = _seal_prob(sig16[:, 0])
    assert prob < 0.5


def test_mp3_tags():
    import mutagen.id3 as id3
    y = noise_audio(sr=44100, dur=2)
    fd, wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    fd, mp3 = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    write_wav(y, 44100, wav)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", wav,
                    "-c:a", "libmp3lame", "-b:a", "128k", mp3],
                   check=True, capture_output=True)
    tags = id3.ID3()
    tags.add(id3.PRIV(owner=b"www.amazon.com", data=b"x"))
    tags.add(id3.UFID(owner=b"http://example.com", data=b"order-123"))
    tags.save(mp3)
    data = open(mp3, "rb").read()
    os.unlink(wav); os.unlink(mp3)

    rep = analyze_audio(data, "song.mp3")
    fids = {f.id for f in rep["findings"]}
    assert "purchase-identity" in fids or "unique-ids" in fids, fids
    ctx = {k: rep[k] for k in rep if k.startswith("_")}
    ctx["filename"] = "song.mp3"
    out, ext, actions, metrics = clean_audio(data, "light", ctx)
    assert ext == ".mp3"
    after = analyze_audio(out, "song.mp3")
    fids2 = {f.id for f in after["findings"]}
    assert not {"purchase-identity", "unique-ids", "owner-identity",
                "software-tags", "embedded-picture", "other-metadata"} & fids2


def test_ultrasonic_tone():
    sr = 44100
    t = np.arange(4 * sr) / sr
    rng = np.random.default_rng(1)
    y = (rng.normal(0, 0.05, 4 * sr)
         + 0.03 * np.sin(2 * np.pi * 18500 * t)).astype(np.float32)
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    write_wav(y, sr, path)
    data = open(path, "rb").read()
    os.unlink(path)
    rep = analyze_audio(data, "u.wav")
    assert "ultrasonic-tone" in {f.id for f in rep["findings"]}
    ctx = {k: rep[k] for k in rep if k.startswith("_")}
    ctx["filename"] = "u.wav"
    out, _, _, _ = clean_audio(data, "standard", ctx)
    fd, p2 = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    open(p2, "wb").write(out)
    after = analyze_audio(open(p2, "rb").read(), "u.wav")
    os.unlink(p2)
    assert "ultrasonic-tone" not in {f.id for f in after["findings"]}
