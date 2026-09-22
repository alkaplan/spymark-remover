import io
import os
import struct
import subprocess
import tempfile
import zlib

import cv2
import numpy as np
import pytest
from PIL import Image

from app.detect.image import analyze_image, SDXL_BITS
from app.clean.image import clean_image


def noisy_image(h=512, w=512, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:h, :w]
    base = (xx / w * 180 + yy / h * 60).astype(np.float32)
    img = np.stack([base, 200 - base / 2, 80 + base / 3], -1)
    img += rng.normal(0, 12, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def png_bytes(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(buf, "PNG")
    return buf.getvalue()


def jpeg_with_exif(rgb):
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        Image.fromarray(rgb, "RGB").save(path, "PNG")
        subprocess.run(
            ["exiftool", "-overwrite_original",
             "-GPSLatitude=48.85", "-Artist=testartist",
             "-CameraSerialNumber=123456", path],
            check=True, capture_output=True)
        return open(path, "rb").read()
    finally:
        os.unlink(path)


def png_with_cabx(rgb):
    """PNG bytes with an injected caBX C2PA-like chunk before IEND."""
    raw = png_bytes(rgb)
    iend = raw.rfind(b"IEND") - 4
    payload = b"jumb\x00c2pa c2pa.claim manifest O=Adobe Inc. CN=Adobe Content Authenticity"
    chunk = struct.pack(">I", len(payload)) + b"caBX" + payload
    chunk += struct.pack(">I", zlib.crc32(b"caBX" + payload) & 0xFFFFFFFF)
    return raw[:iend] + chunk + raw[iend:]


def ids(rep):
    return {f.id for f in rep["findings"]}


@pytest.mark.skipif(not subprocess.run(["which", "exiftool"], capture_output=True).stdout,
                    reason="exiftool missing")
def test_metadata_and_sd_watermark():
    rgb = noisy_image()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    from imwatermark import WatermarkEncoder
    enc = WatermarkEncoder()
    enc.set_watermark("bits", SDXL_BITS)
    wm = enc.encode(bgr, "dwtDct")
    rgb = cv2.cvtColor(wm, cv2.COLOR_BGR2RGB)
    data = jpeg_with_exif(rgb)

    rep = analyze_image(data, "photo.png")
    fids = ids(rep)
    assert "exif-gps" in fids
    assert "device-serial" in fids
    assert "owner-identity" in fids
    assert "sd-dwtdct" in fids

    ctx = {k: rep[k] for k in rep if k.startswith("_")}
    ctx["filename"] = "photo.png"
    out, ext, actions, psnr = clean_image(data, "standard", ctx)
    assert ext == ".png"
    after = analyze_image(out, "out.png")
    fids2 = ids(after)
    assert not {"exif-gps", "device-serial", "owner-identity", "other-metadata"} & fids2
    assert "sd-dwtdct" not in fids2


def test_c2pa_chunk():
    data = png_with_cabx(noisy_image(seed=1))
    rep = analyze_image(data, "ai.png")
    assert "c2pa-manifest" in ids(rep)
    ctx = {k: rep[k] for k in rep if k.startswith("_")}
    ctx["filename"] = "ai.png"
    out, ext, _, _ = clean_image(data, "light", ctx)
    after = analyze_image(out, "out.png")
    assert "c2pa-manifest" not in ids(after)


def test_periodic_carrier():
    rgb = noisy_image(seed=2)
    yy, xx = np.mgrid[:512, :512]
    carrier = (1.5 * np.sin(2 * np.pi * (xx * 0.13 + yy * 0.07))
               + 1.5 * np.sin(2 * np.pi * (xx * 0.31 - yy * 0.19))
               + 1.5 * np.sin(2 * np.pi * (xx * 0.05 + yy * 0.41))).astype(np.float32)
    rgb = np.clip(rgb.astype(np.float32) + carrier[..., None], 0, 255).astype(np.uint8)
    data = png_bytes(rgb)
    rep = analyze_image(data, "s.png")
    assert "periodic-carrier" in ids(rep)
    assert rep["_peaks"]
    ctx = {k: rep[k] for k in rep if k.startswith("_")}
    ctx["filename"] = "s.png"
    out, _, _, _ = clean_image(data, "standard", ctx)
    after = analyze_image(out, "out.png")
    assert "periodic-carrier" not in ids(after)


def test_no_false_positive_natural():
    # smooth gradient + moderate noise should NOT flag periodic-carrier
    data = png_bytes(noisy_image(seed=3))
    rep = analyze_image(data, "n.png")
    assert "periodic-carrier" not in ids(rep)


def test_printer_dots():
    img = np.full((1200, 1600, 3), 255, np.uint8)
    # slight paper texture
    rng = np.random.default_rng(4)
    img = np.clip(img.astype(np.float32) + rng.normal(0, 1.5, img.shape), 0, 255).astype(np.uint8)
    for y in range(30, 1190, 60):
        for x in range(30, 1590, 60):
            img[y:y + 2, x:x + 2] = (255, 255, 0)
    data = png_bytes(img)
    rep = analyze_image(data, "scan.png")
    assert "printer-dots" in ids(rep)
    ctx = {k: rep[k] for k in rep if k.startswith("_")}
    ctx["filename"] = "scan.png"
    out, _, actions, _ = clean_image(data, "light", ctx)
    after = analyze_image(out, "out.png")
    assert "printer-dots" not in ids(after)
    assert any("tracking dots" in a for a in actions)
