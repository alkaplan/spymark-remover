import io
import os

import cv2
import numpy as np
from PIL import Image, ImageEnhance

from ..detect.image import _decode_sd_bits

IMAGE_EXTS_PNG = {".png", ".gif", ".webp", ".bmp", ".tif", ".tiff"}


def _squeeze(arr, f):
    h, w = arr.shape[:2]
    small = cv2.resize(arr, (max(8, round(w * f)), max(8, round(h * f))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LANCZOS4)


def _jpeg_roundtrip(arr, q):
    ok, enc = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        return arr
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def _elastic(arr, alpha_px, sigma, rng):
    h, w = arr.shape[:2]
    dx = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32),
                          (0, 0), sigma) * alpha_px
    dy = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32),
                          (0, 0), sigma) * alpha_px
    dx /= max(1e-6, np.abs(dx).max() / max(alpha_px, 1e-6))
    dy /= max(1e-6, np.abs(dy).max() / max(alpha_px, 1e-6))
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(arr, x + dx, y + dy, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def _affine(arr, angle, zoom, dx, dy):
    h, w = arr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, zoom)
    M[0, 2] += dx
    M[1, 2] += dy
    out = cv2.warpAffine(arr, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
    ch, cw = int(h * 0.01), int(w * 0.01)
    out = out[ch:h - ch, cw:w - cw]
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_LANCZOS4)


def _color_nudge(arr, rng):
    img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
    img = ImageEnhance.Brightness(img).enhance(1 + rng.uniform(-0.015, 0.015))
    img = ImageEnhance.Contrast(img).enhance(1 + rng.uniform(-0.02, 0.02))
    img = ImageEnhance.Color(img).enhance(1 + rng.uniform(-0.03, 0.03))
    out = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + rng.choice([-1, 0, 1])) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _fft_notch(arr, peaks):
    if not peaks:
        return arr
    ycc = cv2.cvtColor(arr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    Y = ycc[..., 0]
    h, w = Y.shape
    cy, cx = h // 2, w // 2
    F = np.fft.fftshift(np.fft.fft2(Y))
    mag, ph = np.abs(F), np.angle(F)
    yy, xx = np.ogrid[:h, :w]
    for (y, x) in peaks:
        for (py, px) in ((y, x), (2 * cy - y, 2 * cx - x)):
            if 0 <= py < h and 0 <= px < w:
                disc = (yy - py) ** 2 + (xx - px) ** 2 <= 9
                mag[disc] *= 0.1
    Y2 = np.real(np.fft.ifft2(np.fft.ifftshift(mag * np.exp(1j * ph))))
    ycc[..., 0] = np.clip(Y2, 0, 255)
    return cv2.cvtColor(ycc.astype(np.uint8), cv2.COLOR_YCrCb2BGR)


def _luma_noise(arr, sigma, rng):
    ycc = cv2.cvtColor(arr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    ycc[..., 0] += rng.normal(0, sigma, ycc[..., 0].shape).astype(np.float32)
    ycc[..., 0] = np.clip(ycc[..., 0], 0, 255)
    return cv2.cvtColor(ycc.astype(np.uint8), cv2.COLOR_YCrCb2BGR)


def _psnr(a, b):
    diff = a.astype(np.float64) - b.astype(np.float64)
    mse = float(np.mean(diff ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def clean_image(data: bytes, strength: str = "standard", ctx: dict | None = None):
    ctx = ctx or {}
    img = Image.open(io.BytesIO(data))
    has_alpha = img.mode in ("RGBA", "LA", "PA") or (
        img.mode == "P" and "transparency" in img.info)
    if has_alpha:
        rgba = np.array(img.convert("RGBA"))
        alpha = rgba[..., 3]
        bgr = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2BGR)
    else:
        alpha = None
        bgr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    orig = bgr.copy()

    ext_in = (ctx.get("filename") or "").lower()
    out_png = has_alpha or any(ext_in.endswith(e) for e in IMAGE_EXTS_PNG)

    seed = int.from_bytes(os.urandom(8), "little")
    rng = np.random.default_rng(seed)

    actions = []

    meta_count = ctx.get("_meta_count") or 0
    if meta_count:
        actions.append(f"Stripped {meta_count} metadata tags (EXIF, XMP, IPTC, ICC, PNG text)")
    if ctx.get("_has_c2pa"):
        actions.append("Removed C2PA manifest")

    mask = ctx.get("_printer_mask")
    if mask is not None:
        n_dots = int((mask > 0).sum())
        dil = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        bgr = cv2.inpaint(bgr, dil, 3, cv2.INPAINT_TELEA)
        actions.append(f"Removed {ctx.get('_printer_dot_count', n_dots)} yellow tracking dots")

    strength = strength if strength in ("light", "standard", "heavy") else "standard"

    # geometric stages act on a stacked BGRA array so alpha stays aligned
    def _geo(arr):
        if strength == "light":
            return _squeeze(arr, 0.97)
        if strength == "standard":
            a_, f, rot, z0, z1 = 1.6, 0.90, 0.4, 1.010, 1.02
        else:
            a_, f, rot, z0, z1 = 2.8, 0.82, 0.8, 1.02, 1.04
        arr = _elastic(arr, a_, 40, rng)
        arr = _affine(arr, rng.uniform(-rot, rot), rng.uniform(z0, z1),
                      rng.uniform(-2, 2), rng.uniform(-2, 2))
        return _squeeze(arr, f)

    geo = bgr if alpha is None else np.dstack([bgr, alpha])
    geo = _geo(geo)
    if alpha is not None:
        bgr, alpha = geo[..., :3], geo[..., 3]
    else:
        bgr = geo

    if strength == "light":
        if out_png:
            bgr = _luma_noise(bgr, 0.8, rng)
            actions.append("Applied 97% resize-squeeze + luma noise σ=0.8")
        else:
            bgr = _jpeg_roundtrip(bgr, 92)
            actions.append("Applied 97% resize-squeeze + JPEG q92")
    else:
        if strength == "standard":
            alpha_px, squeeze_f, noise = 1.6, 0.90, 1.5
            chain = (92, 88)
        else:
            alpha_px, squeeze_f, noise = 2.8, 0.82, 3.0
            chain = (88, 84)
        bgr = _color_nudge(bgr, rng)
        bgr = _fft_notch(bgr, ctx.get("_peaks") or [])
        bgr = _luma_noise(bgr, noise, rng)
        if strength == "standard":
            bgr = cv2.bilateralFilter(bgr, 5, 25, 5)
        else:
            ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
            ycc[..., 1] = cv2.medianBlur(ycc[..., 1], 3)
            ycc[..., 2] = cv2.medianBlur(ycc[..., 2], 3)
            bgr = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
        for q in chain:
            bgr = _jpeg_roundtrip(bgr, q)
        desc = (f"elastic {alpha_px}px, affine, {int(squeeze_f*100)}% squeeze, "
                f"FFT notch ({len(ctx.get('_peaks') or [])} peaks), noise σ={noise}, "
                f"JPEG {'→'.join(map(str, chain))}")
        actions.append("Applied " + desc)

    # verify SD watermark gone; retry harder if needed
    if ctx.get("_sd_frac") and ctx["_sd_frac"] >= 0.85 and min(bgr.shape[:2]) >= 256:
        for _ in range(2):
            frac48, _ = _decode_sd_bits(bgr)
            if frac48 is None or frac48 < 0.85:
                break
            bgr = _jpeg_roundtrip(_squeeze(bgr, 0.85), 85)
            actions.append("SD watermark persisted — applied extra 85% squeeze + JPEG q85")

    psnr = _psnr(orig, bgr)

    out_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if out_png:
        if alpha is not None:
            out_img = Image.fromarray(np.dstack([out_rgb, alpha]), "RGBA")
        else:
            out_img = Image.fromarray(out_rgb, "RGB")
        buf = io.BytesIO()
        out_img.save(buf, "PNG")
        out_ext = ".png"
    else:
        out_img = Image.fromarray(out_rgb, "RGB")
        buf = io.BytesIO()
        out_img.save(buf, "JPEG", quality=90)
        out_ext = ".jpg"

    return buf.getvalue(), out_ext, actions, psnr
