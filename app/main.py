import io
import os
import time
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .detect.image import analyze_image
from .detect.text import analyze_text
from .detect.audio import analyze_audio, AUDIO_EXTS
from .detect.video import analyze_video, VIDEO_EXTS
from .detect.document import analyze_document, ooxml_kind
from .clean.image import clean_image
from .clean.text import clean_text, DEFAULTS as TEXT_DEFAULTS
from .clean.audio import clean_audio
from .clean.video import clean_video
from .clean.document import clean_document

BASE = Path(__file__).resolve().parent.parent
WORK = BASE / ".work"
WORK.mkdir(exist_ok=True)
STATIC = Path(__file__).resolve().parent / "static"

MAX_UPLOAD = 200 * 1024 * 1024
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif", ".bmp",
              ".gif", ".avif", ".heic"}
MAGICS = [(b"\x89PNG\r\n\x1a\n", "image"), (b"\xff\xd8\xff", "image"),
          (b"RIFF", "image"), (b"GIF8", "image"), (b"BM", "image"),
          (b"II*\x00", "image"), (b"MM\x00*", "image")]

app = FastAPI(title="Spymark Remover")


def _purge_work():
    now = time.time()
    for p in WORK.iterdir():
        try:
            if now - p.stat().st_mtime > 3600:
                p.unlink()
        except OSError:
            pass


AUDIO_EXTS_S = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac",
                ".aiff", ".aif"}
DOC_EXTS_S = {".pdf", ".docx", ".xlsx", ".pptx"}


def _kind(data: bytes, filename: str):
    ext = os.path.splitext(filename or "")[1].lower()
    # magic first
    for magic, kind in MAGICS:
        if data.startswith(magic):
            if magic == b"RIFF":
                sub = data[8:12]
                if sub == b"WEBP":
                    return "image"
                if sub == b"WAVE":
                    return "audio"
                if sub == b"AVI ":
                    return "video"
                continue
            return kind
    if data[:4] == b"fLaC" or data[:3] == b"ID3" or data[:2] == b"\xff\xfb" \
            or data[:4] == b"OggS":
        return "audio"
    if data[:4] == b"FORM" and data[8:12] in (b"AIFF", b"AIFC"):
        return "audio"
    if data[:4] == b"\x1aE\xdf\xa3":
        return "video"  # mkv/webm EBML
    if data[:4] == b"%PDF":
        return "document"
    if data[:4] == b"PK\x03\x04":
        try:
            import zipfile, io as _io
            with zipfile.ZipFile(_io.BytesIO(data)) as zf:
                if ooxml_kind(zf):
                    return "document"
        except Exception:
            pass
    if data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand.strip(b" \x00").lower().startswith(b"m4a") or ext == ".m4a":
            return "audio"
        return "video"
    if ext in IMAGE_EXTS:
        return "image"
    if ext in AUDIO_EXTS_S:
        return "audio"
    if ext in VIDEO_EXTS:
        return "video"
    if ext in DOC_EXTS_S:
        return "document"
    try:
        data[:1_000_000].decode("utf-8")
        return "text"
    except UnicodeDecodeError:
        return None


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"ok": True, "paraphrase_available": bool(os.environ.get("ANTHROPIC_API_KEY"))}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "File exceeds 200 MB limit")
    kind = _kind(data, file.filename or "")
    if kind == "image":
        rep = analyze_image(data, file.filename or "")
        return {"kind": "image", "filename": file.filename,
                "findings": rep["findings"], "visual": rep["visual"],
                "visual_spectrum": rep["visual_spectrum"], "notes": rep["notes"]}
    if kind in ("audio", "video", "document"):
        fn = {"audio": analyze_audio, "video": analyze_video,
              "document": analyze_document}[kind]
        rep = fn(data, file.filename or "")
        return {"kind": kind, "filename": file.filename,
                "findings": rep["findings"], "visual": rep.get("visual"),
                "notes": rep["notes"]}
    if kind == "text":
        text = data.decode("utf-8", "replace")
        rep = analyze_text(text)
        return {"kind": "text", "filename": file.filename,
                "findings": rep["findings"], "notes": rep["notes"]}
    raise HTTPException(415, "Unsupported file type (not a recognized image or UTF-8 text)")


@app.post("/api/clean")
async def clean(file: UploadFile = File(...), strength: str = Form("standard")):
    data = await file.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "File exceeds 200 MB limit")
    kind = _kind(data, file.filename or "")
    if kind == "text":
        raise HTTPException(415, "Use /api/text/clean for text")
    if kind is None:
        raise HTTPException(415, "Unsupported file type")
    analyzers = {"image": analyze_image, "audio": analyze_audio,
                 "video": analyze_video, "document": analyze_document}
    cleaners = {"image": clean_image, "audio": clean_audio,
                "video": clean_video, "document": clean_document}
    rep = analyzers[kind](data, file.filename or "")
    ctx = {k: rep[k] for k in rep if k.startswith("_")}
    ctx["filename"] = file.filename or ""
    res = cleaners[kind](data, strength, ctx)
    out, out_ext, actions, metrics = res
    after = analyzers[kind](out, "out" + out_ext)
    _purge_work()
    did = uuid.uuid4().hex
    (WORK / did).write_bytes(out)
    (WORK / f"{did}.name").write_text(
        os.path.splitext(file.filename or "file")[0] + ".cleaned" + out_ext)
    if isinstance(metrics, (int, float)):
        metrics = {"psnr_db": round(metrics, 2), "in_bytes": len(data),
                   "out_bytes": len(out)}
    else:
        metrics = dict(metrics)
        metrics.setdefault("in_bytes", len(data))
        metrics.setdefault("out_bytes", len(out))
    return {"download_id": did,
            "filename": os.path.splitext(file.filename or "file")[0] + ".cleaned" + out_ext,
            "actions": actions, "before": rep["findings"], "after": after["findings"],
            "metrics": metrics}


@app.get("/api/download/{did}")
def download(did: str):
    _purge_work()
    if not all(c in "0123456789abcdef" for c in did):
        raise HTTPException(400, "bad id")
    p = WORK / did
    if not p.exists():
        raise HTTPException(404, "expired or unknown download id")
    name_path = WORK / f"{did}.name"
    name = name_path.read_text().strip() if name_path.exists() else "cleaned"
    return FileResponse(p, filename=name)


class TextIn(BaseModel):
    text: str


class TextCleanIn(BaseModel):
    text: str
    options: dict | None = None


@app.post("/api/text/analyze")
def text_analyze(body: TextIn):
    rep = analyze_text(body.text)
    return {"kind": "text", "findings": rep["findings"], "notes": rep["notes"]}


@app.post("/api/text/clean")
def text_clean(body: TextCleanIn):
    out, actions = clean_text(body.text, body.options)
    after = analyze_text(out)
    return {"text": out, "actions": actions,
            "before": analyze_text(body.text)["findings"],
            "after": after["findings"]}


@app.post("/api/text/paraphrase")
async def paraphrase(body: TextIn):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise HTTPException(503, "ANTHROPIC_API_KEY not configured")
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={
                "model": model, "max_tokens": 4096,
                "system": "Rewrite the user's text so that every sentence is "
                          "rephrased with different word choices and sentence "
                          "structure while preserving meaning, tone, formatting "
                          "and length. Output only the rewritten text.",
                "messages": [{"role": "user", "content": body.text}],
            })
    if r.status_code != 200:
        raise HTTPException(502, f"Anthropic API error {r.status_code}")
    parts = r.json().get("content", [])
    return {"text": "".join(p.get("text", "") for p in parts if p.get("type") == "text")}
