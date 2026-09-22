import os
import re
import subprocess
import tempfile

from ..findings import Finding

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}
C2PA_UUID = bytes.fromhex("d8fec3d61b0e483c92975828877ec481")
IDENTITY_KEYS = {"artist", "author", "copyright", "com.apple.quicktime.author",
                 "com.apple.quicktime.artist", "com.apple.quicktime.copyright"}
GPS_KEYS = {"com.apple.quicktime.location.iso6709", "location", "location-eng",
            "com.apple.quicktime.location"}
DEVICE_KEYS = {"com.apple.quicktime.make", "com.apple.quicktime.model",
               "com.apple.quicktime.software", "encoder", "handler_name",
               "com.android.version", "com.android.capture.fps",
               "com.apple.quicktime.hardware"}
ID_KEYS = {"com.apple.quicktime.content.identifier", "uuid", "xid"}
HEXID_RE = re.compile(r"^(?:[0-9a-fA-F]{8}-){1,3}[0-9a-fA-F]{4,}$|^[0-9a-fA-F]{16,}$")


def _probe(path):
    import json
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", path],
            capture_output=True, text=True, timeout=30)
        return json.loads(out.stdout or "{}")
    except Exception:
        return {}


def analyze_video(data: bytes, filename: str = "") -> dict:
    ext = os.path.splitext(filename or "")[1] or ".mp4"
    fd, path = tempfile.mkstemp(suffix=ext)
    findings, notes = [], []
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        probe = _probe(path)
        fmt = probe.get("format", {})
        GENERIC_HANDLERS = {"videohandler", "soundhandler", "datahandler",
                            "subtitlehandler", "coremediavideo", "coremediaaudio"}
        tags = {k.lower(): str(v) for k, v in (fmt.get("tags") or {}).items()
                if not (k.lower() == "handler_name"
                        and str(v).replace(" ", "").lower() in GENERIC_HANDLERS)}
        for s in probe.get("streams", []):
            for k, v in (s.get("tags") or {}).items():
                if (k.lower() == "handler_name"
                        and str(v).replace(" ", "").lower() in GENERIC_HANDLERS):
                    continue
                tags.setdefault(k.lower(), str(v))

        ident = [f"{k}={v[:60]}" for k, v in tags.items() if k in IDENTITY_KEYS]
        gps = [f"{k}={v[:60]}" for k, v in tags.items() if k in GPS_KEYS]
        dev = [f"{k}={v[:60]}" for k, v in tags.items() if k in DEVICE_KEYS]
        tms = [f"{k}={v[:40]}" for k, v in tags.items() if "time" in k]
        uids = [f"{k}={v[:60]}" for k, v in tags.items()
                if k in ID_KEYS or HEXID_RE.match(str(v).strip())]
        xmp = [k for k in tags if "xmp" in k]

        def add(fid, name, sev, conf, ev, rem=True, removal=""):
            findings.append(Finding(id=fid, category="metadata", name=name,
                                    severity=sev, confidence=conf,
                                    evidence=str(ev)[:200], removable=rem,
                                    removal=removal))

        if gps:
            add("location-gps", "Location tags", "high", 1.0, "; ".join(gps),
                removal="Dropped by -map_metadata -1")
        if ident:
            add("owner-identity", "Author/artist tags", "high", 1.0,
                "; ".join(ident), removal="Dropped by -map_metadata -1")
        if dev:
            add("device-tags", "Device/encoder tags", "medium", 1.0,
                "; ".join(dev), removal="Dropped by -map_metadata -1")
        if tms:
            add("timestamps", "Creation timestamps", "low", 1.0, "; ".join(tms),
                removal="Dropped by -map_metadata -1")
        if uids:
            add("unique-ids", "Content/UUID tags", "medium", 1.0,
                "; ".join(uids), removal="Dropped by -map_metadata -1")
        if xmp:
            add("xmp-tags", "XMP tags", "low", 1.0, ", ".join(xmp[:5]),
                removal="Dropped by -map_metadata -1")

        unknown = [s for s in probe.get("streams", [])
                   if s.get("codec_type") in ("data", "subtitle")
                   or s.get("codec_name") in ("tmcd", "mebx", "bin_data")
                   or s.get("codec_tag_string") in ("tmcd", "mebx")]
        if unknown:
            add("unknown-streams", "Data/metadata tracks", "low", 1.0,
                "; ".join(f"{s.get('codec_name','?')} stream #{s.get('index')}"
                          for s in unknown[:6])
                + " (mebx/tmcd tracks can carry per-frame metadata)",
                removal="Only audio+video mapped on clean")

        if C2PA_UUID in data or re.search(rb"c2pa\.[a-z_.]+|c2pa", data[: min(len(data), 4 << 20)]):
            add("c2pa-manifest", "C2PA manifest box", "high", 0.9,
                "C2PA uuid box / manifest bytes found in container",
                removal="Remux drops uuid boxes", )
            findings[-1].category = "provenance"

        dur = float(fmt.get("duration") or 0)
    finally:
        os.unlink(path)
    return {"findings": findings, "visual": None, "notes": notes,
            "_duration": dur, "_n_tags": len(tags)}
