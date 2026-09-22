import os
import subprocess
import tempfile

from ..detect.video import _probe

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}


def _run(cmd, timeout=600):
    r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    return r.returncode == 0


def clean_video(data: bytes, strength: str = "standard", ctx: dict | None = None):
    ctx = ctx or {}
    ext_in = os.path.splitext(ctx.get("filename") or "x.mp4")[1].lower() or ".mp4"
    fd, inp = tempfile.mkstemp(suffix=ext_in)
    actions = []
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        probe = _probe(inp)
        dur = float(probe.get("format", {}).get("duration") or 0)
        strength = strength if strength in ("light", "standard", "heavy") else "standard"
        if dur > 600 and strength == "standard":
            strength = "light"
            actions.append("Duration >10 min — applying light only (metadata strip + remux)")

        keep_container = ext_in in (".mkv", ".webm")
        out_ext = ext_in if keep_container else ".mp4"
        if ext_in == ".webm" and strength != "light":
            out_ext = ".webm"
        out_fmt = "matroska" if ext_in in (".mkv", ".webm") else "mp4"

        fd, outp = tempfile.mkstemp(suffix=out_ext)
        os.close(fd)
        try:
            if strength == "light":
                cmd = ["ffmpeg", "-v", "error", "-y", "-i", inp,
                       "-map", "0:v", "-map", "0:a?",
                       "-map_metadata", "-1", "-map_metadata:s", "-1",
                       "-c", "copy", "-fflags", "+bitexact"]
                if out_fmt == "mp4":
                    cmd += ["-movflags", "+faststart"]
                cmd += ["-f", out_fmt, outp]
                ok = _run(cmd, 300)
                actions.append("Remuxed with all metadata tracks/tags stripped (-map_metadata -1, bitexact)")
            else:
                crf = "20" if strength == "standard" else "23"
                scale = 0.98 if strength == "standard" else 0.95
                noise = 3 if strength == "standard" else 6
                vf = (f"scale=trunc(iw*{scale}/2)*2:trunc(ih*{scale}/2)*2,"
                      f"noise=alls={noise}:allf=t")
                if strength == "heavy":
                    vf = "rotate=0.3*PI/180:ow=iw:oh=ih:c=none," + vf
                if ext_in == ".webm":
                    vcodec = ["-c:v", "libvpx-vp9", "-crf", "32", "-b:v", "0"]
                else:
                    vcodec = ["-c:v", "libx264", "-crf", crf, "-preset", "medium",
                              "-pix_fmt", "yuv420p"]
                acodec = ["-c:a", "aac", "-b:a", "192k"] if any(
                    s.get("codec_type") == "audio" for s in probe.get("streams", [])) else ["-an"]
                cmd = ["ffmpeg", "-v", "error", "-y", "-i", inp,
                       "-map", "0:v", "-map", "0:a?",
                       "-vf", vf] + vcodec + acodec + [
                       "-map_metadata", "-1", "-map_metadata:s", "-1",
                       "-fflags", "+bitexact"]
                if out_fmt == "mp4":
                    cmd += ["-movflags", "+faststart"]
                cmd += [outp]
                ok = _run(cmd, 600)
                actions.append(f"Re-encoded {out_ext} (scale {scale}, noise {noise}, "
                               f"{'vp9' if ext_in=='.webm' else 'h264 crf ' + crf})")
                if strength == "heavy":
                    actions.append("Applied 0.3° rotate + 95% scale + heavier noise")
            if not ok or not os.path.getsize(outp):
                raise ValueError("ffmpeg processing failed")
            out = open(outp, "rb").read()
            metrics = {"duration_s": round(dur, 2), "in_bytes": len(data),
                       "out_bytes": len(out)}
            return out, out_ext, actions, metrics
        finally:
            try:
                os.unlink(outp)
            except OSError:
                pass
    finally:
        os.unlink(inp)
