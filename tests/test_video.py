import os
import subprocess
import tempfile

from app.detect.video import analyze_video
from app.clean.video import clean_video


def make_mp4():
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
         "testsrc=duration=2:size=160x120:rate=10",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-metadata", "com.apple.quicktime.location.ISO6709=+37.7749-122.4194/",
         "-metadata", "artist=Jane",
         "-metadata", "creation_time=2026-01-01T00:00:00Z",
         path], check=True, capture_output=True)
    data = open(path, "rb").read()
    os.unlink(path)
    return data


def ids(rep):
    return {f.id for f in rep["findings"]}


def test_video_detect_clean():
    data = make_mp4()
    rep = analyze_video(data, "clip.mp4")
    fids = ids(rep)
    assert "location-gps" in fids or "owner-identity" in fids, fids
    assert "owner-identity" in fids
    out, ext, actions, metrics = clean_video(data, "light", {"filename": "clip.mp4"})
    assert ext == ".mp4"
    after = analyze_video(out, "clip.mp4")
    assert not {"location-gps", "owner-identity", "timestamps",
                "device-tags"} & ids(after)
