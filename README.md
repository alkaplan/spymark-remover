# Spymark Remover

Local web app that finds and removes hidden tracking signals in files. Runs
entirely on your machine; files never leave it.

A "spymark" is any hidden signal embedded in a file that identifies you, your
device, or where the file came from — camera GPS/serial metadata, C2PA
provenance manifests, invisible pixel watermarks, ultrasonic audio beacons,
printer tracking dots, Unicode text steganography, purchase-account atoms in
media tags, and document revision/session IDs. See
https://brand.io/article/spymarks/ for background.

## Run

```
./run.sh          # creates .venv, installs requirements, serves on :8765
```

Then open http://127.0.0.1:8765.

Requirements: Python 3.10+, `ffmpeg`, `exiftool`
(`sudo apt-get install -y ffmpeg libimage-exiftool-perl`).
Note: `torch`/`torchaudio` install as CPU wheels via
`--extra-index-url https://download.pytorch.org/whl/cpu` (already in run.sh);
first install is ~1.7 GB. AudioSeal/WavMark model weights download from
huggingface.co on first use.

Optional: `ANTHROPIC_API_KEY` (and `ANTHROPIC_MODEL`, default
`claude-sonnet-4-5`) enables the "Paraphrase with Claude" text feature.

## Deploy (Fly.io)

`fly.toml` and `Dockerfile` are included — just run:

```
fly deploy
```

The image bakes the AudioSeal/WavMark model weights in at build time, so cold
starts don't hit Hugging Face. Set `ANTHROPIC_API_KEY` via `fly secrets set`
to enable paraphrase.

## Supported kinds

| Kind | Detects | Removes |
|---|---|---|
| Image (png, jpg, webp, tiff, bmp, gif, avif, heic) | EXIF/XMP/IPTC/PNG metadata (GPS, serials, owner, IDs, AI-generator labels, TC260 AIGC), C2PA manifests (+remote refs, soft-binding pointers), Stable Diffusion/SDXL DWT-DCT watermark, periodic carrier signals, printer tracking dots | All metadata stripped; elastic warp + affine + resize-squeeze + color nudge + FFT notch + noise + JPEG recompress (strength-scaled); dots inpainted; SD watermark verified and re-attacked |
| Audio (wav, flac, mp3, ogg, opus, m4a, aac, aiff) | ID3/Vorbis/atom tags incl. purchase identity (PRIV/UFID/apID…), cover art, AudioSeal, WavMark, persistent ultrasonic tones, periodic spectral patterns | `-map_metadata -1` remux; resample, time-stretch, pitch-shift, STFT phase jitter, shaped noise, low-pass; AudioSeal re-verified with auto-escalation |
| Document (pdf, docx, xlsx, pptx) | docinfo/XMP/trailer ID, embedded files, JavaScript actions, tracked remote links, invisible text, /PieceInfo, per-page metadata; core/app/custom props, rsids/docIds, people, comments/tracked changes, thumbnails, external rels, macros, fonts | docinfo/XMP/PieceInfo/page-metadata deleted, fresh trailer ID, JS actions removed, URI queries stripped (heavy); core.xml cleared, custom.xml/people.xml/thumbnail dropped, rsids stripped, comment authors neutralized, external rels removed (standard+) |
| Video (mp4, mov, m4v, mkv, webm, avi) | Format/stream tags (location, artist, device, creation_time, UUIDs), C2PA uuid boxes, data/metadata tracks (tmcd, mebx) | Metadata-stripped remux (light) or re-encode with scale+noise (+rotate on heavy); only A/V streams kept |
| Text | Zero-width & tag characters, bidi controls, variation selectors, unusual spaces, soft hyphens, homoglyphs, private-use chars, whitespace patterns, typographic fingerprints, invisible HTML | Selectable normalization options; emoji-safe ZWJ/VS16 handling; optional Claude paraphrase |

## Honest limits

- Deep-learned watermarks (Google SynthID, Meta Stable Signature,
  audiowmark-keyed marks) have no public detector. Standard/Heavy apply the
  perturbations published research shows degrade them, but removal cannot be
  verified locally.
- Statistical text watermarks (e.g. SynthID-Text) can only be defeated by
  rewriting — hence the paraphrase feature.
- PDF invisible text runs and OOXML macros are reported, not removed.

## Tests

```
.venv/bin/pytest -q tests
```

## Techniques & references

- SynthID-Image (arXiv:2510.09263); reverse-SynthID & remove-ai-watermarks
  community repos; `invisible-watermark` (SDXL `dwtDct` payload)
- AudioSeal & WavMark (facebookresearch / wavmark on Hugging Face)
- EFF printer tracking-dot research
- Wen et al. 2025, audio watermarking SoK
