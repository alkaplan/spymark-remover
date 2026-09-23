FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libimage-exiftool-perl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# bake AudioSeal + WavMark weights into the image (torch.hub → TORCH_HOME,
# huggingface_hub → HF_HOME); audioseal loads via torch.hub under TORCH_HOME
ENV HF_HOME=/models TORCH_HOME=/models NO_TORCH_COMPILE=1
RUN python - <<'PY'
from audioseal import AudioSeal
AudioSeal.load_detector("audioseal_detector_16bits")
AudioSeal.load_generator("audioseal_wm_16bits")
import wavmark
wavmark.load_model()
PY

COPY app ./app
ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
