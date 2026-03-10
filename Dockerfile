FROM python:3.11-slim

# System libraries required by OpenCV and MediaPipe on headless Linux.
# ffmpeg is needed by OpenCV to decode the video files uploaded for analysis.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies before copying source so Docker can cache this layer.
COPY requirements.prod.txt .
RUN pip install --no-cache-dir -r requirements.prod.txt

# Copy source.
COPY . .

# Pre-download the MediaPipe pose landmarker model so the first analysis
# request doesn't have to wait for a ~25 MB download.
RUN python - <<'EOF'
import urllib.request
from pathlib import Path
url  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
dest = Path("movementscreen/pose/models/pose_landmarker_lite.task")
dest.parent.mkdir(parents=True, exist_ok=True)
if not dest.exists():
    print(f"Downloading {dest.name} ...")
    urllib.request.urlretrieve(url, dest)
    print("Done.")
EOF

COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 10000

CMD ["/app/start.sh"]
