# Base: OpenVINO Dev 2025.2.0
FROM openvino/ubuntu22_dev:2025.2.0

# ------------------------------------------------------------
# System deps
# ------------------------------------------------------------
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      libgl1 \
      libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Pip behavior + compatibility pins
# ------------------------------------------------------------
# CHANGED: Bevor etwas installiert wird, erzwingen wir Binär‑Wheels
#          und pinnen NumPy < 2, damit OpenVINO's gebundeltes OpenCV
#          (cv2) mit NumPy 1.x kompatibel bleibt.
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_PREFER_BINARY=1

# CHANGED: pip upgraden, NumPy < 2 zuerst installieren
RUN python3 -m pip install --upgrade pip && \
    pip install --no-cache-dir "numpy<2"

# ------------------------------------------------------------
# Python packages
# ------------------------------------------------------------
# CHANGED: opencv-python entfernt (Konflikt mit OpenVINO's cv2)
#          Rest bleibt; sie funktionieren mit NumPy 1.26.x.
RUN pip install --no-cache-dir \
      requests \
      Pillow \
      matplotlib \
      jupyterlab \
      notebook \
      ipywidgets \
      pandas \
      scikit-learn \
      scikit-image \
      seaborn \
      serial \
      tqdm && \
    pip check

# ------------------------------------------------------------
# Default user & working dir
# ------------------------------------------------------------
USER openvino

# ADDED: Standard‑Arbeitsverzeichnis setzen, damit du beim Start
#        automatisch in /workspace landest (harmoniert mit -v ...:/workspace)
WORKDIR /workspace
