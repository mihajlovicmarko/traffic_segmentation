FROM openvino/ubuntu22_dev:2025.2.0

# become root to install packages
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg libgl1 libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
      opencv-python-headless \
      numpy \
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
      tqdm

# drop back to the openvino user
USER openvino
