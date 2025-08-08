FROM openvino/ubuntu22_dev:2025.2.0

# become root to install packages
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg libgl1 libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# drop back to the openvino user
USER openvino
