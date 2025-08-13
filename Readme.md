docker exec -it <id> bash
# OpenVINO Road Segmentation Demo

This project demonstrates real-time semantic segmentation of road scenes using OpenVINO and a socket-based client-server architecture. It processes two video streams in parallel, segments each frame, and saves the results as new videos.

## Features
- **Fast, parallel inference** using OpenVINO Runtime and multiprocessing
- **Socket server** for efficient frame exchange between client and server
- **Supports paired video processing** (e.g., stereo or multi-camera)
- **Easy integration** with your own video sources

## Project Structure

```
├── Dockerfile
├── seg_demo.py                # Segmentation server (OpenVINO, socket server)
├── tests/
│   └── test_socket_client.py  # Example client (sends video frames, receives segmented results)
├── intel/
│   └── semantic-segmentation-adas-0001/  # Pretrained model files (FP16, FP16-INT8, FP32)
├── test_videos/               # Example input videos
├── test_results/              # Output segmented videos
```

## Requirements

- Python 3.8+
- OpenVINO Runtime (2023.0+ recommended)
- OpenCV (`opencv-python`)
- NumPy

> **Tip:** The provided `Dockerfile` can be used to build a ready-to-run environment.

## Quick Start

### 1. Build and Run with Docker

```
docker build -t my-openvino-ffmpeg:2025.2.0 .
docker run -it --rm --user root -v C:\Users\Marko\Projekti\openvino-road:/workspace my-openvino-ffmpeg:2025.2.0 bash
```

### 2. Start the Segmentation Server

```
python seg_demo.py
```

### 3. Run the Test Client

```
python tests/test_socket_client.py
```

Segmented results will be saved in the `test_results/` directory.

## How It Works

1. **Server (`seg_demo.py`)**
    - Loads the OpenVINO model and starts two worker processes.
    - Listens for incoming socket connections.
    - For each pair of frames received, dispatches them to workers, collects results, and sends back segmented images.

2. **Client (`test_socket_client.py`)**
    - Reads two video files frame-by-frame.
    - Sends paired frames to the server over a socket.
    - Receives segmented results and writes them to output videos.

## Model

The default model is [semantic-segmentation-adas-0001](https://docs.openvino.ai/latest/omz_models_model_semantic_segmentation_adas_0001.html), included in the `intel/` directory in multiple precisions (FP16, FP16-INT8, FP32).

## Customization

- **Change input videos:** Edit `VIDEO_PATH_1` and `VIDEO_PATH_2` in `tests/test_socket_client.py`.
- **Change model:** Update `MODEL_PATH` in `seg_demo.py` to use a different model or precision.
- **Tune performance:** Adjust `INFERENCE_THREADS_PER_WORKER`, `JPEG_QUALITY`, and queue sizes in `seg_demo.py`.

## Troubleshooting

- **OpenVINO not found:** Make sure OpenVINO is installed and available in your Python environment.
- **Socket connection errors:** Ensure the server is running before starting the client.
- **Video not found:** Check that the paths in `test_videos/` are correct and files exist.

## License

This project is for research and educational purposes. See individual model licenses for usage restrictions.

---
**Contact:** For questions or issues, open an issue on GitHub or contact the project maintainer.