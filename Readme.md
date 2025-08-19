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

docker run -it --rm --name segserver --user root -p 5000:5000 -v "C:\Users\Marko\Projekti\openvino-road:/workspace" my-openvino-ffmpeg:2025.2.0  bash
```


### 2. Start the Segmentation Server

By default, the server runs in **pair mode** (processes two video streams in parallel):

```
python seg_demo.py --mode pair
```

To run in **single mode** (process a single video stream, using all CPU cores):

```
python seg_demo.py --mode single
```

You can also override model, host, port, threads, and JPEG quality:

```
python seg_demo.py --mode pair --model <path_to_model.xml> --host 0.0.0.0 --port 5000 --threads-per-worker 4 --jpeg-quality 80
```

### 3. Run the Test Client

For paired video processing (default):

```
python tests/test_socket_client.py --mode pair
```

For single video processing (choose source 1 or 2):

```
python tests/test_socket_client.py --mode single --single-source 1
```

Segmented results will be saved in the `test_results/` directory.

## How It Works


1. **Server (`seg_demo.py`)**
    - Loads the OpenVINO model and starts worker process(es).
    - In **pair mode**, two workers each process one video stream, each pinned to a separate set of CPU cores.
    - In **single mode**, one worker uses all available CPU cores for maximum throughput.
    - Listens for incoming socket connections and processes requests according to the selected mode.
    - Logs detailed per-stage timings (preprocessing, inference, postprocessing, encoding) for each frame.

2. **Client (`test_socket_client.py`)**
    - Reads one or two video files frame-by-frame.
    - Sends frames to the server over a socket, matching the selected mode.
    - Receives segmented results and writes them to output videos.

## Model

The default model is [semantic-segmentation-adas-0001](https://docs.openvino.ai/latest/omz_models_model_semantic_segmentation_adas_0001.html), included in the `intel/` directory in multiple precisions (FP16, FP16-INT8, FP32).


## Customization

- **Change input videos:** Use `--video1` and `--video2` arguments for the client, or edit the defaults in `tests/test_socket_client.py`.
- **Change model:** Use `--model` argument for the server, or edit `MODEL_PATH` in `seg_demo.py`.
- **Tune performance:** Use `--threads-per-worker` and `--jpeg-quality` for the server, or edit the defaults in `seg_demo.py`.
- **Switch between single and pair mode:** Use `--mode single` or `--mode pair` for both server and client.


## Troubleshooting

- **OpenVINO not found:** Make sure OpenVINO is installed and available in your Python environment.
- **Socket connection errors:** Ensure the server is running before starting the client, and that both use the same `--mode`.
- **Video not found:** Check that the paths in `test_videos/` are correct and files exist, or use the `--video1`/`--video2` arguments.
- **Performance:** In single mode, all CPU cores are used for maximum throughput. In pair mode, each worker is pinned to a separate set of cores for balanced parallelism. See server logs for detailed timing breakdowns.

## License

This project is for research and educational purposes. See individual model licenses for usage restrictions.

---
**Contact:** For questions or issues, open an issue on GitHub or contact the project maintainer.