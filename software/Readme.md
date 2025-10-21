docker exec -it <id> bash
# OpenVINO Road Segmentation Demo

This project demonstrates real-time semantic segmentation of road scenes using OpenVINO and a socket-based client-server architecture. It processes two video streams in parallel, segments each frame, and saves the results as new videos.

## Features
- **Fast, parallel inference** using OpenVINO Runtime and multiprocessing
- **Socket server** for efficient frame exchange between client and server
- **Supports paired or single video/camera processing**
- **Multiple payload options:** overlay JPG, raw label maps (ids), or advanced visualization (viz)
- **Optional logging of original, processed, and postprocessed videos**
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

### Optional start notebook

'''
python3 -m notebook --allow-root \
  --ServerApp.ip=0.0.0.0 \
  --ServerApp.port=8888 \
  --ServerApp.root_dir=/workspace \
  --ServerApp.token='' --ServerApp.password=''
'''



## Quick Start

### 1. Build and Run with Docker

```
docker build -t my-openvino-ffmpeg:2025.2.0 .

docker run -it --rm --name segserver --user root -w /workspace -p 5000:5000 -p 8888:8888 -v "C:\Users\Marko\Projekti\openvino-road:/workspace" my-openvino-ffmpeg:2025.2.0  bash
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

#### Additional server options

- `--payload` (jpg | ids | viz): Choose what to send to the client. Default is `jpg` (overlay). `ids` sends raw label maps. `viz` sends advanced visualization (BEV/postprocessed).
- `--log-videos`: Save original, processed, and postprocessed videos to `collected_data/`.
- `--threads-per-worker`, `--jpeg-quality`, `--max-inflight`, `--pair-queue-max`: Tune performance and quality.
- `--blend-alpha`: Set overlay transparency (for jpg payload).

Example:

```
python seg_demo.py --mode pair --payload viz --log-videos --threads-per-worker 4 --jpeg-quality 80
```

### 3. Run the Test Client

#### Video or Camera Input

You can use video files (default) or live camera(s) as input for the test client.

**Paired video processing (default):**

```
python tests/test_socket_client.py --mode pair
```

**Paired camera processing:**

```
python tests/test_socket_client.py --mode pair --camera1 0 --camera2 1
```

**Single video processing (choose source 1 or 2):**

```
python tests/test_socket_client.py --mode single --single-source 1
```

**Single camera processing:**

```
python tests/test_socket_client.py --mode single --camera1 0
```

**Show live results (window):**

Add `--show` to display processed frames in a window (works with camera or video).

**Payload selection:**

Add `--payload` (jpg | ids) to select what the client expects from the server. Must match the server's payload.

**Other options:**
- `--jpeg-quality`: Set JPEG quality for sending frames (default 40).
- `--max-inflight`: Number of frames in flight (default 12).

When using a camera, results are displayed live in a window. When using video files, results are saved in the `test_results/` directory by default.

## How It Works


1. **Server (`seg_demo.py`)**
    - Loads the OpenVINO model and starts worker process(es).
    - In **pair mode**, two workers each process one video stream, each pinned to a separate set of CPU cores.
    - In **single mode**, one worker uses all available CPU cores for maximum throughput.
    - Listens for incoming socket connections and processes requests according to the selected mode.
    - Supports multiple payload types: overlay JPG, raw label maps, or advanced visualization.
    - Optionally logs original, processed, and postprocessed videos.
    - Logs detailed per-stage timings (preprocessing, inference, postprocessing, encoding) for each frame.

2. **Client (`test_socket_client.py`)**
    - Reads one or two video files **or** live camera streams frame-by-frame.
    - Sends frames to the server over a socket, matching the selected mode and payload.
    - Receives segmented results and writes them to output videos (for video input) or displays them live (for camera input).
    - Supports `--show` for live display, and `--payload` to match server output.

## Model

The default model is [semantic-segmentation-adas-0001](https://docs.openvino.ai/latest/omz_models_model_semantic_segmentation_adas_0001.html), included in the `intel/` directory in multiple precisions (FP16, FP16-INT8, FP32).


## Customization
  
- **Change input videos:** Use `--video1` and `--video2` arguments for the client, or edit the defaults in `tests/test_socket_client.py`.
- **Use a camera as input:** Use `--camera1` and/or `--camera2` to select camera indices (e.g., `--camera1 0`).
- **Change model:** Use `--model` argument for the server, or edit `MODEL_PATH` in `seg_demo.py`.
- **Tune performance:** Use `--threads-per-worker`, `--jpeg-quality`, `--max-inflight`, and `--pair-queue-max` for the server, or edit the defaults in `seg_demo.py`.
- **Switch between single and pair mode:** Use `--mode single` or `--mode pair` for both server and client.
- **Select payload type:** Use `--payload` (jpg, ids, viz) for the server and client.
- **Enable video logging:** Use `--log-videos` on the server to save all video streams.

## Troubleshooting

- **OpenVINO not found:** Make sure OpenVINO is installed and available in your Python environment.
- **Socket connection errors:** Ensure the server is running before starting the client, and that both use the same `--mode` and `--payload`.
- **Video not found:** Check that the paths in `test_videos/` are correct and files exist, or use the `--video1`/`--video2` arguments.
- **Performance:** In single mode, all CPU cores are used for maximum throughput. In pair mode, each worker is pinned to a separate set of cores for balanced parallelism. See server logs for detailed timing breakdowns.
- **Live display not working:** If OpenCV is built without GUI support, use `--show` only in environments with display capability, or use `xvfb-run` for headless systems.

## License

This project is for research and educational purposes. See individual model licenses for usage restrictions.

---
**Contact:** For questions or issues, open an issue on GitHub or contact the project maintainer.