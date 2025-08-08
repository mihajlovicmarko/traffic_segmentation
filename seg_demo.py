import cv2
import numpy as np
from openvino.runtime import Core
import os
import time

# ===== Settings =====
WRITE_OUTPUT = True
MODEL_PATH = "intel/semantic-segmentation-adas-0001/FP16-INT8/semantic-segmentation-adas-0001.xml"  # fastest
VIDEO_PATH = os.path.join(os.path.dirname(__file__), "test", "test_video_4.mp4")
NUM_THREADS = os.cpu_count() or 8    # use whatever Docker/WSL exposes

# ===== OpenVINO: load & compile =====
core = Core()
model = core.read_model(MODEL_PATH)
compiled_model = core.compile_model(
    model,
    "CPU",
    {
        "PERFORMANCE_HINT": "LATENCY",   # single-stream, best per-frame time
        "NUM_STREAMS": "1",              # avoid oversubscription
        "INFERENCE_NUM_THREADS": NUM_THREADS,
    },
)
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)
# model input is NCHW: [1,3,1024,2048]
in_h, in_w = int(input_layer.shape[2]), int(input_layer.shape[3])

# ===== Video IO =====
cap = cv2.VideoCapture(VIDEO_PATH, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise RuntimeError(
        f"Video exists={os.path.exists(VIDEO_PATH)} but cannot be opened. "
        f"Ensure ffmpeg is installed in the image and using CAP_FFMPEG. Path: {VIDEO_PATH}"
    )

src_fps = cap.get(cv2.CAP_PROP_FPS)
if not src_fps or np.isnan(src_fps) or src_fps <= 1e-3:
    src_fps = 25.0  # sensible default
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

if WRITE_OUTPUT:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("segmented_video_1.mp4", fourcc, src_fps, (width, height))

# ===== Colormap (256 safe) =====
np.random.seed(42)
color_map = np.random.randint(0, 255, size=(256, 3), dtype=np.uint8)

# ===== Warmup (helps JIT/thread pools) =====
dummy = np.zeros((in_h, in_w, 3), dtype=np.uint8)
dummy_blob = dummy.transpose(2, 0, 1)[None, ...]
_ = compiled_model([dummy_blob])[output_layer]

# ===== Loop =====
total_time = 0.0
frame_idx = 0

while True:
    frame_start = time.time()

    # --- Read ---
    t0 = time.time()
    ret, frame = cap.read()
    t_read = time.time() - t0
    if not ret:
        break
    frame_idx += 1

    # --- Preprocess: resize exactly to model input (once) & NCHW uint8 ---
    t0 = time.time()
    if frame.shape[1] != in_w or frame.shape[0] != in_h:
        frame_resized = cv2.resize(frame, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    else:
        frame_resized = frame
    blob = frame_resized.transpose(2, 0, 1)[None, ...].astype(np.uint8)
    t_pre = time.time() - t0

    # --- Inference ---
    t0 = time.time()
    result = compiled_model([blob])[output_layer]
    t_infer = time.time() - t0

    # --- Postprocess (labels -> colors, resize back to original) ---
    t0 = time.time()
    seg_map = result.squeeze().astype(np.uint8)        # (H,W) == (1024,2048)
    seg_overlay = color_map[seg_map]                   # (H,W,3)
    seg_overlay_resized = cv2.resize(seg_overlay, (width, height), interpolation=cv2.INTER_NEAREST)
    t_post = time.time() - t0

    # --- Blend & write ---
    t0 = time.time()
    blended = cv2.addWeighted(frame, 0.5, seg_overlay_resized, 0.5, 0)
    t_blend = time.time() - t0

    t0 = time.time()
    if WRITE_OUTPUT:
        out.write(blended)
    t_write = time.time() - t0

    frame_time = time.time() - frame_start
    total_time += frame_time

    if frame_idx % 10 == 0 or frame_idx == 1:
        fps_now = 1.0 / frame_time if frame_time > 0 else float("inf")
        print(f"Frame {frame_idx}/{total_frames or '?'} - Total: {frame_time:.3f}s ({fps_now:.2f} FPS)")
        print(f"    Read:   {t_read:.3f}s")
        print(f"    Pre:    {t_pre:.3f}s   (resize+NCHW)")
        print(f"    Infer:  {t_infer:.3f}s  <-- bottleneck")
        print(f"    Post:   {t_post:.3f}s")
        print(f"    Blend:  {t_blend:.3f}s")
        print(f"    Write:  {t_write:.3f}s")

cap.release()
if WRITE_OUTPUT:
    out.release()

avg_fps = frame_idx / total_time if total_time > 0 else 0.0
print("✅ Processed video saved as segmented_video.mp4" if WRITE_OUTPUT else "✅ Finished (no output saved)")
print(f"Average FPS: {avg_fps:.2f}")
